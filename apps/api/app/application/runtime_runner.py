from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import threading

from sqlalchemy.orm import Session, sessionmaker

from app.application.runtime_state import AnalysisRunCancelled, RuntimeStateStore
from app.db.session import get_engine, init_db


class RuntimeRunner:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] | None = None,
        max_workers: int = 2,
        heartbeat_seconds: float = 5.0,
        state_store: RuntimeStateStore | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sales-agent-runtime",
        )
        self._heartbeat_seconds = heartbeat_seconds
        self._state = state_store or RuntimeStateStore()
        self._lock = threading.Lock()
        self._active_task_ids: set[str] = set()

    def _create_worker_session(self) -> Session:
        if self._session_factory is not None:
            return self._session_factory()
        init_db()
        factory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
        return factory()

    def submit(
        self,
        task_id: str,
        *,
        llm_profile: str | None = None,
        output_language: str | None = None,
    ) -> bool:
        with self._lock:
            if task_id in self._active_task_ids:
                return False
            self._active_task_ids.add(task_id)
        self._executor.submit(
            self._run_worker,
            task_id,
            llm_profile=llm_profile,
            output_language=output_language,
        )
        return True

    def _run_worker(
        self,
        task_id: str,
        *,
        llm_profile: str | None,
        output_language: str | None,
    ) -> None:
        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(task_id, stop_heartbeat),
            daemon=True,
        )
        session = self._create_worker_session()
        heartbeat_thread.start()
        try:
            from app.application.analysis_run_service import AnalysisRunService

            AnalysisRunService(session, runtime_state=self._state).run(
                task_id,
                llm_profile=llm_profile,
                output_language=output_language,
            )
            self._state.mark_completed(task_id)
        except AnalysisRunCancelled:
            self._state.mark_cancelled(task_id)
        except Exception as exc:
            self._state.mark_failed(task_id, exc)
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1.0)
            session.close()
            with self._lock:
                self._active_task_ids.discard(task_id)

    def _heartbeat_loop(self, task_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._state.heartbeat(task_id)
            except Exception:
                pass
            stop_event.wait(self._heartbeat_seconds)


_RUNTIME_RUNNER: RuntimeRunner | None = None
_RUNTIME_RUNNER_LOCK = threading.Lock()


def get_runtime_runner() -> RuntimeRunner:
    global _RUNTIME_RUNNER
    with _RUNTIME_RUNNER_LOCK:
        if _RUNTIME_RUNNER is None:
            _RUNTIME_RUNNER = RuntimeRunner()
        return _RUNTIME_RUNNER
