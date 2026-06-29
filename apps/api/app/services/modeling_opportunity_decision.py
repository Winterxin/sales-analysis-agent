from __future__ import annotations

from typing import Any

from app.schemas.llm_trace import LLMStageTrace
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.output_language import user_facing_language_instruction


STAGE = "modeling_opportunity_decision"
DECISION_KEYS = {"decision_summary", "notebook_message", "risk_warning"}
FORBIDDEN_PHRASES = (
    "已经训练",
    "已训练",
    "训练了一个新",
    "自动排序",
    "自动拦截",
    "自动审批",
    "自动拒单",
)


def _trace(
    *,
    llm_client,
    status: str,
    reason: str,
    attempted: bool = False,
    applied: bool = False,
) -> LLMStageTrace:
    return build_llm_stage_trace(
        stage=STAGE,
        llm_client=llm_client,
        status=status,
        reason=reason,
        attempted=attempted,
        applied=applied,
    )


def _candidate_tasks(plan: dict[str, Any]) -> list[str]:
    return [
        str(task.get("task_type"))
        for task in plan.get("available_tasks", [])
        if isinstance(task, dict) and task.get("status") in {"candidate", "runnable"}
    ]


def _deterministic_decision(plan: dict[str, Any]) -> dict[str, str]:
    recommended = str(plan.get("recommended_modeling_task") or "no_modeling_descriptive_only")
    decision_status = str(plan.get("decision_status") or "not_recommended")
    notebook_message = str(plan.get("notebook_message") or "").strip()
    candidates = ", ".join(_candidate_tasks(plan)) or "暂无稳定候选任务"
    hard_gates = ", ".join(str(item) for item in plan.get("hard_gate_reasons", []) or []) or "无"
    if not notebook_message:
        notebook_message = "当前数据未形成稳定建模目标，本轮保留描述性分析。"
    return {
        "decision_summary": (
            f"当前推荐建模方向为 {recommended}，决策状态为 {decision_status}。"
            f"候选任务包括：{candidates}；硬性限制：{hard_gates}。"
        ),
        "notebook_message": notebook_message,
        "risk_warning": "G6-2 只记录建模机会，不训练新模型，不生成自动决策，也不把弱模型写成强业务结论。",
    }


def _safe_text(value: Any, fallback: str, limit: int = 220) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = " ".join(text.split())
    return text[:limit]


def _has_forbidden_content(payload: dict[str, str]) -> bool:
    if any(phrase in value for value in payload.values() for phrase in FORBIDDEN_PHRASES):
        return True
    return any("自动决策" in value and "不用于自动决策" not in value for value in payload.values())


def _sanitize(raw: dict[str, Any], fallback: dict[str, str]) -> dict[str, str] | None:
    sanitized = {
        "decision_summary": _safe_text(raw.get("decision_summary"), fallback["decision_summary"], limit=240),
        "notebook_message": _safe_text(raw.get("notebook_message"), fallback["notebook_message"], limit=220),
        "risk_warning": _safe_text(raw.get("risk_warning"), fallback["risk_warning"], limit=180),
    }
    if _has_forbidden_content(sanitized):
        return None
    if "不训练新模型" not in sanitized["risk_warning"]:
        sanitized["risk_warning"] = f"{sanitized['risk_warning']} G6-2 不训练新模型。"
    return sanitized


def _payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Explain a deterministic modeling opportunity plan for a sales analysis notebook.",
        "recommended_modeling_task": plan.get("recommended_modeling_task"),
        "decision_status": plan.get("decision_status"),
        "should_train_model": plan.get("should_train_model"),
        "risk_level": plan.get("risk_level"),
        "hard_gate_reasons": plan.get("hard_gate_reasons"),
        "available_tasks": plan.get("available_tasks"),
        "deterministic_notebook_message": plan.get("notebook_message"),
    }


def build_modeling_opportunity_decision_with_trace(
    plan: dict[str, Any],
    llm_client=None,
    output_language: str | None = None,
) -> tuple[dict[str, str], LLMStageTrace]:
    fallback = _deterministic_decision(plan)
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return fallback, _trace(
            llm_client=llm_client,
            status="disabled",
            reason="LLM is disabled, so modeling opportunity decision used deterministic text.",
        )
    suggest = getattr(llm_client, "suggest_modeling_opportunity_decision", None)
    if suggest is None:
        return fallback, _trace(
            llm_client=llm_client,
            status="skipped",
            reason="LLM client does not expose suggest_modeling_opportunity_decision.",
        )
    try:
        raw = suggest(
            _payload(plan),
            language_instruction=user_facing_language_instruction(output_language),
        )
    except Exception as exc:
        return fallback, _trace(
            llm_client=llm_client,
            status="fallback_on_error",
            reason=f"Modeling opportunity decision failed; used deterministic text. {describe_llm_error(exc)}",
            attempted=True,
        )
    sanitized = _sanitize(raw if isinstance(raw, dict) else {}, fallback)
    if sanitized is None:
        return fallback, _trace(
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason="Modeling opportunity decision contained unsafe or out-of-scope language.",
            attempted=True,
        )
    return sanitized, _trace(
        llm_client=llm_client,
        status="llm_applied",
        reason="LLM modeling opportunity decision was applied.",
        attempted=True,
        applied=True,
    )
