from __future__ import annotations

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.report import AnalysisReport
from app.services.evidence_guard import downgrade_strong_inference_claims
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.output_language import user_facing_language_instruction


def generate_report_summary(report: AnalysisReport, llm_client=None) -> AnalysisReport:
    summarized_report, _ = generate_report_summary_with_trace(report, llm_client=llm_client)
    return summarized_report


def generate_report_summary_with_trace(
    report: AnalysisReport,
    llm_client=None,
    evidence_pack: dict[str, object] | None = None,
    output_language: str | None = None,
) -> tuple[AnalysisReport, LLMStageTrace]:
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return report, build_llm_stage_trace(
            stage="report_summary",
            llm_client=llm_client,
            status="disabled",
            reason="LLM is disabled, so the report summary used deterministic module findings.",
        )

    try:
        summary = llm_client.summarize_report(
            report,
            evidence_pack=evidence_pack,
            language_instruction=user_facing_language_instruction(output_language),
        )
    except Exception as exc:
        return report, build_llm_stage_trace(
            stage="report_summary",
            llm_client=llm_client,
            status="fallback_on_error",
            reason=f"Report summarization failed; kept the deterministic summary. {describe_llm_error(exc)}",
            attempted=True,
        )

    if not summary:
        return report, build_llm_stage_trace(
            stage="report_summary",
            llm_client=llm_client,
            status="llm_no_change",
            reason="Report summarization returned no usable content, so the deterministic summary was kept.",
            attempted=True,
        )

    guarded_summary = [downgrade_strong_inference_claims(item) for item in summary]
    return report.model_copy(update={"summary": guarded_summary}), build_llm_stage_trace(
        stage="report_summary",
        llm_client=llm_client,
        status="llm_applied",
        reason="Report summary bullets were replaced with the LLM-authored summary.",
        attempted=True,
        applied=True,
    )
