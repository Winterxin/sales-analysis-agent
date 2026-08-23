from __future__ import annotations

from pathlib import Path

from app.analysis.runner import load_analysis_frames
from app.schemas.report import ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_agent.tool_calls import AnalysisToolCall
from app.services.analysis_tools import AnalysisToolRegistry


def execute_analysis_tools(
    tool_calls: list[AnalysisToolCall],
    *,
    csv_path: Path,
    schema_mapping: SchemaMapping,
    registry: AnalysisToolRegistry,
) -> tuple[list[ModuleReport], dict[str, str], list[AnalysisToolCall]]:
    raw_frame, clean_frame, canonical_columns = load_analysis_frames(
        csv_path, schema_mapping
    )
    reports: list[ModuleReport] = []
    errors: dict[str, str] = {}
    successful_calls: list[AnalysisToolCall] = []
    for call in tool_calls:
        tool_name = call.tool_name
        try:
            frame = raw_frame if tool_name == "loss_risk_modeling" else clean_frame
            result = registry.execute(
                tool_name, frame, canonical_columns, arguments=call.arguments
            )
            reports.append(ModuleReport.model_validate(result.model_dump()))
            successful_calls.append(call)
        except Exception as exc:
            errors[tool_name] = f"{type(exc).__name__}: {exc}"
    return reports, errors, successful_calls
