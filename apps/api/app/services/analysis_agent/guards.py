from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from app.services.analysis_agent.tool_calls import (
    AnalysisToolCall,
    normalized_call_signature,
)
from app.services.analysis_tools import AnalysisToolRegistry


@dataclass(frozen=True)
class PlanValidationResult:
    accepted_tool_calls: list[AnalysisToolCall] = field(default_factory=list)
    rejected_tool_calls: list[AnalysisToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def accepted_tools(self) -> list[str]:
        return [call.tool_name for call in self.accepted_tool_calls]

    @property
    def rejected_tools(self) -> list[str]:
        return [call.tool_name for call in self.rejected_tool_calls]


def validate_tool_calls(
    tool_calls: list[AnalysisToolCall],
    *,
    registry: AnalysisToolRegistry,
    available_fields: set[str],
    dataset_profile: dict[str, object],
    executed_successfully: set[str],
    max_tools_per_round: int,
    has_budget: bool,
) -> PlanValidationResult:
    if not has_budget:
        return PlanValidationResult(
            rejected_tool_calls=list(tool_calls), errors=["budget_exhausted"]
        )

    accepted: list[AnalysisToolCall] = []
    errors: list[str] = []
    rejected: list[AnalysisToolCall] = []
    seen: set[str] = set()
    for raw_call in tool_calls:
        name = raw_call.tool_name
        spec = registry.get(name)
        if spec is None:
            rejected.append(raw_call)
            errors.append(f"unknown_tool:{name}")
            continue
        try:
            arguments = spec.argument_model.model_validate(raw_call.arguments).model_dump(
                mode="json"
            )
        except ValidationError as exc:
            rejected.append(raw_call)
            details = ";".join(
                f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
                for item in exc.errors()
            )
            errors.append(f"invalid_arguments:{name}:{details}")
            continue
        call = AnalysisToolCall(tool_name=name, arguments=arguments)
        signature = normalized_call_signature(name, arguments)
        if signature in seen:
            rejected.append(call)
            errors.append(f"duplicate_in_plan:{name}:{signature}")
            continue
        seen.add(signature)
        missing = spec.missing_fields(available_fields, dataset_profile)
        if missing:
            rejected.append(call)
            errors.append(f"missing_required_fields:{name}:{','.join(missing)}")
            continue
        if name == "dimension_breakdown_analysis":
            dimension = str(arguments.get("dimension") or "")
            metric = str(arguments.get("metric") or "sales_amount")
            unsupported = [
                f"unsupported_dataset_argument:{name}:dimension:{dimension}"
                for _ in [0]
                if dimension not in available_fields
            ]
            unsupported.extend(
                f"unsupported_dataset_argument:{name}:metric:{metric}"
                for _ in [0]
                if metric not in available_fields
            )
            if unsupported:
                rejected.append(call)
                errors.extend(unsupported)
                continue
        if signature in executed_successfully:
            rejected.append(call)
            errors.append(f"duplicate_executed_tool_call:{name}:{signature}")
            continue
        if not spec.allow_distinct_argument_calls and any(
            item.startswith(f"{name}:") for item in executed_successfully
        ):
            rejected.append(call)
            errors.append(f"duplicate_executed_tool:{name}")
            continue
        if len(accepted) >= max_tools_per_round:
            rejected.append(call)
            errors.append(f"max_tools_per_round_exceeded:{name}")
            continue
        accepted.append(call)
    return PlanValidationResult(
        accepted_tool_calls=accepted,
        rejected_tool_calls=rejected,
        errors=errors,
    )
