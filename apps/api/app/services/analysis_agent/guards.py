from __future__ import annotations

from dataclasses import dataclass, field

from app.services.analysis_tools import AnalysisToolRegistry


@dataclass(frozen=True)
class PlanValidationResult:
    accepted_tools: list[str] = field(default_factory=list)
    rejected_tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_selected_tools(
    selected_tools: list[str],
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
            rejected_tools=list(selected_tools), errors=["budget_exhausted"]
        )

    accepted: list[str] = []
    errors: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for name in selected_tools:
        if name in seen:
            rejected.append(name)
            errors.append(f"duplicate_in_plan:{name}")
            continue
        seen.add(name)
        spec = registry.get(name)
        if spec is None:
            rejected.append(name)
            errors.append(f"unknown_tool:{name}")
            continue
        missing = spec.missing_fields(available_fields, dataset_profile)
        if missing:
            rejected.append(name)
            errors.append(f"missing_required_fields:{name}:{','.join(missing)}")
            continue
        if name in executed_successfully:
            rejected.append(name)
            errors.append(f"duplicate_executed_tool:{name}")
            continue
        if len(accepted) >= max_tools_per_round:
            rejected.append(name)
            errors.append(f"max_tools_per_round_exceeded:{name}")
            continue
        accepted.append(name)
    return PlanValidationResult(
        accepted_tools=accepted,
        rejected_tools=rejected,
        errors=errors,
    )
