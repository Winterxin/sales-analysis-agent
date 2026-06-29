from __future__ import annotations

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.schema_rules import map_schema_fields


def _should_use_llm(rule_mapping: SchemaMapping) -> bool:
    return (
        rule_mapping.dataset_type == "unknown"
        or bool(rule_mapping.missing_required_fields)
        or rule_mapping.confidence < 0.75
    )


def _merge_schema_mappings(
    rule_mapping: SchemaMapping, llm_mapping: SchemaMapping
) -> SchemaMapping:
    merged_fields = {**llm_mapping.field_mapping, **rule_mapping.field_mapping}
    missing_required_fields = sorted(
        {"order_id", "order_datetime", "quantity", "sales_amount"} - set(merged_fields.values())
    )
    return SchemaMapping(
        dataset_type=rule_mapping.dataset_type
        if rule_mapping.dataset_type != "unknown"
        else llm_mapping.dataset_type,
        field_mapping=merged_fields,
        confidence=max(rule_mapping.confidence, llm_mapping.confidence),
        missing_required_fields=missing_required_fields,
        uncertain_fields=[
            field
            for field in {*(rule_mapping.uncertain_fields), *(llm_mapping.uncertain_fields)}
            if field not in merged_fields
        ],
    )


def map_schema(columns: list[str], llm_client=None) -> SchemaMapping:
    mapping, _ = map_schema_with_trace(columns, llm_client=llm_client)
    return mapping


def map_schema_with_trace(
    columns: list[str], llm_client=None
) -> tuple[SchemaMapping, LLMStageTrace]:
    rule_mapping = map_schema_fields(columns)
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return rule_mapping, build_llm_stage_trace(
            stage="schema_mapping",
            llm_client=llm_client,
            status="disabled",
            reason="LLM is disabled, so schema mapping used rule-based detection only.",
        )
    if not _should_use_llm(rule_mapping):
        return rule_mapping, build_llm_stage_trace(
            stage="schema_mapping",
            llm_client=llm_client,
            status="skipped",
            reason="Rule-based schema mapping was already confident enough, so the LLM stage was skipped.",
        )

    try:
        llm_mapping = llm_client.suggest_schema_mapping(columns, rule_mapping)
    except Exception as exc:
        return rule_mapping, build_llm_stage_trace(
            stage="schema_mapping",
            llm_client=llm_client,
            status="fallback_on_error",
            reason=f"LLM schema mapping failed; fell back to rule-based mapping. {describe_llm_error(exc)}",
            attempted=True,
        )

    merged_mapping = _merge_schema_mappings(rule_mapping, llm_mapping)
    applied = merged_mapping != rule_mapping
    return merged_mapping, build_llm_stage_trace(
        stage="schema_mapping",
        llm_client=llm_client,
        status="llm_applied" if applied else "llm_no_change",
        reason=(
            "LLM schema mapping refined the canonical field mapping."
            if applied
            else "LLM returned a schema mapping equivalent to the rule-based result."
        ),
        attempted=True,
        applied=applied,
    )
