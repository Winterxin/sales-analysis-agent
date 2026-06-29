from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook.markdown_sanitizer import clean_business_text
from app.services.output_language import contains_cjk


_DIMENSION_ALIASES = {
    "category": {"category", "cat"},
    "sub_category": {"sub_category", "subcategory", "subcat"},
    "product_name": {"product_name", "product", "item_name", "item", "sku_name"},
    "sku": {"sku", "item_sku", "product_sku"},
    "segment": {"segment", "customer_segment", "client_segment"},
    "region": {"region", "area", "territory"},
    "country": {"country", "nation"},
    "market": {"market"},
    "city": {"city"},
    "discount_bucket": {"discount_bucket", "discount_band", "discount_range", "discount_tier"},
    "discount_tier": {"discount_tier", "discount_level"},
    "channel": {"channel", "sales_channel", "order_channel"},
    "month": {"month", "order_month", "period", "date_month"},
}

_METRIC_ALIASES = {
    "profit": {"profit", "current_profit", "total_profit", "profit_amount"},
    "avg_profit": {"avg_profit", "average_profit", "mean_profit"},
    "profit_margin": {"profit_margin", "margin", "avg_margin", "average_margin"},
    "negative_profit_rate": {"negative_profit_rate", "loss_rate", "negative_rate"},
}

_DIMENSION_LOOKUP = {
    alias: role for role, aliases in _DIMENSION_ALIASES.items() for alias in aliases
}
_METRIC_LOOKUP = {alias: role for role, aliases in _METRIC_ALIASES.items() for alias in aliases}

_GENERIC_ENGLISH_MARKERS = (
    "this section reviews",
    "section purpose",
    "the analysis synthesizes",
    "coherent risk narrative",
    "available module evidence",
    "mapped sales dataset",
    "key findings, risks, and next actions",
)

_SCOPE_RISK_WORDS = (
    "unprofitable",
    "loss",
    "loss-making",
    "negative margin",
    "margin risk",
    "stop-sell",
    "stop sell",
    "stop selling",
    "full category",
    "category-wide",
    "overall category",
)

_CONDITIONAL_MARKERS = (
    "if available",
    "before diagnosing",
    "before identifying",
    "before assigning",
    "before changing",
    "before broader",
    "supporting records",
    "validate",
    "add approval data",
    "add cost",
    "add supplier",
    "add campaign",
    "add promotion",
    "add inventory",
)

_CAUSAL_STRENGTH_PATTERNS = (
    r"\b(?:a\s+)?primary\s+driver(?:\s+of)?\b",
    r"\bprimarily\s+driven\s+by\b",
    r"\bdriven\s+by\b",
    r"\bdrivers?\b",
    r"\bdirectly\s+linked\s+to\b",
    r"\bdirectly\s+tied\s+to\b",
    r"\btied\s+directly\s+to\b",
    r"\bdirectly\s+(?:destroying|destroys|destroyed|undermining|undermines|undermined)\s+(?:profitability|profits?|margins?)\b",
    r"\bdirectly\s+eroding\s+(?:overall\s+)?margin\b",
    r"\berodes?\b",
    r"\beroding\b",
    r"\bcaused?\s+by\b",
    r"\bcaused?\b",
    r"\bdrove\b",
    r"\bdriving\b",
    r"\bmay\s+drive\b",
    r"\b(?:almost|virtually)\s+guarantee(?:s|d)?\s+(?:a\s+)?(?:loss(?:es)?|negative\s+profit|unprofitable|negative\s+margin|margin\s+loss)\b",
    r"\bnear[-\s]guaranteed\s+(?:loss(?:es)?|negative\s+profit|unprofitable|negative\s+margin|margin\s+loss)\b",
    r"\bresponsible\s+for\b",
    r"\broot\s+causes?\b",
    r"\bproves?\b",
    r"\bconfirms?\b.{0,40}\bcaus",
    r"\bconfirms?\b.{0,80}\b(?:driver|discounting|pricing\s+failure)\b",
    r"\bdemonstrates?\b.{0,40}\bcaus",
    r"\bprofit\s+leakage\b",
    r"\bprofit\s+destroyer\b",
    r"\bfind\s+the\s+leak\b",
    r"\bfinancial\s+drain\b",
    r"\breverse\s+the\s+financial\s+drain\b",
    r"\bdoes\s+not\s+protect\s+margin\b",
    r"\bcannot\s+protect\s+margin\b",
    r"\bnot\s+being\s+effectively\s+managed\b",
    r"\bnot\s+effectively\s+managed\b",
    r"\bnot\s+leveraging\b",
    r"\bfailed\s+to\s+manage\b",
    r"\bfailed\s+to\s+protect\b",
    r"\bgovernance\s+is\s+(?:weak|ineffective)\b",
    r"\bsystemic\s+control\s+gaps?\b",
    r"\bcontrol\s+gaps?\b",
    r"\bcontrol\s+failure\b",
    r"\bstrategy\s+failure\b",
    r"\boperational\s+failure\b",
    r"\bdiagnose\s+causes?\s+such\s+as\b",
    r"\bcontributing\s+to\s+(?:significant\s+)?losses\b",
    r"\bnot\s+value[- ]accretive\b",
    r"\bpricing\s+does\s+not\s+cover\s+(?:their\s+)?costs?\b",
    r"\bapproval\s+controls?\s+(?:are\s+)?(?:weak|failing|broken)\b",
    r"\bsupplier\s+terms?\s+caused?\s+loss",
    r"\b(?:promotion|campaign|inventory)\b.{0,50}\bcaused?\b",
    r"\bmechanically\s+compress(?:es)?\b",
    r"\bcompensating\s+volume\s+gains?\b",
)

_UNSUPPORTED_OPERATIONAL_PATTERNS = (
    r"\btop\s+\d+(?:\.\d+)?%\b",
    r"\bfixed\s+review\s+quota\b",
    r"\bdedicated\s+review\s+team\b",
    r"\bfixed\s+staffing\b",
    r"\bmandatory\s+review(?:\s+process)?\b",
    r"\bmandatory\s+approval\s+workflow\b",
    r"\bmandatory\s+approval\b",
    r"\bapproval\s+logs?\b",
    r"\bapproval\s+workflows?\b",
    r"\bapproval\s+rules?\b",
    r"\bapproval\s+red\s+lines?\b",
    r"\bapproval\s+thresholds?\b",
    r"\bapproval\s+requests?\b",
    r"\bapproval\s+gates?\b",
    r"\bauthorization\s+process(?:es)?\b",
    r"\bauthorization\s+records?\b",
    r"\bsystemic\s+control\s+gaps?\b",
    r"\bcontrol\s+gaps?\b",
    r"\bweak\s+controls?\b",
    r"\bcontrol\s+failure\b",
    r"\bautomatic\s+operational\s+control\s+threshold\b",
)

_GUARD_METADATA_KEYS = {
    "raw_llm_final_synthesis_used",
    "guarded_llm_final_synthesis_used",
    "dropped_for_language",
    "dropped_for_scope",
    "dropped_for_unsupported_cause",
    "dropped_for_truncation",
    "downgraded_for_missing_fields",
    "fallback_item_count",
    "english_public_narrative_guard_applied",
    "english_public_narrative_guard_signature",
}

_PUBLIC_SYNTHESIS_ITEM_KEYS = (
    "brief_findings",
    "brief_actions",
    "main_conclusions",
    "recommended_actions",
)


def _canonical_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", token).strip("_")


def _raw_role_lookup(schema_mapping: SchemaMapping | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not schema_mapping:
        return lookup
    for raw_key, role in (schema_mapping.field_mapping or {}).items():
        normalized_role = _canonical_token(role)
        lookup[_canonical_token(raw_key)] = normalized_role
    return lookup


def _canonical_dimension_key(raw_key: object, schema_mapping: SchemaMapping | None) -> str | None:
    normalized = _canonical_token(raw_key)
    mapped_role = _raw_role_lookup(schema_mapping).get(normalized)
    if mapped_role in _DIMENSION_LOOKUP:
        return _DIMENSION_LOOKUP[mapped_role]
    if mapped_role in _DIMENSION_ALIASES:
        return mapped_role
    if normalized in _DIMENSION_LOOKUP:
        return _DIMENSION_LOOKUP[normalized]
    if normalized in _DIMENSION_ALIASES:
        return normalized
    return None


def _canonical_metric_key(raw_key: object, schema_mapping: SchemaMapping | None) -> str | None:
    normalized = _canonical_token(raw_key)
    mapped_role = _raw_role_lookup(schema_mapping).get(normalized)
    if mapped_role in _METRIC_LOOKUP:
        return _METRIC_LOOKUP[mapped_role]
    if mapped_role in _METRIC_ALIASES:
        return mapped_role
    if normalized in _METRIC_LOOKUP:
        return _METRIC_LOOKUP[normalized]
    if normalized in _METRIC_ALIASES:
        return normalized
    return None


def _number(value: object) -> float | None:
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip().rstrip("%")
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _metric_tokens(value: object) -> set[str]:
    number = _number(value)
    if number is None:
        return set()
    tokens = {
        f"{number:.4f}",
        f"{number:.3f}",
        f"{number:.2f}",
        f"{number:,.2f}",
        f"{number:.1f}",
        f"{number:,.1f}",
        f"{number:.0f}",
        f"{number:,.0f}",
    }
    if abs(number) <= 1:
        tokens.update({f"{number:.2%}", f"{number:.1%}"})
    return {token for token in tokens if token not in {"0", "0.0", "0.00", "0.000", "0.00%"}}


def build_scoped_evidence_rows(
    report: AnalysisReport,
    schema_mapping: SchemaMapping | None = None,
) -> list[dict[str, Any]]:
    scoped: list[dict[str, Any]] = []
    for module in report.modules:
        for table_rows in (module.tables or {}).values():
            if not isinstance(table_rows, list):
                continue
            for row in table_rows:
                if not isinstance(row, dict):
                    continue
                dimensions: dict[str, str] = {}
                metric_tokens: set[str] = set()
                for key, value in row.items():
                    dim_key = _canonical_dimension_key(key, schema_mapping)
                    if dim_key:
                        text_value = str(value).strip()
                        if text_value:
                            dimensions[dim_key] = text_value
                    metric_key = _canonical_metric_key(key, schema_mapping)
                    if metric_key:
                        metric_tokens.update(_metric_tokens(value))
                if len(dimensions) >= 2 and metric_tokens:
                    scoped.append({"dimensions": dimensions, "metric_tokens": metric_tokens})
    return scoped


def violates_scoped_evidence(text: str, scoped_rows: list[dict[str, Any]]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    risk_claim = any(word in lowered for word in _SCOPE_RISK_WORDS)
    for row in scoped_rows:
        dimensions = row.get("dimensions") if isinstance(row, dict) else {}
        metric_tokens = row.get("metric_tokens") if isinstance(row, dict) else set()
        if not isinstance(dimensions, dict) or not dimensions:
            continue
        parent_hits = [
            value
            for role, value in dimensions.items()
            if role not in {"discount_bucket", "discount_tier"}
            and str(value).strip()
            and str(value).lower() in lowered
        ]
        if not parent_hits:
            continue
        missing_scope = [
            value
            for value in dimensions.values()
            if str(value).strip() and str(value).lower() not in lowered
        ]
        metric_hit = any(str(token) and str(token) in text for token in metric_tokens or set())
        if missing_scope and (metric_hit or risk_claim):
            return True
    return False


def _mapped_field_tokens(schema_mapping: SchemaMapping | None) -> set[str]:
    tokens: set[str] = set()
    if not schema_mapping:
        return tokens
    for raw_key, role in (schema_mapping.field_mapping or {}).items():
        tokens.add(_canonical_token(raw_key))
        tokens.add(_canonical_token(role))
    return tokens


def _field_available(tokens: set[str], aliases: tuple[str, ...]) -> bool:
    return any(alias in tokens for alias in aliases)


def _has_causal_strength_claim(lowered: str) -> bool:
    return any(re.search(pattern, lowered) for pattern in _CAUSAL_STRENGTH_PATTERNS)


def _public_synthesis_signature(payload: dict[str, Any]) -> str:
    public_items = {key: payload.get(key, []) for key in _PUBLIC_SYNTHESIS_ITEM_KEYS}
    return json.dumps(public_items, sort_keys=True, ensure_ascii=False, default=str)


def _unsupported_model_operationalization_group(lowered: str) -> str | None:
    model_context = any(
        term in lowered
        for term in (
            "model",
            "risk score",
            "risk scores",
            "highest-risk",
            "loss-risk",
            "order processing",
            "order-processing",
            "fulfillment",
        )
    )
    if any(
        phrase in lowered
        for phrase in (
            "order processing system",
            "order-processing workflow",
            "before final processing",
            "before fulfillment",
        )
    ):
        return "model_review_threshold"
    if re.search(r"\btop\s+\d+(?:\.\d+)?%\s+highest-risk\b", lowered):
        return "model_review_threshold"
    if not model_context:
        return None
    unsafe_phrases = (
        "automatically flag",
        "automatic routing",
        "automatic approval",
        "automatic blocking",
        "automatic execution",
        "automated workflow",
        "mandatory human review",
        "mandatory review process",
        "daily list",
        "weekly list",
        "monthly list",
        "daily review",
        "weekly review",
        "monthly review",
        "daily audit",
        "weekly audit",
        "monthly audit",
        "daily manual audit",
        "weekly manual review",
        "audit workflow",
        "review workflow",
        "operations team",
        "operations workflow",
        "integrate into operations",
        "integrating into operations",
        "continuous risk management",
        "operationalize the model",
        "deploy the model",
        "deploy the loss-risk model",
        "production workflow",
        "production deployment",
        "mitigate losses operationally",
        "potentially saving significant margin",
        "saving significant margin",
    )
    if any(phrase in lowered for phrase in unsafe_phrases):
        return "model_review_threshold"
    return None


def _unsupported_operational_group(lowered: str) -> str | None:
    model_operational_group = _unsupported_model_operationalization_group(lowered)
    if model_operational_group:
        return model_operational_group
    if any(re.search(pattern, lowered) for pattern in _UNSUPPORTED_OPERATIONAL_PATTERNS):
        if "mandatory review" in lowered and "discount" in lowered:
            return "discount_profit_causality"
        if "approval" in lowered or "mandatory review" in lowered:
            return "approval"
        if "top" in lowered or "quota" in lowered or "staffing" in lowered or "review team" in lowered:
            return "model_review_threshold"
        return "unsupported_causality"
    return None


def _is_safe_conditional_sentence(lowered: str) -> bool:
    if re.search(r"\breview\s+approval\s+records\s+if\s+available\s+before\s+changing\s+approval\s+controls\b", lowered):
        return True
    if re.search(r"\badd\s+cost\s+or\s+supplier\s+data\s+before\s+identifying\s+root\s+causes\b", lowered):
        return True
    if re.search(
        r"\breview\s+campaign,\s*promotion,\s*or\s*inventory\s+data\s+if\s+available\s+before\s+assigning\s+a\s+root\s+cause\b",
        lowered,
    ):
        return True
    if re.search(
        r"\breview\s+the\s+scoped\s+discount\s+tier\s+against\s+profit\s+outcomes\s+before\s+broader\s+pricing\s+changes\b",
        lowered,
    ):
        return True
    if re.search(
        r"\bvalidate\s+(?:the\s+)?(?:observed\s+relationship|evidence|field|data|record|records)\s+before\s+(?:assigning|diagnosing|changing)\b",
        lowered,
    ):
        return True
    if re.search(
        r"\bselect\s+a\s+review\s+threshold\s+based\s+on\s+review\s+capacity\s+and\s+the\s+observed\s+precision-recall\s+trade-off\b",
        lowered,
    ):
        return True
    if re.search(
        r"\bselect\s+a\s+review\s+threshold\s+based\s+on\s+review\s+capacity\s+and\s+the\s+observed\s+false-positive\s*/\s*false-negative\s+trade-off\b",
        lowered,
    ):
        return True
    if re.search(r"\bvalidate\s+rather\s+than\s+assert\s+cause\b", lowered):
        return True
    if re.search(
        r"\breview\s+the\s+scoped\s+evidence\s+before\s+changing\s+broader\s+category,\s*region,\s*product,\s*or\s*pricing\s+policies\b",
        lowered,
    ):
        return True
    return False


def _unsupported_cause_group(text: str, schema_mapping: SchemaMapping | None) -> str | None:
    lowered = text.lower()
    operational_group = _unsupported_operational_group(lowered)
    if operational_group:
        return operational_group
    tokens = _mapped_field_tokens(schema_mapping)
    approval_available = _field_available(tokens, ("approval", "approver", "approval_status", "approval_tier"))
    cost_available = _field_available(tokens, ("cost", "cogs", "unit_cost", "supplier", "vendor", "procurement"))
    shipping_cost_available = _field_available(
        tokens,
        (
            "shipping_cost",
            "shipping_expense",
            "shipping_expenses",
            "ship_cost",
            "freight_cost",
            "fulfillment_cost",
            "fulfillment_expense",
            "fulfillment_expenses",
            "logistics_cost",
            "logistics_expense",
        ),
    )
    campaign_available = _field_available(tokens, ("campaign", "promotion", "promo", "inventory", "stock"))
    approval_terms = (
        "approval",
        "approvals",
        "approval gate",
        "approval gates",
        "approval log",
        "approval logs",
        "approval workflow",
        "approval workflows",
        "approval rule",
        "approval rules",
        "approval red line",
        "approval red lines",
        "approval threshold",
        "approval thresholds",
        "approval request",
        "approval requests",
        "authorization process",
        "authorization records",
        "systemic control gap",
        "systemic control gaps",
        "control gap",
        "control gaps",
        "weak control",
        "weak controls",
        "control failure",
    )
    cost_terms = (
        "supplier",
        "procurement",
        "cost structure",
        "high cost of goods",
        "cost of goods",
        "supplier cost inflation",
        "cogs",
        "supplier terms",
        "renegotiate supplier",
        "does not cover",
    )
    shipping_cost_terms = (
        "high shipping expenses",
        "shipping expense",
        "shipping expenses",
        "shipping cost",
        "shipping costs",
        "fulfillment cost",
        "fulfillment costs",
        "fulfillment expense",
        "fulfillment expenses",
        "logistics cost",
        "logistics costs",
    )
    root_cause_cost_terms = (
        "diagnose causes such as",
        "contributing to significant losses",
        "contributing to losses",
    )
    campaign_terms = (
        "campaign",
        "promotional history",
        "promotional strategies",
        "promotional strategy",
        "promotion",
        "inventory",
        "demand",
    )
    strong_claim = _has_causal_strength_claim(lowered)
    if any(marker in lowered for marker in _CONDITIONAL_MARKERS) and _is_safe_conditional_sentence(lowered):
        return None
    mandatory_approval = "mandatory approval" in lowered or ("implement" in lowered and "approval" in lowered)
    if any(term in lowered for term in approval_terms) and (mandatory_approval or strong_claim or not approval_available):
        return "approval"
    if any(term in lowered for term in shipping_cost_terms) and not shipping_cost_available:
        return "cost_supplier"
    if any(term in lowered for term in root_cause_cost_terms) and (
        any(term in lowered for term in cost_terms) or any(term in lowered for term in shipping_cost_terms)
    ):
        return "cost_supplier"
    if any(term in lowered for term in cost_terms) and (strong_claim or not cost_available):
        return "cost_supplier"
    if any(term in lowered for term in campaign_terms) and (strong_claim or not campaign_available):
        return "campaign_inventory"
    if strong_claim and any(term in lowered for term in ("discount", "margin", "profit", "loss", "pricing", "financial drain")):
        return "discount_profit_causality"
    if strong_claim:
        return "unsupported_causality"
    return None


def _conditional_cause_sentence(group: str) -> str:
    if group == "approval":
        return "Review approval records if available before changing approval controls."
    if group == "cost_supplier":
        return "Add cost or supplier data before identifying root causes."
    if group == "discount_profit_causality":
        return "Review the scoped discount tier against profit outcomes before broader pricing changes."
    if group == "model_review_threshold":
        return "Use model risk scores only to prioritize manual review, with thresholds set against review capacity and observed false-positive / false-negative trade-offs."
    if group == "scope":
        return "Review the scoped evidence before changing broader category, region, product, or pricing policies."
    if group == "unsupported_causality":
        return "Validate the observed relationship before assigning a root cause."
    return "Review campaign, promotion, or inventory data if available before assigning a root cause."


def _clean_english_sentence(value: object, *, preserve_numeric_literals: bool = False) -> str:
    if "\ufffd" in str(value or ""):
        return ""
    if preserve_numeric_literals:
        text = " ".join(str(value or "").replace("\ufffd", "").split())
    else:
        text = " ".join(clean_business_text(value).replace("\ufffd", "").split())
    if not text or contains_cjk(text):
        return ""
    if "…" in text or "..." in text:
        marker = "…" if "…" in text else "..."
        prefix = text.split(marker, 1)[0].strip()
        complete = re.findall(r"[^.!?]+[.!?]", prefix)
        text = " ".join(item.strip() for item in complete)
        if not text:
            return ""
    if any(marker in text.lower() for marker in _GENERIC_ENGLISH_MARKERS):
        return ""
    lowered = text.casefold().strip()
    if lowered.endswith(("vs.", "e.g.", "i.e.", "etc.", "mr.", "ms.", "dr.")):
        return ""
    if re.search(r"\b(?:and|or|with|to|of|for|in|at|by|from|the|a|an)$", lowered):
        return ""
    if re.search(
        r"^(?:connecting this to|alongside|compared with|based on|relative to)\b.*\b(?:peer|chart|scatter|bar|line|heatmap)\b",
        lowered,
    ) and not re.search(
        r"\b(?:shows?|indicates?|summarizes?|compares?|supports?|requires?|lists?|uses?|review|select|interpret)\b",
        lowered,
    ):
        return ""
    if not re.search(r"[.!?]$", text):
        text += "."
    return text


def _fallback_conclusion(report: AnalysisReport, scoped_rows: list[dict[str, Any]]) -> str:
    total_sales = None
    for module in report.modules:
        total_sales = (module.summary_metrics or {}).get("total_sales_amount")
        if total_sales is not None:
            break
    if total_sales is not None:
        return f"Total sales reached {total_sales:,.2f} based on the mapped sales field."
    if scoped_rows:
        values = list(scoped_rows[0].get("dimensions", {}).values())
        if values:
            return f"The {' / '.join(str(value) for value in values)} slice should be interpreted at its full evidence scope."
    return "Use the module evidence above as the source of scope for operational conclusions."


def _fallback_action(scoped_rows: list[dict[str, Any]]) -> str:
    if scoped_rows:
        values = list(scoped_rows[0].get("dimensions", {}).values())
        if values:
            return f"Review the {' / '.join(str(value) for value in values)} slice before changing broader category, region, product, or pricing policies."
    return "Add approval, cost, supplier, campaign, promotion, or inventory fields before assigning unsupported root causes."


def _safe_priority(value: object, fallback: str) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"P1", "P2", "P3", "P4"} else fallback


def _action_issue_for_group(group: str) -> str:
    if group == "approval":
        return "Approval records are required before changing controls."
    if group == "cost_supplier":
        return "Cost or supplier data is required before root-cause claims."
    if group == "campaign_inventory":
        return "Campaign, promotion, or inventory records are required before assigning cause."
    if group == "discount_profit_causality":
        return "Discount-profit relationship requires scoped review."
    if group == "model_review_threshold":
        return "Model use must stay limited to manual review prioritization."
    if group == "scope":
        return "Scoped evidence must not be expanded into broader action."
    return "Observed relationship requires validation."


def _action_evidence_for_group(group: str) -> str:
    if group == "approval":
        return "Approval fields are not mapped as direct evidence."
    if group == "cost_supplier":
        return "Cost or supplier fields are not mapped as direct evidence."
    if group == "campaign_inventory":
        return "Campaign, promotion, or inventory fields are not mapped as direct evidence."
    if group == "discount_profit_causality":
        return "Mapped profit and discount evidence supports association review, not causality."
    if group == "model_review_threshold":
        return "Use observed false-positive / false-negative trade-offs and review capacity."
    if group == "scope":
        return "Keep the action at the same scope as the evidence row."
    return "Validate the relationship with supporting records before assigning cause."


def _fallback_action_item(
    group: str,
    *,
    primary_key: str,
    detail_keys: tuple[str, ...],
    priority: object,
) -> dict[str, str]:
    action_text = _conditional_cause_sentence(group)
    item: dict[str, str] = {primary_key: action_text}
    evidence_text = _action_evidence_for_group(group)
    for key in detail_keys:
        if key in {"linked_metric_or_segment", "linked_evidence"}:
            item[key] = evidence_text
        elif key in {"expected_use", "priority_reason"}:
            item[key] = action_text
        else:
            item[key] = evidence_text
    if primary_key == "action":
        item["issue"] = _action_issue_for_group(group)
        item["display_text"] = action_text
        item["priority"] = _safe_priority(priority, "P3")
    return item


def _model_review_threshold_conclusion_item() -> dict[str, str]:
    return {
        "conclusion": "The loss-risk model can support manual review prioritization.",
        "evidence": "Use observed model validation evidence only to compare review trade-offs.",
        "business_meaning": (
            "Select a review threshold based on review capacity and the observed "
            "false-positive / false-negative trade-off."
        ),
    }


def _discount_profit_association_item(primary_key: str) -> dict[str, str]:
    return {
        primary_key: "The scoped discount-tier evidence should be interpreted as an association with profit outcomes.",
        "evidence": "Use the mapped discount and profit evidence at its stated scope.",
        "business_meaning": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
    }


def _has_near_terms(lowered: str, left: tuple[str, ...], right: tuple[str, ...], *, distance: int = 80) -> bool:
    for left_term in left:
        for right_term in right:
            left_pattern = re.escape(left_term).replace(r"\ ", r"\s+")
            right_pattern = re.escape(right_term).replace(r"\ ", r"\s+")
            if re.search(rf"\b{left_pattern}\b.{{0,{distance}}}\b{right_pattern}\b", lowered):
                return True
            if re.search(rf"\b{right_pattern}\b.{{0,{distance}}}\b{left_pattern}\b", lowered):
                return True
    return False


def _has_model_review_threshold_overreach(lowered: str) -> bool:
    exact_phrases = (
        "before final processing",
        "before fulfillment",
        "before shipping",
        "order processing",
        "order-processing",
        "operational workflow",
        "production workflow",
        "production deployment",
        "proactive risk management",
        "operational safeguard",
        "operational control",
        "risk-control mechanism",
        "transition from reactive analysis to proactive risk management",
        "deploy",
        "integrate",
        "automatic",
        "automatically",
        "mandatory",
        "daily review workflow",
        "weekly review workflow",
        "monthly review workflow",
        "daily audit workflow",
        "weekly audit workflow",
        "monthly audit workflow",
        "daily manual review",
        "weekly manual review",
        "monthly manual review",
        "daily manual audit",
        "weekly manual audit",
        "monthly manual audit",
        "mitigate losses operationally",
        "potentially saving significant margin",
        "saving significant margin",
        "measurable benefit",
        "measurable safeguard",
        "significant improvement",
        "safeguard against profit erosion",
    )
    if any(phrase in lowered for phrase in exact_phrases):
        return True
    if re.search(r"\bbefore\b.{0,60}\bship(?:s|ped|ping)?\b", lowered):
        return True
    if re.search(r"\bbefore\b.{0,60}\bfulfillment\b", lowered):
        return True
    if re.search(r"\bbefore\b.{0,60}\bprocessing\b", lowered):
        return True
    if _has_near_terms(
        lowered,
        ("intercept", "block", "hold", "stop", "reroute", "prevent"),
        ("order", "orders", "shipment", "ship", "shipping", "fulfillment", "processing"),
    ):
        return True
    if _has_near_terms(
        lowered,
        ("protect", "protection", "safeguard"),
        ("profit", "margin", "loss", "losses", "erosion"),
    ):
        return True
    if _has_near_terms(
        lowered,
        ("save", "recover", "avoid", "prevent", "reduce", "mitigate"),
        ("profit", "margin", "loss", "losses"),
    ):
        return True
    return False


def _joined_model_review_threshold_group(raw: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    joined = " ".join(str(raw.get(key) or "") for key in keys).casefold()
    model_context = any(
        term in joined
        for term in (
            "model",
            "predictive model",
            "classification model",
            "loss-risk",
            "risk score",
            "risk scores",
            "logisticregression",
            "high-risk orders",
        )
    )
    if not model_context:
        return None
    if _has_model_review_threshold_overreach(joined):
        return "model_review_threshold"
    return None


def _joined_discount_profit_association_overclaim_group(raw: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    joined = " ".join(str(raw.get(key) or "") for key in keys).casefold()
    has_discount_context = any(
        term in joined
        for term in ("discount", "discount tier", "discount bucket", "discount rate", "discounting")
    )
    has_profit_context = any(
        term in joined
        for term in ("profit", "margin", "loss", "loss-making", "negative profit", "losses")
    )
    if not (has_discount_context and has_profit_context):
        return None
    exact_phrases = (
        "strongly associated",
        "highly associated",
        "almost perfectly correlated",
        "near-perfect correlation",
        "perfectly correlated",
        "highly correlated",
        "strongly correlated",
        "almost guaranteed to result in a loss",
        "nearly guaranteed to result in a loss",
        "virtually guaranteed to result in a loss",
        "guaranteed to result in a loss",
        "primary trigger",
        "main trigger",
        "clear trigger",
        "primary driver",
        "main driver",
        "clear driver",
        "converting a sale into a loss",
        "turns sales into losses",
        "profit protection",
        "guarantees loss",
    )
    if any(phrase in joined for phrase in exact_phrases):
        return "discount_profit_causality"
    if "likely to result in a loss" in joined and any(
        term in joined for term in ("strong", "strongly", "highly", "clear", "primary", "main", "guaranteed")
    ):
        return "discount_profit_causality"
    return None


def _guard_items(
    items: Any,
    *,
    primary_key: str,
    detail_keys: tuple[str, ...],
    action: bool,
    report: AnalysisReport,
    schema_mapping: SchemaMapping | None,
    scoped_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    guarded: list[dict[str, str]] = []
    seen: set[str] = set()
    rejected = 0
    for raw in items:
        if not isinstance(raw, dict):
            continue
        cleaned: dict[str, str] = {}
        reject_item = False
        cause_groups: list[str] = []
        optional_keys = (
            "display_text",
            "issue",
            "linked_metric_or_segment",
            "linked_evidence",
            "expected_use",
            "priority_reason",
            "priority",
        ) if action else ("display_text",)
        item_keys = tuple(dict.fromkeys((primary_key, *detail_keys, *optional_keys)))
        item_level_group = _joined_model_review_threshold_group(raw, item_keys)
        if item_level_group and not action:
            cleaned = _model_review_threshold_conclusion_item()
            metadata["dropped_for_unsupported_cause"] += 1
            metadata["downgraded_for_missing_fields"] += 1
            key = " ".join(cleaned.values()).casefold()
            if key not in seen:
                seen.add(key)
                guarded.append(cleaned)
            continue
        item_level_group = _joined_discount_profit_association_overclaim_group(raw, item_keys)
        if item_level_group and not action and primary_key in {"finding", "conclusion"}:
            cleaned = _discount_profit_association_item(primary_key)
            metadata["dropped_for_unsupported_cause"] += 1
            metadata["downgraded_for_missing_fields"] += 1
            key = " ".join(cleaned.values()).casefold()
            if key not in seen:
                seen.add(key)
                guarded.append(cleaned)
            continue
        for key in item_keys:
            if key not in raw:
                continue
            value = _clean_english_sentence(raw.get(key))
            if not value:
                if contains_cjk(raw.get(key)):
                    metadata["dropped_for_language"] += 1
                elif "…" in str(raw.get(key) or "") or "..." in str(raw.get(key) or ""):
                    metadata["dropped_for_truncation"] += 1
                reject_item = True
                continue
            cause_group = _unsupported_cause_group(value, schema_mapping)
            if cause_group:
                cause_groups.append(cause_group)
                reject_item = True
            cleaned[key] = value
        joined = " ".join(cleaned.values())
        if not cleaned.get(primary_key) or not joined:
            rejected += 1
            continue
        if any(violates_scoped_evidence(value, scoped_rows) for value in cleaned.values()):
            metadata["dropped_for_scope"] += 1
            cause_groups.insert(0, "scope")
            reject_item = True
        if reject_item:
            if cause_groups:
                metadata["dropped_for_unsupported_cause"] += 1
            if action:
                group = cause_groups[0] if cause_groups else "unsupported_causality"
                cleaned = _fallback_action_item(
                    group,
                    primary_key=primary_key,
                    detail_keys=detail_keys,
                    priority=raw.get("priority"),
                )
                metadata["downgraded_for_missing_fields"] += 1
            else:
                rejected += 1
                continue
        key = joined.casefold()
        if action:
            key = " ".join(cleaned.values()).casefold()
        if key in seen:
            continue
        seen.add(key)
        guarded.append(cleaned)
    return guarded


def _fallback_public_sentence(role: str, chart_context: dict[str, Any] | None = None) -> str:
    if role == "chart_commentary":
        title = _clean_english_sentence((chart_context or {}).get("chart_title") or "The executed chart")
        title = title.rstrip(".")
        kind = str((chart_context or {}).get("chart_kind") or "chart").replace("_", " ")
        return (
            f"{title} compares the available {kind} evidence without assigning root causes. "
            "Review the underlying records and peer charts before operational changes."
        )
    return "Validate the observed relationship before assigning a root cause."


_PUBLIC_SENTENCE_ABBREVIATIONS = (
    "vs.",
    "e.g.",
    "i.e.",
    "etc.",
    "mr.",
    "ms.",
    "dr.",
)


def _is_public_sentence_abbreviation_period(text: str, index: int) -> bool:
    prefix = text[: index + 1].lower()
    return any(prefix.endswith(abbreviation) for abbreviation in _PUBLIC_SENTENCE_ABBREVIATIONS)


def _split_english_public_sentences(raw_text: str) -> list[str]:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    sentences: list[str] = []
    start = 0
    length = len(text)
    for index, char in enumerate(text):
        if char == "\n":
            candidate = text[start:index].strip()
            if candidate:
                sentences.append(candidate)
            start = index + 1
            continue
        if char not in ".!?":
            continue
        prev_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < length else ""
        if char == "." and prev_char.isdigit() and next_char.isdigit():
            continue
        if char == "." and _is_public_sentence_abbreviation_period(text, index):
            continue
        if index + 1 >= length:
            candidate = text[start:].strip()
            if candidate:
                sentences.append(candidate)
            start = length
            break
        if not next_char.isspace():
            continue
        lookahead_index = index + 1
        while lookahead_index < length and text[lookahead_index].isspace() and text[lookahead_index] != "\n":
            lookahead_index += 1
        if lookahead_index >= length:
            candidate = text[start:].strip()
            if candidate:
                sentences.append(candidate)
            start = length
            break
        next_visible = text[lookahead_index]
        if char in "!?" or text[index + 1] == "\n" or next_visible.isupper() or next_visible in "\"'“‘-•*":
            candidate = text[start : index + 1].strip()
            if candidate:
                sentences.append(candidate)
            start = lookahead_index
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def guard_english_public_narrative(
    text: object,
    *,
    report: AnalysisReport | None = None,
    schema_mapping: SchemaMapping | None = None,
    chart_context: dict[str, Any] | None = None,
    role: str = "chart_commentary",
    fallback: str | None = None,
) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        raw_text = fallback or ""
    scoped_rows = build_scoped_evidence_rows(report, schema_mapping) if report is not None else []
    candidates = _split_english_public_sentences(raw_text)
    safe_sentences: list[str] = []
    rejected_groups: list[str] = []
    for candidate in candidates:
        sentence = _clean_english_sentence(candidate, preserve_numeric_literals=True)
        if not sentence:
            continue
        cause_group = _unsupported_cause_group(sentence, schema_mapping)
        if cause_group:
            rejected_groups.append(cause_group)
            continue
        if violates_scoped_evidence(sentence, scoped_rows):
            rejected_groups.append("scope")
            continue
        safe_sentences.append(sentence)
    if safe_sentences:
        if role == "chart_commentary" and len(safe_sentences) < 2:
            supplement_source = (
                _conditional_cause_sentence(rejected_groups[0])
                if rejected_groups
                else fallback or _fallback_public_sentence(role, chart_context)
            )
            for supplement in _split_english_public_sentences(supplement_source):
                sentence = _clean_english_sentence(supplement, preserve_numeric_literals=True)
                if not sentence or sentence in safe_sentences:
                    continue
                if _unsupported_cause_group(sentence, schema_mapping):
                    continue
                if violates_scoped_evidence(sentence, scoped_rows):
                    continue
                safe_sentences.append(sentence)
                break
        return " ".join(safe_sentences)
    fallback_text = (
        _conditional_cause_sentence(rejected_groups[0])
        if rejected_groups
        else fallback or _fallback_public_sentence(role, chart_context)
    )
    if fallback_text != raw_text:
        return guard_english_public_narrative(
            fallback_text,
            report=report,
            schema_mapping=schema_mapping,
            chart_context=chart_context,
            role=role,
            fallback=_fallback_public_sentence(role, chart_context),
        )
    return _fallback_public_sentence(role, chart_context)


def guard_english_final_synthesis(
    final_synthesis: dict[str, Any] | None,
    *,
    report: AnalysisReport,
    schema_mapping: SchemaMapping | None = None,
    dataset_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del dataset_profile
    source = deepcopy(final_synthesis or {})
    metadata = dict(source.get("metadata") if isinstance(source.get("metadata"), dict) else {})
    source_signature = _public_synthesis_signature(source)
    if (
        metadata.get("english_public_narrative_guard_applied")
        and metadata.get("english_public_narrative_guard_signature") == source_signature
    ):
        return source
    raw_used_llm = bool(metadata.get("final_synthesis_used_llm"))
    guard_metadata = {
        "raw_llm_final_synthesis_used": raw_used_llm,
        "guarded_llm_final_synthesis_used": False,
        "dropped_for_language": 0,
        "dropped_for_scope": 0,
        "dropped_for_unsupported_cause": 0,
        "dropped_for_truncation": 0,
        "downgraded_for_missing_fields": 0,
        "fallback_item_count": 0,
        "english_public_narrative_guard_applied": True,
    }
    scoped_rows = build_scoped_evidence_rows(report, schema_mapping)
    guarded = {
        "brief_findings": _guard_items(
            source.get("brief_findings"),
            primary_key="finding",
            detail_keys=("evidence", "business_meaning"),
            action=False,
            report=report,
            schema_mapping=schema_mapping,
            scoped_rows=scoped_rows,
            metadata=guard_metadata,
        ),
        "brief_actions": _guard_items(
            source.get("brief_actions"),
            primary_key="action",
            detail_keys=("linked_evidence", "priority_reason"),
            action=True,
            report=report,
            schema_mapping=schema_mapping,
            scoped_rows=scoped_rows,
            metadata=guard_metadata,
        ),
        "main_conclusions": _guard_items(
            source.get("main_conclusions"),
            primary_key="conclusion",
            detail_keys=("evidence", "business_meaning"),
            action=False,
            report=report,
            schema_mapping=schema_mapping,
            scoped_rows=scoped_rows,
            metadata=guard_metadata,
        ),
        "recommended_actions": _guard_items(
            source.get("recommended_actions"),
            primary_key="action",
            detail_keys=("linked_metric_or_segment", "expected_use"),
            action=True,
            report=report,
            schema_mapping=schema_mapping,
            scoped_rows=scoped_rows,
            metadata=guard_metadata,
        ),
        "metadata": metadata,
    }
    raw_main_count = sum(1 for item in source.get("main_conclusions", []) if isinstance(item, dict))
    if raw_main_count > len(guarded["main_conclusions"]):
        fallback_conclusion = _fallback_conclusion(report, scoped_rows)
        fallback_key = fallback_conclusion.casefold()
        existing = " ".join(
            " ".join(str(value) for value in item.values()).casefold()
            for item in guarded["main_conclusions"]
            if isinstance(item, dict)
        )
        if fallback_key not in existing:
            guarded["main_conclusions"].append(
                {
                    "conclusion": fallback_conclusion,
                    "evidence": "Mapped report evidence supports this bounded summary.",
                    "business_meaning": "Use the mapped sales evidence as context before business changes.",
                }
            )
            guard_metadata["fallback_item_count"] += 1
    has_guarded_content = bool(
        guarded["brief_findings"]
        or guarded["brief_actions"]
        or guarded["main_conclusions"]
        or guarded["recommended_actions"]
    )
    guard_metadata["guarded_llm_final_synthesis_used"] = bool(raw_used_llm and has_guarded_content)
    metadata.update(guard_metadata)
    metadata["english_public_narrative_guard_signature"] = _public_synthesis_signature(guarded)
    if raw_used_llm and not has_guarded_content:
        metadata["final_synthesis_fallback_reason"] = "guarded_english_final_synthesis_unavailable"
    return guarded
