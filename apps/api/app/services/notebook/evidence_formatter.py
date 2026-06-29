from __future__ import annotations

import re
from typing import Any

from app.schemas.report import AnalysisReport

from app.services.notebook.formatters import format_scalar
from app.services.notebook.markdown_sanitizer import clean_business_text


def evidence_text_id(text: str) -> str | None:
    match = re.match(r"^\[([^\]]+)\]", text.strip())
    if match:
        return match.group(1).strip()
    return None


def append_unique_text(items: list[str], text: str, seen_ids: set[str]) -> None:
    cleaned = clean_business_text(text)
    if not cleaned:
        return
    evidence_id = evidence_text_id(cleaned)
    if evidence_id:
        if evidence_id in seen_ids:
            return
        seen_ids.add(evidence_id)
    if cleaned not in items:
        items.append(cleaned)


def evidence_pack_findings(evidence_pack: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence_pack, dict):
        return []
    findings: list[str] = []
    for item in evidence_pack.get("distinctive_facts", []) or []:
        if isinstance(item, dict):
            text = clean_business_text(item.get("text", ""))
            if text and text not in findings:
                findings.append(text)
    return findings


def evidence_pack_risks(evidence_pack: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence_pack, dict):
        return []
    risks: list[str] = []
    seen_ids: set[str] = set()
    preferred_focuses = [
        "discount_erosion_focus",
        "profit_quality_focus",
        "segment_region_focus",
        "product_concentration_focus",
        "trend_volatility_focus",
        "country_market_focus",
        "customer_order_structure_focus",
    ]
    focus_evidence = evidence_pack.get("focus_evidence") or {}
    for focus in preferred_focuses:
        items = focus_evidence.get(focus, []) or []
        for item in items or []:
            if isinstance(item, dict):
                evidence_id = str(item.get("evidence_id", "")).strip()
                meaning = clean_business_text(item.get("business_meaning", ""))
                if evidence_id in {"region_count", "segment_count", "category_count", "product_count", "order_count", "customer_count"}:
                    continue
                if evidence_id and meaning:
                    before_count = len(risks)
                    append_unique_text(risks, f"[{evidence_id}] {meaning}", seen_ids)
                    if len(risks) > before_count:
                        break
        if len(risks) >= 3:
            break
    return risks


def row_evidence(row: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in row.items():
        text_value = clean_business_text(value)
        if not text_value:
            continue
        label = str(key)
        parts.append(f"{label}={format_scalar(value)}")
        if len(parts) >= 3:
            break
    return "，".join(parts)


def collect_evidence_items(report: AnalysisReport) -> list[str]:
    items: list[str] = []
    preferred_module_order = {
        "discount_profit_analysis": 0,
        "product_contribution_analysis": 1,
        "dimension_breakdown_analysis": 2,
        "sales_trend_analysis": 3,
        "metric_distribution_analysis": 4,
        "forecast_analysis": 5,
        "data_quality_check": 6,
    }
    sorted_modules = sorted(
        report.modules,
        key=lambda module: preferred_module_order.get(module.module_id, 50),
    )
    for module in sorted_modules:
        for finding in module.findings:
            text = clean_business_text(finding)
            if text:
                items.append(text)
        for rows in module.tables.values():
            for row in rows[:2]:
                if isinstance(row, dict):
                    evidence = row_evidence(row)
                    if evidence:
                        items.append(evidence)
    for summary in report.summary:
        text = clean_business_text(summary)
        if text:
            items.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def collect_risk_items(report: AnalysisReport) -> list[str]:
    risk_terms = ("风险", "亏损", "负利润", "低利润", "折扣", "侵蚀", "利润率")
    risks: list[str] = []
    for module in report.modules:
        for warning in module.warnings:
            text = clean_business_text(warning)
            if text:
                risks.append(text)
        for finding in module.findings:
            text = clean_business_text(finding)
            if text and any(term in text for term in risk_terms):
                risks.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in risks:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def remove_loss_evidence_when_no_negative_profit(
    evidence: list[str],
    profile: dict[str, Any],
) -> list[str]:
    if profile.get("negative_profit_rate") != 0:
        return evidence
    blocked_terms = ("亏损来源", "亏损商品", "利润为负", "负利润")
    filtered = [item for item in evidence if not any(term in item for term in blocked_terms)]
    replacement = "负利润记录占比为 0，本轮应关注利润率结构差异而不是亏损排查。"
    if replacement not in filtered:
        filtered.insert(0, replacement)
    return filtered


_evidence_text_id = evidence_text_id
_append_unique_text = append_unique_text
_evidence_pack_findings = evidence_pack_findings
_evidence_pack_risks = evidence_pack_risks
_row_evidence = row_evidence
_evidence_items = collect_evidence_items
_risk_items = collect_risk_items
_remove_loss_evidence_when_no_negative_profit = remove_loss_evidence_when_no_negative_profit
