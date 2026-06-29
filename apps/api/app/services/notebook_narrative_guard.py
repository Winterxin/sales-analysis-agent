from __future__ import annotations

from typing import Any


UNSUPPORTED_RULES: tuple[tuple[str, set[str], tuple[str, ...]], ...] = (
    ("discount", {"discount"}, ("折扣", "discount", "折扣策略", "折扣侵蚀")),
    ("profit", {"profit"}, ("利润", "利润率", "利润质量", "profit", "margin")),
    ("customer_order", {"customer_id", "order_id"}, ("复购", "客户生命周期", "订单结构")),
)


LIMITATION_MARKERS = (
    "缺少",
    "补充",
    "补齐",
    "没有",
    "未提供",
    "不可用",
    "不支持",
    "missing",
    "unavailable",
    "not available",
)


DATA_COVERAGE_BOUNDARY_MARKERS = (
    "missing",
    "not available",
    "unavailable",
    "out of scope",
    "缺少",
    "未提供",
    "不可用",
    "暂不纳入",
)


DATA_COVERAGE_VERBS = (
    "add",
    "include",
    "provide",
    "map",
    "collect",
    "capture",
    "supplement",
    "补充",
    "补齐",
)


DATA_COVERAGE_OBJECTS = (
    "fields",
    "field",
    "columns",
    "attributes",
    "business data",
    "data coverage",
    "字段",
    "口径",
)


def _is_markdown_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


def _is_markdown_bullet(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("- ") or stripped.startswith("* ")


def _bullet_prefix(line: str) -> str:
    leading = line[: len(line) - len(line.lstrip())]
    marker = line.lstrip()[:2]
    return f"{leading}{marker}"


def _rewrite_soft_unsupported_claims(block: str) -> str:
    rewritten = block
    rewritten = rewritten.replace("是否同步贡献利润", "是否具备稳定销售贡献")
    rewritten = rewritten.replace("是否同步转化为利润", "是否具备稳定销售贡献")
    rewritten = rewritten.replace("利润质量", "销售贡献质量")
    rewritten = rewritten.replace("客户生命周期价值不足", "需结合长期价值字段进一步验证客户价值")
    return rewritten


def _is_data_coverage_action(text: str) -> bool:
    lower = text.lower()
    if any(marker in lower or marker in text for marker in DATA_COVERAGE_BOUNDARY_MARKERS):
        return True
    has_verb = any(marker in lower or marker in text for marker in DATA_COVERAGE_VERBS)
    has_object = any(marker in lower or marker in text for marker in DATA_COVERAGE_OBJECTS)
    return has_verb and has_object


def _guard_text_fragment(text: str, *, mapped_fields: set[str]) -> tuple[str, bool, list[str]]:
    lower = text.lower()
    removed_terms: list[str] = []
    for _, required_fields, terms in UNSUPPORTED_RULES:
        if required_fields & mapped_fields:
            continue
        matched_terms = [term for term in terms if term.lower() in lower or term in text]
        if not matched_terms:
            continue
        removed_terms.extend(matched_terms)
        if any(marker in lower or marker in text for marker in LIMITATION_MARKERS):
            continue
        if _is_data_coverage_action(text):
            continue
        rewritten = _rewrite_soft_unsupported_claims(text)
        rewritten_lower = rewritten.lower()
        still_matches = [
            term for term in terms if term.lower() in rewritten_lower or term in rewritten
        ]
        if not still_matches:
            return rewritten, False, removed_terms
        return "", True, removed_terms
    return text, False, removed_terms


def _guard_markdown_bullet_block(
    block: str,
    *,
    mapped_fields: set[str],
) -> tuple[str, list[str], list[str]] | None:
    lines = block.splitlines()
    if not lines or not _is_markdown_heading(lines[0]):
        return None
    if not any(_is_markdown_bullet(line) for line in lines[1:]):
        return None

    kept_lines = [lines[0]]
    removed_lines: list[str] = []
    removed_terms: list[str] = []
    kept_bullet_count = 0
    for line in lines[1:]:
        if not _is_markdown_bullet(line):
            kept_lines.append(line)
            continue
        bullet_text = line.lstrip()[2:].strip()
        guarded_text, should_remove, terms = _guard_text_fragment(
            bullet_text,
            mapped_fields=mapped_fields,
        )
        removed_terms.extend(terms)
        if should_remove:
            removed_lines.append(line)
            continue
        if guarded_text != bullet_text:
            kept_lines.append(f"{_bullet_prefix(line)}{guarded_text}")
        else:
            kept_lines.append(line)
        kept_bullet_count += 1

    if kept_bullet_count == 0:
        return "", removed_lines or lines, removed_terms
    return "\n".join(kept_lines), removed_lines, removed_terms


def remove_unsupported_narrative(text: str, *, mapped_fields: set[str]) -> tuple[str, dict[str, Any]]:
    blocks = [block for block in str(text or "").split("\n\n") if block.strip()]
    kept: list[str] = []
    removed: list[str] = []
    removed_terms: list[str] = []
    for block in blocks:
        bullet_result = _guard_markdown_bullet_block(block, mapped_fields=mapped_fields)
        if bullet_result is not None:
            guarded_block, removed_lines, terms = bullet_result
            removed_terms.extend(terms)
            if removed_lines:
                removed.extend(removed_lines)
            if guarded_block.strip():
                kept.append(guarded_block)
            continue

        guarded_block, should_remove, terms = _guard_text_fragment(
            block,
            mapped_fields=mapped_fields,
        )
        removed_terms.extend(terms)
        if should_remove:
            removed.append(block)
        else:
            kept.append(guarded_block)
    return "\n\n".join(kept), {
        "unsupported_narrative_removed": bool(removed),
        "removed_terms": sorted(set(removed_terms)),
        "removed_blocks": removed,
        "reason": "removed narrative blocks referencing unavailable canonical fields" if removed else "",
    }
