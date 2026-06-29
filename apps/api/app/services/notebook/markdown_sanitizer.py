from __future__ import annotations

import re

from app.schemas.schema_mapping import SchemaMapping
from app.services.evidence_guard import downgrade_strong_inference_claims
from app.services.output_language import contains_cjk, safe_english_sentence


def normalize_markdown_punctuation(text: str) -> str:
    cleaned = text
    replacements = {
        "。。": "。",
        "。；": "。",
        "，，": "，",
        "；。": "。",
    }
    previous = None
    while previous != cleaned:
        previous = cleaned
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
    return re.sub(r"%{2,}", "%", cleaned)


def remove_raw_value_duplicate_parentheses(text: str) -> str:
    pattern = re.compile(r"(?P<formatted>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?)\s*\((?P<raw>-?\d+(?:\.\d+)?)\)")

    def replace(match: re.Match[str]) -> str:
        formatted = match.group("formatted")
        raw = match.group("raw")
        try:
            if abs(float(formatted.replace(",", "")) - float(raw)) < 0.000001:
                return formatted
        except ValueError:
            return match.group(0)
        return match.group(0)

    cleaned = pattern.sub(replace, text)
    integer_pattern = re.compile(r"(?P<formatted>-?\d{1,3}(?:,\d{3})+)\s*\((?P<raw>-?\d+)\)")
    return integer_pattern.sub(replace, cleaned)


def clean_business_text(text: object) -> str:
    cleaned = str(text).strip()
    blocked = {"", "unknown", "none", "nan", "null"}
    if cleaned.lower() in blocked:
        return ""
    cleaned = re.sub(
        r"\b(?:unknown|none|nan|null)\b",
        "数据限制说明：缺少可验证字段",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?<![\d,])0\.(\d{2,4})(?![\d%])",
        lambda match: f"{float(match.group(0)) * 100:.2f}%",
        cleaned,
    )
    cleaned = remove_raw_value_duplicate_parentheses(cleaned)
    cleaned = normalize_markdown_punctuation(cleaned)
    return cleaned.replace("\ufffd", "").replace("?" * 4, "")


def schema_has_supporting_field(schema_mapping: SchemaMapping | None, terms: tuple[str, ...]) -> bool:
    if schema_mapping is None:
        return False
    for original_name, canonical_name in schema_mapping.field_mapping.items():
        haystack = f"{original_name} {canonical_name}".lower()
        if any(term.lower() in haystack for term in terms):
            return True
    return False


def sanitize_unqualified_business_speculation(
    text: str,
    schema_mapping: SchemaMapping | None = None,
) -> str:
    sanitized = text
    if not schema_has_supporting_field(schema_mapping, ("approval", "approve", "审批")):
        sanitized = sanitized.replace(
            "审批流程存在漏洞",
            "需结合审批记录进一步验证折扣流程是否存在漏洞",
        )
        sanitized = sanitized.replace(
            "折扣审批存在漏洞",
            "需结合审批记录进一步验证折扣流程是否存在漏洞",
        )
    if not schema_has_supporting_field(schema_mapping, ("cost", "成本", "cogs")):
        sanitized = sanitized.replace(
            "供应链成本过高",
            "需结合成本字段进一步判断是否存在成本压力",
        )
        sanitized = sanitized.replace(
            "成本失控",
            "需结合成本字段进一步判断是否存在成本压力",
        )
        sanitized = sanitized.replace(
            "履约成本过高",
            "需结合履约成本字段进一步验证履约压力",
        )
    if not schema_has_supporting_field(schema_mapping, ("campaign", "activity", "活动成本", "roi")):
        sanitized = sanitized.replace(
            "促销 ROI 不足",
            "需结合活动成本和转化数据验证促销 ROI",
        )
    if not schema_has_supporting_field(schema_mapping, ("ltv", "lifetime", "生命周期", "复购价值")):
        sanitized = sanitized.replace(
            "客户生命周期价值不足",
            "需结合 LTV 或长期价值字段进一步验证客户价值",
        )
    return sanitized


def remove_followup_question_text(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    skip_until_blank = False
    for line in lines:
        normalized = line.strip().lower()
        if "follow-up question" in normalized or "followup_question" in normalized or "后续追问" in line:
            skip_until_blank = True
            continue
        if skip_until_blank:
            if not line.strip():
                skip_until_blank = False
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def remove_template_discourse_markers(text: str) -> str:
    markers = r"(?:首先|其次|最后|第一|第二|第三)"
    cleaned = re.sub(
        rf"(?m)(^|\n)(\s*){markers}[，,、：:]\s*",
        lambda match: match.group(1) + match.group(2),
        text,
    )
    cleaned = re.sub(
        rf"([。！？；：:]\s*){markers}[，,、：:]\s*",
        r"\1",
        cleaned,
    )
    cleaned = re.sub(
        r"(?m)(^|\n)(\s*)(?:看到什么|说明什么|下一步做什么)[：:]\s*",
        lambda match: match.group(1) + match.group(2),
        cleaned,
    )
    cleaned = re.sub(r"(?:看到什么|说明什么|下一步做什么)[：:]\s*", "", cleaned)
    cleaned = re.sub(r"(?:这张图主要展示|当前图表用于观察)[，,：:\s]*", "", cleaned)
    return cleaned


def _remove_terminal_incomplete_cjk_ellipsis_from_line(line: str) -> str:
    stripped = line.rstrip()
    trailing = line[len(stripped) :]
    if not stripped or not contains_cjk(stripped) or not stripped.endswith("…"):
        return line
    delimiter_positions = [stripped.rfind(mark, 0, len(stripped) - 1) for mark in "，,；;：:"]
    breakpoint = max(delimiter_positions)
    if breakpoint >= 0:
        kept = stripped[:breakpoint].rstrip()
        return (kept.rstrip("。！？!?") + "。" + trailing) if kept else ""
    sentence_positions = [stripped.rfind(mark, 0, len(stripped) - 1) for mark in "。！？!?"]
    sentence_breakpoint = max(sentence_positions)
    if sentence_breakpoint >= 0:
        return stripped[: sentence_breakpoint + 1].rstrip() + trailing
    return ""


def _remove_terminal_incomplete_cjk_ellipsis(text: str) -> str:
    cleaned_lines = [
        _remove_terminal_incomplete_cjk_ellipsis_from_line(line) if line.strip() else line
        for line in text.splitlines()
    ]
    if text.endswith(("\n", "\r")):
        return "\n".join(cleaned_lines) + "\n"
    return "\n".join(cleaned_lines)


def sanitize_final_markdown_text(text: str, schema_mapping: SchemaMapping | None = None) -> str:
    sanitized = re.sub(
        r"[^。\n]*\b(?:unknown|none|nan|null)\b\s*(?:\([^)]*\))?[^。\n]*。?",
        "数据限制说明：该结论缺少可验证字段，已避免输出未知数值。",
        text,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"\b(?:unknown|none|nan|null)\b",
        "数据限制说明：缺少可验证字段",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = remove_followup_question_text(sanitized)
    sanitized = sanitize_unqualified_business_speculation(sanitized, schema_mapping)
    sanitized = downgrade_strong_inference_claims(sanitized)
    sanitized = sanitized.replace("当前更值得关注的是", "需要复核的是")
    sanitized = sanitized.replace("值得关注的是", "需要复核的是")
    sanitized = sanitized.replace("值得关注", "需要复核")
    sanitized = sanitized.replace("需要进一步分析", "需要结合明细继续验证")
    sanitized = remove_raw_value_duplicate_parentheses(sanitized)
    sanitized = normalize_markdown_punctuation(sanitized)
    sanitized = _remove_terminal_incomplete_cjk_ellipsis(sanitized)
    return sanitized.replace("\ufffd", "").replace("?" * 4, "").strip()


def paragraph_text(block: str) -> str:
    lines = [line.strip() for line in block.strip().splitlines()]
    body = [line for line in lines if line and not line.startswith("#")]
    return "\n".join(body).strip()


def trim_markdown_paragraph(
    text: str,
    max_chars: int = 180,
    *,
    require_complete_sentence: bool = False,
) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if not contains_cjk(cleaned):
        sentence_match = re.match(r"^(.{1,%d}?[.!?])(?:\s|$)" % max_chars, cleaned)
        if sentence_match:
            return sentence_match.group(1).strip()
        if require_complete_sentence:
            return safe_english_sentence(cleaned, fallback="", require_complete_sentence=True)
        cut = cleaned[:max_chars].rstrip(" ,;:-")
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0].rstrip(" ,;:-")
        cut = re.sub(r"\b[A-Za-z]{1,4}$", "", cut).rstrip(" ,;:-")
        if not cut:
            cut = cleaned[:max_chars].rstrip(" ,;:-")
        return cut.rstrip(".!?") + "."
    cut = cleaned[:max_chars].rstrip("，、；： ")
    return cut + "。"


_clean_business_text = clean_business_text
_schema_has_supporting_field = schema_has_supporting_field
_sanitize_unqualified_business_speculation = sanitize_unqualified_business_speculation
_remove_followup_question_text = remove_followup_question_text
_remove_template_discourse_markers = remove_template_discourse_markers
_sanitize_final_markdown_text = sanitize_final_markdown_text
_paragraph_text = paragraph_text
_trim_markdown_paragraph = trim_markdown_paragraph
_normalize_markdown_punctuation = normalize_markdown_punctuation
_remove_raw_value_duplicate_parentheses = remove_raw_value_duplicate_parentheses
