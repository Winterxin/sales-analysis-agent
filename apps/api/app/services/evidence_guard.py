from __future__ import annotations


STRONG_INFERENCE_REPLACEMENTS = (
    ("折扣审批失控", "折扣审批或折扣策略需结合明细进一步验证"),
    ("审批失控", "审批策略需结合明细进一步验证"),
    ("充分证明", "当前证据提示"),
    ("核心原因", "可能的重要风险信号"),
    ("根本原因", "需要优先验证的潜在原因"),
    ("必然导致", "可能加剧"),
    ("直接导致", "可能推动"),
    ("证明了", "提示"),
    ("决定性因素", "关键复核因素"),
    ("唯一原因", "可能原因之一"),
    ("一定会", "可能会"),
    ("完全由", "可能与"),
)


def _downgrade_segment(text: str) -> str:
    guarded = text
    for source, target in STRONG_INFERENCE_REPLACEMENTS:
        guarded = guarded.replace(source, target)
    return guarded


def downgrade_strong_inference_claims(text: str) -> str:
    """Downgrade unsupported causal/attribution wording without rewriting evidence."""
    if not text:
        return text
    parts = text.split("```")
    for index in range(0, len(parts), 2):
        parts[index] = _downgrade_segment(parts[index])
    return "```".join(parts)
