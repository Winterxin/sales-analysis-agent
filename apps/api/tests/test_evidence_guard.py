from __future__ import annotations

from app.services.evidence_guard import downgrade_strong_inference_claims


def test_evidence_guard_downgrades_strong_inference_phrases() -> None:
    text = "折扣审批失控是核心原因，必然导致利润质量恶化，并充分证明这是决定性因素。"

    guarded = downgrade_strong_inference_claims(text)

    assert "折扣审批或折扣策略需结合明细进一步验证" in guarded
    assert "可能的重要风险信号" in guarded
    assert "可能加剧" in guarded
    assert "当前证据提示" in guarded
    assert "关键复核因素" in guarded
    assert "折扣审批失控" not in guarded
    assert "核心原因" not in guarded
    assert "必然导致" not in guarded


def test_evidence_guard_preserves_numbers_thresholds_and_valid_conclusions() -> None:
    text = "30%+ 是高风险折扣区间，负利润率为 18.72%，AUC 为 0.86，利润质量恶化，需要优先复盘。"

    guarded = downgrade_strong_inference_claims(text)

    assert "30%+" in guarded
    assert "18.72%" in guarded
    assert "0.86" in guarded
    assert "高风险折扣区间" in guarded
    assert "利润质量恶化" in guarded
    assert "需要优先复盘" in guarded


def test_evidence_guard_does_not_rewrite_code_fences() -> None:
    text = (
        "正文判断：核心原因需要降级。\n\n"
        "```python\n"
        "label = '核心原因'\n"
        "print('必然导致')\n"
        "```\n"
    )

    guarded = downgrade_strong_inference_claims(text)

    assert "正文判断：可能的重要风险信号需要降级。" in guarded
    assert "label = '核心原因'" in guarded
    assert "print('必然导致')" in guarded
