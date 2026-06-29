from __future__ import annotations

from app.services.notebook.markdown_sanitizer import sanitize_final_markdown_text


def test_sanitize_final_markdown_removes_terminal_cjk_ellipsis_regression_sample() -> None:
    source = (
        "模型预测的主要信号集中在国家、子类目和销售额特征组。"
        "业务复盘时，应结合这些线索，重点审视法国市场的订单、帽类（Caps）等子类目的商品利润结构，"
        "以及高销售额订单是…"
    )

    cleaned = sanitize_final_markdown_text(source)

    assert "以及高销售额订单是" not in cleaned
    assert not cleaned.endswith("…")
    assert "重点审视法国市场的订单、帽类（Caps）等子类目的商品利润结构。" in cleaned


def test_sanitize_final_markdown_removes_generic_terminal_cjk_ellipsis_fragment() -> None:
    source = "需要优先复核利润较低的商品，以及后续订单是…"

    assert sanitize_final_markdown_text(source) == "需要优先复核利润较低的商品。"


def test_sanitize_final_markdown_keeps_complete_chinese_sentence_unchanged() -> None:
    source = "利润率低于阈值，需要人工复核。"

    assert sanitize_final_markdown_text(source) == source


def test_sanitize_final_markdown_keeps_middle_unicode_ellipsis_unchanged() -> None:
    source = "字段A…字段B仍需结合明细复核。"

    assert sanitize_final_markdown_text(source) == source


def test_sanitize_final_markdown_keeps_english_three_dot_ellipsis_unchanged() -> None:
    source = "The details are still loading..."

    assert sanitize_final_markdown_text(source) == source
