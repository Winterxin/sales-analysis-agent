from __future__ import annotations

from app.services.notebook_postrun_reflection import build_postrun_chart_reflections


class EscapedNewlineReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        return {
            "reflection_markdown": (
                "第一段已经引用 P99 2481.69 和中位数 54.49，说明均值会被少数大单拉高，"
                "因此需要区分常规订单和极端订单。\\n\\n"
                "第二段说明后续应筛选高销售额且利润异常的订单，避免少数极端订单掩盖真实经营结构。"
            )
        }


def test_postrun_chart_reflection_normalizes_literal_newline_escapes() -> None:
    chart_contexts = [
        {
            "section_id": "metric_distributions",
            "section_title": "指标分布与异常值分析",
            "chart_title": "核心数值指标分布画像",
            "table_preview": "Sales 9994.0 229.85 54.49 572.706 2481.6946",
            "code_source": "sns.histplot(metric_profile[col].dropna(), kde=True, ax=ax)",
            "cell_index": 6,
            "output_formats": ["image/png"],
            "chart_kind": "histogram",
            "chart_summary": "核心数值指标分布画像用于观察分布形态、偏态和长尾。",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=EscapedNewlineReflectionLLM(),
    )

    assert trace.status == "llm_applied"
    assert "\\n" not in reflections[6]
    assert "\n\n" in reflections[6]
