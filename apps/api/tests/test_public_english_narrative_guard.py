from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan, NotebookSectionContent
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.client_report_builder import (
    build_client_report_payload_with_trace,
    build_client_report_view_model,
    render_client_report_html,
)
from app.services.final_synthesis_guard import (
    guard_english_final_synthesis,
    guard_english_public_narrative,
    _split_english_public_sentences,
)
from app.services.notebook.summary_builder import build_final_conclusion_markdown
from app.services.notebook.section_renderer import section_conclusion_markdown
from app.services.notebook_builder import build_notebook
from app.services.output_language import safe_english_sentence
from app.services.notebook_postrun_reflection import (
    apply_postrun_chart_reflections,
    build_postrun_chart_reflections,
)


def _schema() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Category": "category",
            "Discount Bucket": "discount_bucket",
            "Segment": "segment",
            "Region": "region",
            "Order Date": "order_datetime",
        },
        confidence=0.95,
    )


def _report() -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["Total sales reached 2,326,534.35."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend Analysis",
                chart_type="line",
                summary_metrics={"total_sales_amount": 2326534.35},
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="Discount and Profit Analysis",
                chart_type="bar",
                summary_metrics={"total_profit_amount": 292296.81, "negative_profit_rate": 0.1865},
                tables={
                    "discount_threshold_candidates": [
                        {
                            "Category": "Furniture",
                            "discount_bucket": "30%+",
                            "avg_profit": -44169.46,
                            "profit_margin": -0.4590,
                            "negative_profit_rate": 0.9781,
                        },
                        {
                            "category": "Furniture",
                            "profit": 19730.00,
                            "profit_margin": 0.0261,
                        },
                    ]
                },
            ),
        ],
    )


def _raw_final_synthesis() -> dict[str, object]:
    return {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "Profit erosion appears to be a primary driver of financial loss.",
                "evidence": "Discounting is directly eroding overall margin.",
                "business_meaning": "Financial drain should be reversed.",
                "display_text": "Financial drain should be reversed.",
            },
            {
                "conclusion": "Furniture in the 30%+ discount tier had a -45.90% margin.",
                "evidence": "The scoped evidence row includes Category and discount_bucket.",
                "business_meaning": "Keep this as scoped threshold evidence.",
            },
            {
                "conclusion": "The current discount tier is associated with elevated loss risk.",
                "evidence": "The threshold analysis shows elevated loss risk.",
                "business_meaning": "Association is allowed; causality is not asserted.",
            },
        ],
        "recommended_actions": [
            {
                "priority": "P1",
                "action": "Implement a mandatory approval workflow.",
                "issue": "Enforce discount governance with approval gates.",
                "linked_metric_or_segment": "Renegotiate supplier terms.",
                "expected_use": "Route the top 10% of flagged orders to a dedicated review team.",
                "display_text": "Reverse the financial drain with an approval red line.",
            },
            {
                "priority": "P2",
                "action": "Launch a profitability turnaround project for the Furniture category.",
                "issue": "Furniture category requires a broad stop-sell review.",
                "linked_metric_or_segment": "Furniture category loss.",
                "expected_use": "Treat the scoped row as category-wide proof.",
                "display_text": "Launch a profitability turnaround project for the Furniture category.",
            },
            {
                "priority": "P3",
                "action": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
                "issue": "30%+ discount tier requires review",
                "linked_metric_or_segment": "30%+ discount tier.",
                "expected_use": "Keep the action evidence-bound.",
                "display_text": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
            },
        ],
    }


def _assert_no_public_leaks(text: str) -> None:
    forbidden = [
        "primary driver of financial loss",
        "directly eroding overall margin",
        "financial drain",
        "mandatory approval workflow",
        "approval gates",
        "approval red line",
        "supplier terms",
        "top 10% of flagged orders",
        "dedicated review team",
        "profitability turnaround project",
        "Furniture category loss",
        "category-wide proof",
        "High discount tiers are driving margin risk",
    ]
    for phrase in forbidden:
        assert phrase.lower() not in text.lower()
    assert "…" not in text
    assert not re.search(r"[\u3400-\u9fff]", text)


PUBLIC_ARTIFACT_FORBIDDEN_PHRASES = [
    "shows (",
    "blocked_fields",
    "field_mapping",
    "dtype",
    "columns",
    "Index(",
    "DataFrame",
    "order_day Sales Quantity",
    "sales_rolling_mean",
    "quantity_rolling_mean",
    "Discount Profit Sales Category Segment Region",
    "approval logs",
    "approval red lines",
    "approval workflow",
    "authorization process",
    "systemic control gap",
    "control gaps",
    "directly tied to",
    "tied directly to",
    "not being effectively managed",
    "not leveraging",
    "governance is weak",
    "strategy failure",
    "directly destroying profitability",
    "directly destroys profitability",
    "directly undermining profitability",
    "directly undermines profitability",
    "**Discount vs.",
    "needed to cover.",
    "significantly erode profitability",
    "erode profitability",
    "erodes profitability",
    "may drive the peaks",
    "drive the peaks",
    "operational decisions",
    "one-off large orders",
    "profit destroyer",
    "directly undermining margins",
    "recent movement against the available trend baseline",
    "peak and trough periods",
    "stable growth or decline",
    "high cost of goods",
    "shipping expenses",
    "fulfillment cost",
    "diagnose causes such as",
    "contributing to significant losses",
    "almost guarantee losses",
    "almost guarantees losses",
    "almost guaranteed losses",
    "almost guarantee a loss",
    "almost guarantees a loss",
    "almost guarantee negative profit",
    "almost guarantees negative profit",
    "near-guaranteed loss",
    "near guaranteed loss",
    "virtually guarantees losses",
    "visible relationship between the plotted measures",
    "high-value and low-value clusters",
    "automatically flag",
    "automatic routing",
    "automatic approval",
    "automatic blocking",
    "automatic execution",
    "automated workflow",
    "order processing system",
    "order-processing workflow",
    "before fulfillment",
    "mandatory human review",
    "mandatory review process",
    "top 7% highest-risk",
    "top 5% highest-risk",
    "discount cap profit improvement",
    "weakening the effectiveness of",
    "most financially damaging orders",
    "proves that the model",
    "cluster within Furniture",
    "cost burden",
    "discount burden",
    "high-discount segments",
    "discount-profit correlation",
    "negative profit orders",
    "directly contributes",
    "60–70%",
    "60-70%",
    "daily list",
    "weekly list",
    "monthly list",
    "daily review",
    "weekly review",
    "daily audit",
    "weekly audit",
    "operations team",
    "audit workflow",
    "review workflow",
    "integrate into operations",
    "continuous risk management",
    "before final processing",
    "potentially saving significant margin",
    "saving significant margin",
    "mitigate losses operationally",
    "compares model recall with review workload across threshold choices",
    "across threshold choices",
    "proactive risk management",
    "reactive analysis",
    "intercept",
    "before they ship",
    "before shipping",
    "operational safeguard",
    "safeguard against profit erosion",
    "protect margin",
    "protect profit",
    "almost perfectly correlated",
    "near-perfect correlation",
    "highly correlated",
    "primary, clear trigger",
    "converting a sale into a loss",
    "profit protection",
    "strongly associated",
    "highly associated",
    "almost guaranteed",
    "guaranteed to result in a loss",
    "Discount vs. Review",
]

HANGING_TAIL_RE = re.compile(
    r"\b(?:while|and|or|with|to|of|for|in|at|by|from|the|a|an)\.(?:\s|$)",
    flags=re.IGNORECASE,
)


def _assert_public_scan_passes(text: str) -> None:
    assert "…" not in text
    assert not HANGING_TAIL_RE.search(text)
    for phrase in PUBLIC_ARTIFACT_FORBIDDEN_PHRASES:
        assert phrase.lower() not in text.lower()


def _assert_balanced_fixture_bold(text: str) -> None:
    assert text.count("**") % 2 == 0


def _html_visible_text(html: str) -> str:
    text = re.sub(r"(?is)<style.*?</style>", " ", html)
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def _notebook_markdown_text(path: Path) -> str:
    return "\n\n".join(
        str(cell.source)
        for cell in nbformat.read(path, as_version=4).cells
        if cell.cell_type == "markdown"
    )


def test_english_final_synthesis_items_are_atomically_guarded_across_public_outputs() -> None:
    report = _report()
    schema = _schema()

    guarded = guard_english_final_synthesis(_raw_final_synthesis(), report=report, schema_mapping=schema)
    guarded_again = guard_english_final_synthesis(guarded, report=report, schema_mapping=schema)
    assert guarded_again == guarded

    notebook_markdown = build_final_conclusion_markdown(
        report,
        schema,
        dataset_profile={"row_count": 999},
        final_synthesis=guarded,
        output_language="en",
    )
    client_payload, _trace = build_client_report_payload_with_trace(
        report=report,
        schema_mapping=schema,
        dataset_profile={"row_count": 999, "column_count": 8},
        analysis_focus=None,
        evidence_pack=None,
        section_priority=None,
        llm_client=None,
        final_synthesis=guarded,
        output_language="en",
    )
    html = render_client_report_html(client_payload, output_language="en")
    combined = "\n".join(
        [
            json.dumps(guarded, ensure_ascii=False),
            notebook_markdown,
            json.dumps(client_payload, ensure_ascii=False),
            html,
        ]
    )

    _assert_no_public_leaks(combined)
    assert "Furniture in the 30%+ discount tier had a -45.90% margin" in combined
    assert "associated with elevated loss risk" in combined
    assert "Review approval records if available before changing approval controls." in combined
    assert "Review the scoped discount tier against profit outcomes before broader pricing changes." in combined

    first_action = guarded["recommended_actions"][0]
    action_text = json.dumps(first_action, ensure_ascii=False)
    _assert_no_public_leaks(action_text)
    assert first_action["action"] == first_action["display_text"]
    assert first_action["expected_use"] == first_action["display_text"]
    assert "approval" in first_action["action"].lower()


class UnsafeEnglishPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown, language_instruction=None):
        return {
            "reflection_markdown": (
                "High discounts mechanically compress margins and fail to generate compensating volume gains. "
                "Supplier cost inflation and weak approval controls are causing the issue."
            )
        }


class CjkEnglishPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown, language_instruction=None):
        return {"reflection_markdown": "这张图显示销售趋势存在波动，需要继续复核。"}


def test_english_postrun_chart_commentary_uses_public_evidence_guard(tmp_path: Path) -> None:
    report = _report()
    schema = _schema()
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "Discount and Profit Analysis",
            "chart_title": "Discount Tier Profit Quality",
            "table_preview": "30%+ -45.90% 97.81%",
            "code_source": "fig = px.bar(...)",
            "cell_index": 1,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "The 30%+ discount tier shows weaker profit quality in the chart evidence.",
        }
    ]

    reflections, _trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        schema_mapping=schema,
        llm_client=UnsafeEnglishPostrunReflectionLLM(),
        allow_fallback_reflections=True,
        output_language="en",
    )
    notebook = new_notebook(cells=[new_markdown_cell("## Discount and Profit Analysis"), new_code_cell("fig.show()")])
    path = tmp_path / "analysis.executed.ipynb"
    path.write_text(nbformat.writes(notebook), encoding="utf-8")
    apply_postrun_chart_reflections(path, path, reflections, output_language="en")
    markdown = "\n\n".join(str(cell.source) for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == "markdown")

    forbidden = [
        "mechanically",
        "compensating volume",
        "supplier cost inflation",
        "approval controls",
        "root cause",
        "financial drain",
    ]
    for phrase in forbidden:
        assert phrase.lower() not in markdown.lower()
    assert "Chart Commentary" in markdown
    assert "executed chart" in markdown or "30%+" in markdown or "profit quality" in markdown
    assert not re.search(r"[\u3400-\u9fff]", markdown)
    assert "…" not in markdown


def test_english_chart_commentary_guard_preserves_decimal_tokens() -> None:
    text = (
        "The scatter plot shows a correlation of -0.2189. "
        "The 30%+ discount tier has a 97.81% negative-profit rate and average profit of -$105.91. "
        "Furniture margin is 2.61%, while average order value is $66.34. "
        "The benchmark reference value is 1,234.56."
    )

    guarded = guard_english_public_narrative(
        text,
        report=None,
        schema_mapping=None,
        chart_context={
            "chart_title": "Discount Profit Scatter",
            "chart_kind": "scatter",
            "chart_summary": "The chart compares discount and profit outcomes.",
        },
        role="chart_commentary",
    )

    for token in ("-0.2189", "97.81%", "-$105.91", "$66.34", "2.61%", "1,234.56"):
        assert token in guarded
    for broken in ("-0. 2189", "97. 81%", "-$105. 91", "$66. 34"):
        assert broken not in guarded
    assert not re.search(r"(?<![\d,])2\.(?!\d)", guarded)


def test_english_public_sentence_splitter_preserves_vs_markdown_bold_title() -> None:
    text = (
        "The model should be viewed alongside the **Discount vs. Profit Relationship**. "
        "The model's current recall requires manual review."
    )

    split_sentences = _split_english_public_sentences(text)
    assert split_sentences[0] == "The model should be viewed alongside the **Discount vs. Profit Relationship**."
    assert split_sentences[1] == "The model's current recall requires manual review."

    guarded = guard_english_public_narrative(
        text,
        report=None,
        schema_mapping=None,
        chart_context={
            "chart_title": "Discount vs. Profit Relationship",
            "chart_kind": "scatter",
            "chart_summary": "The chart compares discount and profit outcomes.",
        },
        role="chart_commentary",
    )

    assert "**Discount vs. Profit Relationship**" in guarded
    assert "**Discount vs." not in guarded.replace("**Discount vs. Profit Relationship**", "")
    assert "Profit Relationship**." not in guarded.replace("**Discount vs. Profit Relationship**", "")
    _assert_balanced_fixture_bold(guarded)
    assert "SENTINEL" not in guarded
    assert "ABBREVIATION" not in guarded


def test_english_chart_commentary_validate_does_not_bypass_unsafe_operations() -> None:
    text = (
        "Use this evidence to trigger a mandatory review process for new discount requests, "
        "then validate the what-if scenario."
    )

    guarded = guard_english_public_narrative(
        text,
        report=_report(),
        schema_mapping=_schema(),
        chart_context={
            "chart_title": "Discount Threshold Trade-off",
            "chart_kind": "dual_axis_bar_line",
            "chart_summary": "The current threshold analysis compares profit and loss-rate outcomes.",
        },
        role="chart_commentary",
    )

    forbidden = ("mandatory review", "new discount requests", "validate the what-if scenario")
    for phrase in forbidden:
        assert phrase not in guarded.lower()
    assert "scoped" in guarded.lower() or "observed relationship" in guarded.lower()
    assert "before" in guarded.lower()


def test_english_chart_commentary_guard_removes_strong_causality_fixture() -> None:
    text = (
        "The 30%+ discount tier is associated with weaker profit outcomes. "
        "High discounting erodes profit quality far more aggressively. "
        "Revenue volume does not protect margin when discount intensity crosses this threshold. "
        "The chart confirms that discounting is the primary driver."
    )

    guarded = guard_english_public_narrative(
        text,
        report=_report(),
        schema_mapping=_schema(),
        chart_context={
            "chart_title": "Discount Tier Profit Quality",
            "chart_kind": "bar",
            "chart_summary": "The 30%+ discount tier shows weaker profit quality.",
        },
        role="chart_commentary",
    )

    for phrase in ("erodes", "does not protect margin", "confirms", "primary driver"):
        assert phrase not in guarded.lower()
    assert (
        "associated with weaker profit outcomes" in guarded.lower()
        or "weaker profit" in guarded.lower()
    )


def test_english_chart_commentary_guard_removes_direct_profitability_destruction_variants() -> None:
    text = (
        "The 30%+ discount tier is associated with elevated loss risk. "
        "The current discounting strategy is directly destroying profitability. "
        "It is directly undermining margins."
    )

    guarded = guard_english_public_narrative(
        text,
        report=_report(),
        schema_mapping=_schema(),
        chart_context={
            "chart_title": "Discount vs. Profit Relationship",
            "chart_kind": "scatter",
            "chart_summary": "The chart compares discount and profit outcomes.",
        },
        role="chart_commentary",
    )

    _assert_public_scan_passes(guarded)
    assert "directly destroying" not in guarded.lower()
    assert "directly undermining" not in guarded.lower()
    assert "associated with elevated loss risk" in guarded.lower()


def test_english_public_guard_removes_near_certain_loss_claims_without_dropping_safe_evidence() -> None:
    text = (
        "Discounts above 20% almost guarantee losses. "
        "The 20%+ discount tiers are associated with weaker profit outcomes in the current analysis. "
        "Some review notes virtually guarantees losses when discounting is high."
    )

    guarded = guard_english_public_narrative(
        text,
        report=_report(),
        schema_mapping=_schema(),
        chart_context={
            "chart_title": "Discount vs. Profit Relationship",
            "chart_kind": "scatter",
            "chart_summary": "The chart compares discount and profit outcomes.",
        },
        role="chart_commentary",
    )

    _assert_public_scan_passes(guarded)
    for phrase in (
        "almost guarantee",
        "almost guarantees",
        "near-guaranteed",
        "virtually guarantees",
        "guarantee losses",
        "guarantees losses",
    ):
        assert phrase not in guarded.lower()
    assert "associated with weaker profit outcomes" in guarded.lower()


def test_english_chart_commentary_header_only_fallback_rejects_internal_tokens() -> None:
    chart_contexts = [
        {
            "section_id": "sales_trends",
            "section_title": "Sales Trend Analysis",
            "chart_title": "Sales Trend with Rolling Average",
            "table_preview": "order_day Sales Quantity sales_rolling_mean quantity_rolling_mean\nblocked_fields",
            "code_source": "fig = px.line(...)",
            "cell_index": 18,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=CjkEnglishPostrunReflectionLLM(),
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[18]
    assert trace.status == "fallback_on_error"
    assert "order_day Sales Quantity sales_rolling_mean quantity_rolling_mean" not in reflection
    assert "blocked_fields" not in reflection
    assert "shows (" not in reflection
    assert not re.search(r"[\u3400-\u9fff]", reflection)
    assert "…" not in reflection
    assert len([part for part in re.split(r"[.!?]+", reflection) if part.strip()]) >= 2
    assert not re.search(r"\b\d[\d,.]*\b", reflection)


def test_english_notebook_section_markdown_does_not_emit_hanging_tail_words() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="Discount and Profit Analysis",
                chart_type="bar",
                findings=[
                    (
                        "The 20-30% tier shows an average profit of -$105.91 per order after comparing "
                        "repeat customer behavior regional results segment results product mix category slices "
                        "discount bands order volatility monthly sales patterns while the loss rate remains elevated."
                    ),
                    "The weak slice requires review while Consumer / Central remains below the regional benchmark.",
                ],
            )
        ],
    )
    section = NotebookSection(
        section_id="discount_and_profit",
        title="Discount and Profit Analysis",
        purpose="Review discount and profit outcomes.",
    )

    markdown = section_conclusion_markdown(section, report, _schema(), output_language="en")

    assert markdown is not None
    assert "while." not in markdown.lower()
    assert not HANGING_TAIL_RE.search(markdown)
    assert "-$105.91" in markdown
    assert (
        "while the loss rate remains elevated." in markdown
        or "mapped fields and module outputs define the current review scope." in markdown
    )
    assert not re.search(r"[\u3400-\u9fff]", markdown)
    assert "…" not in markdown


def test_safe_english_sentence_complete_mode_preserves_complete_sentence_without_hard_truncation() -> None:
    text = (
        "The 20-30% tier shows an average profit of -$105.91 per order after comparing "
        "repeat customer behavior regional results segment results product mix category slices "
        "discount bands order volatility monthly sales patterns while the loss rate remains elevated."
    )

    sentence = safe_english_sentence(
        text,
        fallback="mapped fields and module outputs define the current review scope.",
        require_complete_sentence=True,
    )

    assert "while." not in sentence.lower()
    assert not HANGING_TAIL_RE.search(sentence)
    assert "-$105.91" in sentence
    assert "while the loss rate remains elevated." in sentence
    assert not re.search(r"[\u3400-\u9fff]", sentence)
    assert "…" not in sentence


def test_english_chart_commentary_fallback_never_exposes_raw_table_preview() -> None:
    raw_preview = "Discount Profit Sales Category Segment Region"
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "Discount and Profit Analysis",
            "chart_title": "Discount vs. Profit Relationship",
            "table_preview": raw_preview,
            "code_source": "fig = px.scatter(...)",
            "cell_index": 9,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "scatter",
            "chart_summary": raw_preview,
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=CjkEnglishPostrunReflectionLLM(),
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[9]
    assert trace.status == "fallback_on_error"
    _assert_public_scan_passes(reflection)
    assert "Discount vs. Profit Relationship" in reflection
    assert len([part for part in re.split(r"[.!?]+", reflection) if part.strip()]) >= 2


def test_english_public_artifact_scan_covers_notebook_final_synthesis_and_client_report(tmp_path: Path) -> None:
    schema = _schema()
    report = _report()
    guarded = guard_english_final_synthesis(
        {
            "metadata": {"final_synthesis_used_llm": True},
            "main_conclusions": [
                {
                    "conclusion": "The 30%+ discount tier is loss-making, directly tied to high discounts.",
                    "evidence": "Discounting strategy is not being effectively managed to protect margins.",
                    "business_meaning": "The business is not leveraging its loyal customer base.",
                },
                {
                    "conclusion": "The current discounting strategy is directly destroying profitability.",
                    "evidence": "The discount policy directly undermines margins.",
                    "business_meaning": "Treat this as a direct profitability destruction finding.",
                },
                {
                    "conclusion": "Furniture in the 30%+ discount tier had a -45.90% margin.",
                    "evidence": "The scoped evidence row includes Category and discount_bucket.",
                    "business_meaning": "Keep this as scoped threshold evidence.",
                },
            ],
            "recommended_actions": [
                {
                    "priority": "P1",
                    "action": "Calibrate approval red lines from approval logs.",
                    "issue": "Systemic control gap in discount governance.",
                    "linked_metric_or_segment": "Approval workflow failure.",
                    "expected_use": "Fix governance is weak signal.",
                    "display_text": "Discount strategy failure is directly tied to margin loss.",
                }
            ],
        },
        report=report,
        schema_mapping=schema,
    )
    final_markdown = build_final_conclusion_markdown(
        report,
        schema,
        dataset_profile={"row_count": 999},
        final_synthesis=guarded,
        output_language="en",
    )
    chart_contexts = [
        {
            "section_id": "sales_trends",
            "section_title": "Sales Trend Analysis",
            "chart_title": "Sales Trend with Rolling Average",
            "table_preview": "order_day Sales Quantity sales_rolling_mean quantity_rolling_mean\nblocked_fields",
            "code_source": "fig = px.line(...)",
            "cell_index": 1,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "order_day Sales Quantity sales_rolling_mean quantity_rolling_mean",
        }
    ]
    reflections, _trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        schema_mapping=schema,
        llm_client=CjkEnglishPostrunReflectionLLM(),
        allow_fallback_reflections=False,
        output_language="en",
    )
    notebook_path = tmp_path / "analysis.ipynb"
    notebook = new_notebook(cells=[new_markdown_cell("## Sales Trend Analysis"), new_code_cell("fig.show()")])
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")
    apply_postrun_chart_reflections(notebook_path, notebook_path, reflections, output_language="en")
    notebook_markdown = "\n\n".join(
        str(cell.source)
        for cell in nbformat.read(notebook_path, as_version=4).cells
        if cell.cell_type == "markdown"
    )
    client_payload, _client_trace = build_client_report_payload_with_trace(
        report=report,
        schema_mapping=schema,
        dataset_profile={"row_count": 999, "column_count": 8},
        analysis_focus=None,
        evidence_pack=None,
        section_priority=None,
        llm_client=None,
        final_synthesis=guarded,
        output_language="en",
    )
    client_html = render_client_report_html(client_payload, output_language="en")
    combined = "\n".join(
        [
            notebook_markdown,
            final_markdown,
            json.dumps(guarded, ensure_ascii=False),
            json.dumps(client_payload, ensure_ascii=False),
            _html_visible_text(client_html),
        ]
    )

    _assert_public_scan_passes(combined)
    assert "Furniture in the 30%+ discount tier had a -45.90% margin" in combined
    assert "Furniture overall" not in combined
    assert "Review approval records if available before changing approval controls." in combined


def test_fullish_english_notebook_public_markdown_scan_covers_section_and_postrun_paths(tmp_path: Path) -> None:
    schema = _schema()
    report = _report()
    outline = NotebookOutline(
        title="Sales Data Analysis Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category Analysis",
                purpose="Review product contribution and category quality.",
            ),
            NotebookSection(
                section_id="conclusions",
                title="Conclusions and Recommended Actions",
                purpose="Summarize final actions.",
            ),
        ],
    )
    long_intro = (
        "The Pareto chart shows that the top 20 products contribute 12.94% of sales across multiple "
        "customer segments, regions, category groups, and product families in the current portfolio, "
        "while 424 products are needed to cover 80% of sales."
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="product_and_category",
                markdown_blocks=[
                    long_intro,
                    (
                        "### Business Takeaway\n\n"
                        "Discounts above 20% significantly erode profitability. "
                        "Review the relationship against order-level evidence before broader pricing changes."
                    ),
                    (
                        "### 图表分析\n\n"
                        "The plotted pattern is a profit destroyer. "
                        "Short-term operational decisions or one-off large orders may drive the peaks. "
                        "The Discount vs. Profit Relationship should be reviewed with -0.2189 and $66.34."
                    ),
                ],
                code_cells=[
                    "chart_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'threshold_tradeoff'}]}\nfig = None\nfig"
                ],
            ),
            NotebookSectionContent(
                section_id="conclusions",
                markdown_blocks=["Conclusion intro."],
                code_cells=[],
            ),
        ]
    )
    final_synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "Furniture in the 30%+ discount tier had a -45.90% margin.",
                "evidence": "The scoped evidence row includes Category and discount_bucket.",
                "business_meaning": "Keep this as scoped threshold evidence.",
            }
        ],
        "recommended_actions": [
            {
                "priority": "P1",
                "action": "Implement a mandatory approval workflow.",
                "issue": "Approval workflow failure.",
                "linked_metric_or_segment": "Approval logs.",
                "expected_use": "Approval red lines.",
                "display_text": "Calibrate approval red lines from approval logs.",
            }
        ],
    }
    notebook_path = build_notebook(
        task_id="tatest-api-key",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema,
        plan=AnalysisPlan(),
        outline=outline,
        content_plan=content_plan,
        final_synthesis=final_synthesis,
        output_language="en",
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    chart_cell_index = next(
        index for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "threshold_tradeoff" in str(cell.source)
    )
    reflections, _trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "modeling",
                "section_title": "Modeling Analysis",
                "chart_title": "Threshold Trade-off Curve: Recall and Review Load",
                "table_preview": "threshold recall review_load\n0.42 0.66 120",
                "code_source": "fig = px.line(...)",
                "cell_index": chart_cell_index,
                "output_formats": ["application/vnd.plotly.v1+json"],
                "chart_kind": "line",
                "chart_summary": "",
            }
        ],
        report=report,
        schema_mapping=schema,
        llm_client=None,
        allow_fallback_reflections=True,
        output_language="en",
    )
    apply_postrun_chart_reflections(notebook_path, notebook_path, reflections, output_language="en")
    markdown = _notebook_markdown_text(notebook_path)

    _assert_public_scan_passes(markdown)
    assert "80% of sales" in markdown
    assert "Discount vs. Profit Relationship" in markdown
    assert "-0.2189" in markdown
    assert "$66.34" in markdown
    assert "Furniture in the 30%+ discount tier had a -45.90% margin" in markdown
    assert "Review approval records if available before changing approval controls." in markdown
    assert (
        "Threshold Trade-off Curve: Recall and Review Load summarizes model recall and review workload"
        in markdown
    )
    assert "manual review prioritization" in markdown


def test_english_final_synthesis_brief_actions_replace_cost_shipping_root_cause_leaks() -> None:
    guarded = guard_english_final_synthesis(
        {
            "metadata": {"final_synthesis_used_llm": True},
            "brief_actions": [
                {
                    "priority": "P1",
                    "action": (
                        "Isolate and investigate the top loss-making products to diagnose causes such as "
                        "high discounts, high cost of goods, or high shipping expenses."
                    ),
                    "linked_evidence": "High cost of goods and fulfillment cost are contributing to significant losses.",
                    "priority_reason": "Shipping expenses and fulfillment costs require root-cause diagnosis.",
                    "display_text": (
                        "Investigate high cost of goods, high shipping expenses, and fulfillment cost "
                        "as root causes contributing to losses."
                    ),
                }
            ],
        },
        report=_report(),
        schema_mapping=_schema(),
    )
    serialized = json.dumps(guarded, ensure_ascii=False)

    _assert_public_scan_passes(serialized)
    assert guarded["brief_actions"][0]["action"] == "Add cost or supplier data before identifying root causes."
    assert guarded["brief_actions"][0]["display_text"] == "Add cost or supplier data before identifying root causes."
    assert guarded["brief_actions"][0]["priority_reason"] == "Add cost or supplier data before identifying root causes."


def test_english_final_synthesis_scope_fallback_stays_scoped() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "recommended_actions": [
            {
                "priority": "P2",
                "action": "Expand the Furniture 30%+ discount finding into a full category pricing policy.",
                "issue": "Furniture category loss requires broad promotion changes.",
                "linked_metric_or_segment": "Furniture category loss.",
                "expected_use": "Treat the scoped row as category-wide proof.",
                "display_text": "Change Furniture category pricing and campaign policy from the scoped discount row.",
            }
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    action = guarded["recommended_actions"][0]
    combined = json.dumps(action, ensure_ascii=False)

    assert "Scoped evidence must not be expanded into broader action." in combined
    assert "Review the scoped evidence before changing broader category, region, product, or pricing policies." in combined
    assert "campaign" not in combined.lower()
    assert "promotion" not in combined.lower()
    assert "inventory" not in combined.lower()
    assert "category-wide proof" not in combined.lower()


def test_english_final_synthesis_stale_guard_metadata_does_not_skip_model_action_seal() -> None:
    raw = {
        "metadata": {
            "final_synthesis_used_llm": True,
            "english_public_narrative_guard_applied": True,
        },
        "recommended_actions": [
            {
                "priority": "P1",
                "action": "Deploy the risk model for order auditing.",
                "issue": "Model scores need operational workflow integration.",
                "linked_metric_or_segment": "Loss-risk model scores.",
                "linked_evidence": "Model output supports review triage.",
                "expected_use": "Focus audit resources on the highest-risk orders.",
                "priority_reason": "The model can rank order risk.",
                "display_text": (
                    "Integrate risk scores into an order-processing workflow to automatically flag "
                    "the top 7% highest-risk orders for mandatory human review before fulfillment."
                ),
            }
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    guarded_again = guard_english_final_synthesis(guarded, report=_report(), schema_mapping=_schema())
    serialized = json.dumps(guarded, ensure_ascii=False)
    action = guarded["recommended_actions"][0]
    action_text = json.dumps(action, ensure_ascii=False)

    assert guarded == guarded_again
    _assert_public_scan_passes(serialized)
    for phrase in (
        "automatically",
        "automatic flag",
        "order processing",
        "workflow",
        "top 7%",
        "mandatory human review",
        "before fulfillment",
    ):
        assert phrase not in serialized.lower()
    assert "review threshold" in action_text.lower() or "manual review" in action_text.lower()
    assert "review capacity" in action_text.lower() or "precision-recall" in action_text.lower()
    assert action["action"] == action["display_text"]
    assert action["expected_use"] == action["display_text"]


def test_english_model_workflow_and_cadence_actions_are_atomically_replaced() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "recommended_actions": [
            {
                "priority": "P1",
                "action": "Use the trained risk model to generate a daily list of high-risk orders for manual audit.",
                "issue": "The model needs daily manual audit cadence.",
                "linked_metric_or_segment": "Loss-risk model scores.",
                "expected_use": "Create a daily list for audit owners.",
                "priority_reason": "Daily review would support continuous risk management.",
                "display_text": "Use the trained risk model to generate a daily list of high-risk orders for manual audit.",
            },
            {
                "priority": "P2",
                "action": "Deploy the loss-risk model into a weekly audit workflow.",
                "issue": "Weekly audit workflow is needed.",
                "linked_metric_or_segment": "Loss-risk model.",
                "expected_use": "Deploy the model into a weekly audit workflow.",
                "priority_reason": "The operations team can use the workflow.",
                "display_text": "Deploy the loss-risk model into a weekly audit workflow.",
            },
            {
                "priority": "P3",
                "action": "The model must not be used for automatic decisions, but automatically flag orders before fulfillment.",
                "issue": "Mixed limitation and automation.",
                "linked_metric_or_segment": "Risk score.",
                "expected_use": "Automatically flag orders before fulfillment.",
                "priority_reason": "Automation should route risk.",
                "display_text": "The model must not be used for automatic decisions, but automatically flag orders before fulfillment.",
            },
            {
                "priority": "P4",
                "action": "Review weekly sales trend before adjusting forecasts.",
                "issue": "Forecast review cadence comes from sales trend monitoring.",
                "linked_metric_or_segment": "Weekly sales trend.",
                "expected_use": "Use the trend review before forecast changes.",
                "display_text": "Review weekly sales trend before adjusting forecasts.",
            },
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    serialized = json.dumps(guarded, ensure_ascii=False)

    assert "Review weekly sales trend before adjusting forecasts." in serialized
    forbidden = (
        "daily",
        "weekly audit",
        "monthly",
        "operations",
        "team",
        "workflow",
        "deploy",
        "integrate",
        "automatic",
        "automatically",
        "mandatory",
        "top 7%",
        "before fulfillment",
        "order processing",
        "continuous risk management",
    )
    for action in guarded["recommended_actions"][:3]:
        action_text = json.dumps(action, ensure_ascii=False).lower()
        for phrase in forbidden:
            assert phrase not in action_text
        assert "manual review" in action_text
        assert "review capacity" in action_text
        assert "false-positive" in action_text or "precision-recall" in action_text


def test_english_model_main_conclusion_workflow_claim_is_atomically_replaced() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": (
                    "A high-performing predictive model is available to identify at-risk orders, "
                    "providing a practical tool to mitigate losses operationally."
                ),
                "evidence": "LogisticRegression validation shows recall and ROC AUC evidence.",
                "business_meaning": (
                    "Instead of reactive loss discovery, the company can proactively identify and review "
                    "high-risk orders before final processing, potentially saving significant margin."
                ),
            },
            {
                "conclusion": "Review weekly sales trend before adjusting forecasts.",
                "evidence": "Weekly sales trend evidence remains a non-model monitoring input.",
                "business_meaning": "Use trend review before forecast changes.",
            },
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    guarded_again = guard_english_final_synthesis(guarded, report=_report(), schema_mapping=_schema())
    serialized = json.dumps(guarded, ensure_ascii=False)
    first = guarded["main_conclusions"][0]

    assert guarded_again == guarded
    assert first == {
        "conclusion": "The loss-risk model can support manual review prioritization.",
        "evidence": "Use observed model validation evidence only to compare review trade-offs.",
        "business_meaning": (
            "Select a review threshold based on review capacity and the observed "
            "false-positive / false-negative trade-off."
        ),
    }
    _assert_public_scan_passes(serialized)
    for phrase in (
        "before final processing",
        "potentially saving significant margin",
        "saving significant margin",
        "mitigate losses operationally",
        "order processing",
        "workflow",
        "deploy",
        "integrate",
        "automatic",
        "mandatory",
    ):
        assert phrase not in serialized.lower()
    assert "manual review prioritization" in serialized
    assert "review capacity" in serialized
    assert "false-positive / false-negative trade-off" in serialized
    assert "Review weekly sales trend before adjusting forecasts." in serialized


def test_english_model_main_conclusion_replacement_reaches_public_rendering() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "The predictive model can mitigate losses operationally.",
                "evidence": "Loss-risk model recall and ROC AUC were observed.",
                "business_meaning": (
                    "Use high-risk orders before final processing, potentially saving significant margin."
                ),
            }
        ],
        "recommended_actions": [
            {
                "priority": "P3",
                "action": "Use the model risk scores only for manual review prioritization.",
                "issue": "Model output requires bounded use.",
                "linked_metric_or_segment": "Loss-risk model.",
                "expected_use": "Select a review threshold based on review capacity.",
                "display_text": "Use the model risk scores only for manual review prioritization.",
            }
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    final_markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        dataset_profile={"row_count": 999},
        final_synthesis=guarded,
        output_language="en",
    )
    client_payload, _client_trace = build_client_report_payload_with_trace(
        report=_report(),
        schema_mapping=_schema(),
        dataset_profile={"row_count": 999, "column_count": 8},
        analysis_focus=None,
        evidence_pack=None,
        section_priority=None,
        llm_client=None,
        final_synthesis=guarded,
        output_language="en",
    )
    client_html = render_client_report_html(client_payload, output_language="en")
    public_texts = [
        json.dumps(guarded, ensure_ascii=False),
        final_markdown,
        json.dumps(client_payload, ensure_ascii=False),
        _html_visible_text(client_html),
    ]

    for text in public_texts:
        _assert_public_scan_passes(text)
        assert "manual review prioritization" in text.lower() or "review threshold" in text.lower()


def test_english_model_main_conclusion_semantic_bypass_is_atomically_replaced() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "The LogisticRegression loss-risk model supports a transition from reactive analysis to proactive risk management.",
                "evidence": "Observed validation metrics describe model recall and ROC AUC.",
                "business_meaning": (
                    "Use the predictive model to systematically identify and intercept high-risk orders "
                    "before they ship, creating a measurable safeguard against profit erosion."
                ),
            },
            {
                "conclusion": "Review weekly sales trend before adjusting forecasts.",
                "evidence": "Weekly sales trend evidence remains a non-model monitoring input.",
                "business_meaning": "Use trend review before forecast changes.",
            },
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    guarded_again = guard_english_final_synthesis(guarded, report=_report(), schema_mapping=_schema())
    serialized = json.dumps(guarded, ensure_ascii=False)

    assert guarded_again == guarded
    assert guarded["main_conclusions"][0] == {
        "conclusion": "The loss-risk model can support manual review prioritization.",
        "evidence": "Use observed model validation evidence only to compare review trade-offs.",
        "business_meaning": (
            "Select a review threshold based on review capacity and the observed "
            "false-positive / false-negative trade-off."
        ),
    }
    _assert_public_scan_passes(serialized)
    for phrase in (
        "proactive risk management",
        "reactive analysis",
        "intercept",
        "before they ship",
        "before shipping",
        "safeguard against profit erosion",
        "operational safeguard",
        "protect margin",
        "protect profit",
        "mitigate losses operationally",
    ):
        assert phrase not in serialized.lower()
    assert "manual review prioritization" in serialized
    assert "review capacity" in serialized
    assert "false-positive / false-negative trade-off" in serialized
    assert "Review weekly sales trend before adjusting forecasts." in serialized


def test_english_discount_profit_association_overclaim_is_atomically_replaced() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "brief_findings": [
            {
                "finding": "High discount rates are almost perfectly correlated with loss-making transactions.",
                "evidence": "The observed discount bucket has a stated loss rate.",
                "business_meaning": (
                    "Discounting is the primary, clear trigger for converting a sale into a loss and profit protection."
                ),
            }
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    serialized = json.dumps(guarded, ensure_ascii=False)

    assert guarded["brief_findings"][0] == {
        "finding": "The scoped discount-tier evidence should be interpreted as an association with profit outcomes.",
        "evidence": "Use the mapped discount and profit evidence at its stated scope.",
        "business_meaning": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
    }
    _assert_public_scan_passes(serialized)
    for phrase in (
        "almost perfectly correlated",
        "near-perfect",
        "highly correlated",
        "primary, clear trigger",
        "converting a sale into a loss",
        "profit protection",
    ):
        assert phrase not in serialized.lower()
    assert "association with profit outcomes" in serialized
    assert "scoped discount tier" in serialized
    assert "before broader pricing changes" in serialized


def test_english_model_and_discount_replacements_reach_public_rendering() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "brief_findings": [
            {
                "finding": "Discount rates are highly correlated with negative profit.",
                "evidence": "The observed discount bucket has a stated loss rate.",
                "business_meaning": "Discounting is the clear trigger for converting a sale into a loss.",
            }
        ],
        "main_conclusions": [
            {
                "conclusion": "The predictive model enables proactive risk management.",
                "evidence": "Loss-risk model recall was observed.",
                "business_meaning": "Intercept high-risk orders before shipping as an operational safeguard.",
            }
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    final_markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        dataset_profile={"row_count": 999},
        final_synthesis=guarded,
        output_language="en",
    )
    client_payload, _client_trace = build_client_report_payload_with_trace(
        report=_report(),
        schema_mapping=_schema(),
        dataset_profile={"row_count": 999, "column_count": 8},
        analysis_focus=None,
        evidence_pack=None,
        section_priority=None,
        llm_client=None,
        final_synthesis=guarded,
        output_language="en",
    )
    client_html = render_client_report_html(client_payload, output_language="en")

    for text in (
        json.dumps(guarded, ensure_ascii=False),
        final_markdown,
        json.dumps(client_payload, ensure_ascii=False),
        _html_visible_text(client_html),
    ):
        _assert_public_scan_passes(text)


def test_english_public_narrative_rejects_terminal_abbreviation_fragment() -> None:
    guarded = guard_english_public_narrative(
        "Connecting this to the peer scatter plot on Discount vs.",
        chart_context={
            "chart_title": "Discount vs. Profit Relationship",
            "chart_kind": "scatter",
        },
        role="chart_commentary",
        fallback="Review the scoped evidence before broader pricing changes.",
    )

    assert "Discount vs. Review" not in guarded
    assert not guarded.rstrip().lower().endswith("vs.")
    assert "Review the scoped evidence before broader pricing changes." in guarded
    _assert_public_scan_passes(guarded)


def test_english_public_narrative_preserves_internal_abbreviation() -> None:
    guarded = guard_english_public_narrative(
        "The Discount vs. Profit Relationship shows an observed association in the plotted records. "
        "Review the scoped evidence before broader pricing changes.",
        chart_context={"chart_title": "Discount vs. Profit Relationship", "chart_kind": "scatter"},
        role="chart_commentary",
    )

    assert "Discount vs. Profit Relationship" in guarded
    assert "observed association" in guarded
    assert "Discount vs. Review" not in guarded


def test_english_discount_profit_light_overclaim_is_atomically_replaced() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "brief_findings": [
            {
                "finding": "High discount levels are strongly associated with negative profit outcomes.",
                "evidence": "The current discount tier has a stated loss rate.",
                "business_meaning": (
                    "Discounting beyond a certain threshold is almost guaranteed to result in a loss."
                ),
            }
        ],
    }

    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    serialized = json.dumps(guarded, ensure_ascii=False)

    assert guarded["brief_findings"][0] == {
        "finding": "The scoped discount-tier evidence should be interpreted as an association with profit outcomes.",
        "evidence": "Use the mapped discount and profit evidence at its stated scope.",
        "business_meaning": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
    }
    _assert_public_scan_passes(serialized)
    for phrase in (
        "strongly associated",
        "highly associated",
        "strongly correlated",
        "almost guaranteed",
        "guaranteed to result in a loss",
    ):
        assert phrase not in serialized.lower()
    assert "association with profit outcomes" in serialized
    assert "scoped discount tier" in serialized
    assert "before broader pricing changes" in serialized


def test_english_public_rendering_blocks_publication_closure_residuals() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "brief_findings": [
            {
                "finding": "High discount levels are strongly associated with negative profit outcomes.",
                "evidence": "The current discount tier has a stated loss rate.",
                "business_meaning": "Discounting is almost guaranteed to result in a loss.",
            }
        ],
    }
    guarded = guard_english_final_synthesis(raw, report=_report(), schema_mapping=_schema())
    final_markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        dataset_profile={"row_count": 999},
        final_synthesis=guarded,
        output_language="en",
    )
    client_payload, _client_trace = build_client_report_payload_with_trace(
        report=_report(),
        schema_mapping=_schema(),
        dataset_profile={"row_count": 999, "column_count": 8},
        analysis_focus=None,
        evidence_pack=None,
        section_priority=None,
        llm_client=None,
        final_synthesis=guarded,
        output_language="en",
    )
    client_html = render_client_report_html(client_payload, output_language="en")

    for text in (
        json.dumps(guarded, ensure_ascii=False),
        final_markdown,
        json.dumps(client_payload, ensure_ascii=False),
        _html_visible_text(client_html),
        guard_english_public_narrative(
            "Connecting this to the peer scatter plot on Discount vs.",
            chart_context={"chart_title": "Discount vs. Profit Relationship", "chart_kind": "scatter"},
            role="chart_commentary",
            fallback="Review the scoped evidence before broader pricing changes.",
        ),
    ):
        _assert_public_scan_passes(text)
        assert not re.search(r"\b(?:and|or|with|to|of|for|in|at|by|from|the|a|an)\.(?:\s|$)", text, re.I)


def test_english_postrun_header_only_fallback_does_not_emit_dataframe_headers() -> None:
    chart_contexts = [
        {
            "section_id": "sales_trends",
            "section_title": "Sales Trend Analysis",
            "chart_title": "Sales Trend with Rolling Average",
            "table_preview": "order_day Sales Quantity sales_rolling_mean quantity_rolling_mean",
            "code_source": "fig = px.line(...)",
            "cell_index": 18,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=CjkEnglishPostrunReflectionLLM(),
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[18]
    assert trace.status == "fallback_on_error"
    assert "language_mismatch_fallback" in trace.reason
    assert "order_day Sales Quantity sales_rolling_mean quantity_rolling_mean" not in reflection
    assert "shows (" not in reflection
    assert not re.search(r"[\u3400-\u9fff]", reflection)
    assert len([part for part in re.split(r"[.!?]+", reflection) if part.strip()]) >= 2
    assert not re.search(r"\b\d[\d,.]*\b", reflection)


def test_english_public_titles_remove_causal_discount_heading() -> None:
    base_payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "dataset_type": "sales_transaction",
        "sample_size": 100,
        "field_count": 6,
        "business_theme": "Review margin slices.",
        "executive_summary": ["The 30%+ discount tier requires review."],
        "kpi_cards": [],
        "priority_actions": [],
    }
    discount_payload = {
        **base_payload,
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P2",
                    "issue": "High discount tiers are driving margin risk",
                    "action": "Review high-discount margin risk before changing pricing rules.",
                    "display_text": "Review high-discount margin risk before changing pricing rules.",
                    "linked_metric_or_segment": "High-discount tier.",
                }
            ]
        },
    }
    slice_payload = {
        **base_payload,
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P3",
                    "issue": "High discount tiers are driving margin risk",
                    "action": "Review Consumer × Central order structure and discount patterns.",
                    "display_text": "Review Consumer × Central order structure and discount patterns.",
                    "linked_metric_or_segment": "Consumer × Central margin review.",
                }
            ]
        },
    }
    model_payload = {
        **base_payload,
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P3",
                    "issue": "High discount tiers are driving margin risk",
                    "action": "Use the LogisticRegression loss-risk model for review prioritization.",
                    "display_text": "Use the LogisticRegression loss-risk model with recall and score thresholds to support manual review prioritization.",
                    "linked_metric_or_segment": "Recall 66.67%",
                }
            ]
        },
    }

    discount_title = build_client_report_view_model(discount_payload)["action_cards"][0]["title"]
    slice_title = build_client_report_view_model(slice_payload)["action_cards"][0]["title"]
    model_title = build_client_report_view_model(model_payload)["action_cards"][0]["title"]

    assert discount_title == "High-discount margin risk requires review"
    assert "driving" not in discount_title.lower()
    assert "Consumer × Central" in slice_title
    assert model_title == "Loss-risk model supports review prioritization"


def test_chinese_public_narrative_paths_do_not_use_english_guard() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [{"conclusion": "\u6298\u6263\u5ba1\u6279\u6d41\u7a0b\u9700\u8981\u590d\u6838\u3002", "evidence": "\u4e2d\u6587\u8bc1\u636e\u3002"}],
        "recommended_actions": [{"action": "\u7ee7\u7eed\u590d\u6838\u9ad8\u6298\u6263\u8ba2\u5355\u3002", "display_text": "\u7ee7\u7eed\u590d\u6838\u9ad8\u6298\u6263\u8ba2\u5355\u3002"}],
    }
    markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        final_synthesis=raw,
        output_language="zh-CN",
    )
    assert "\u7ee7\u7eed\u590d\u6838\u9ad8\u6298\u6263\u8ba2\u5355" in markdown
    assert "Review approval records if available" not in markdown
    assert "High-discount margin risk requires review" not in markdown
