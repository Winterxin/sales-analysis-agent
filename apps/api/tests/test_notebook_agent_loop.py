from __future__ import annotations

import json
from pathlib import Path

from nbformat.v4 import new_code_cell, new_notebook

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_toolset import save_notebook


class FakeRevisionDecisionLLMClient:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_notebook_revision_decisions(
        self,
        report_context,
        chart_contexts,
        allowed_revisions,
    ):
        return {
            "decisions": [
                {
                    "revision_key": "discount_bucket_loss_rate",
                    "reason": "高折扣负利润明显，需要补充折扣区间亏损率分析。",
                },
                {
                    "revision_key": "invented_revision",
                    "reason": "Should be dropped.",
                },
            ]
        }


class FakeTwoRevisionDecisionLLMClient(FakeRevisionDecisionLLMClient):
    def suggest_notebook_revision_decisions(
        self,
        report_context,
        chart_contexts,
        allowed_revisions,
    ):
        return {
            "decisions": [
                {
                    "revision_key": "discount_bucket_loss_rate",
                    "reason": "Need discount analysis.",
                },
                {
                    "revision_key": "top_product_profit_bridge",
                    "reason": "Need product profit bridge.",
                },
            ]
        }


def _sample_report() -> AnalysisReport:
    return AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=3,
        summary=[],
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                summary_metrics={
                    "distinct_products": 1850,
                    "top_product": "Canon imageCLASS 2200 Advanced Copier",
                },
                tables={
                    "top_products": [
                        {
                            "Product Name": "Canon imageCLASS 2200 Advanced Copier",
                            "Sales": 61599.824,
                            "Quantity": 20,
                        }
                    ]
                },
                findings=["Top 商品贡献明显。"],
            ),
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="客群与区域分析",
                chart_type="stacked_bar",
                summary_metrics={
                    "primary_dimension": "segment",
                    "secondary_dimension": "region",
                },
                tables={
                    "weak_performance_cuts": [
                        {"Segment": "Home Office", "Sales": 429653.1485, "Profit": 60298.6785}
                    ]
                },
                findings=["Home Office 的利润表现最弱。"],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                summary_metrics={
                    "high_discount_order_count": 1393,
                    "negative_profit_order_count": 1871,
                    "negative_profit_rate": 0.1872,
                },
                tables={
                    "discount_buckets": [
                        {"discount_bucket": "20-30%", "avg_profit": -45.6796, "order_count": 227},
                        {"discount_bucket": "30%+", "avg_profit": -107.2099, "order_count": 1166},
                    ]
                },
                findings=["30%+ 折扣区间平均利润最低。"],
            ),
        ],
    )


def _sample_schema_mapping() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Product Name": "product_name",
            "Category": "category",
            "Segment": "segment",
            "Region": "region",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )


def test_notebook_agent_loop_builds_whitelisted_revision_plan_from_llm() -> None:
    from app.services.notebook_agent_loop import build_notebook_revision_plan_with_trace

    plan, trace = build_notebook_revision_plan_with_trace(
        report=_sample_report(),
        schema_mapping=_sample_schema_mapping(),
        chart_contexts=[],
        llm_client=FakeRevisionDecisionLLMClient(),
    )

    assert [decision.revision_key for decision in plan.decisions] == ["discount_bucket_loss_rate"]
    assert plan.decisions[0].reason.startswith("高折扣负利润明显")
    assert trace.stage == "notebook_revision_decision"
    assert trace.status == "llm_applied"


def test_notebook_agent_loop_can_limit_revision_plan_to_one_decision() -> None:
    from app.services.notebook_agent_loop import build_notebook_revision_plan_with_trace

    plan, trace = build_notebook_revision_plan_with_trace(
        report=_sample_report(),
        schema_mapping=_sample_schema_mapping(),
        chart_contexts=[],
        llm_client=FakeTwoRevisionDecisionLLMClient(),
        max_revision_decisions=1,
    )

    assert [decision.revision_key for decision in plan.decisions] == [
        "discount_bucket_loss_rate"
    ]
    assert trace.status == "llm_applied"


def test_notebook_agent_loop_skips_revision_when_quick_profile_disables_it() -> None:
    from app.services.notebook_agent_loop import build_notebook_revision_plan_with_trace

    plan, trace = build_notebook_revision_plan_with_trace(
        report=_sample_report(),
        schema_mapping=_sample_schema_mapping(),
        chart_contexts=[],
        llm_client=FakeTwoRevisionDecisionLLMClient(),
        max_revision_decisions=0,
        llm_profile="quick",
    )

    assert plan.decisions == []
    assert trace.status == "skipped_by_profile"
    assert trace.attempted is False
    assert "quick profile disables notebook revision decision" in trace.reason


def test_notebook_agent_loop_marks_revision_stage_not_applicable_without_safe_templates() -> None:
    from app.services.notebook_agent_loop import build_notebook_revision_plan_with_trace

    plan, trace = build_notebook_revision_plan_with_trace(
        report=AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount"},
            confidence=0.9,
        ),
        chart_contexts=[],
        llm_client=FakeRevisionDecisionLLMClient(),
    )

    assert plan.decisions == []
    assert trace.status == "not_applicable"
    assert "skipped" not in trace.reason.lower()


def test_revision_append_markdown_uses_business_action_structure() -> None:
    from app.schemas.notebook_revision import NotebookRevisionDecision
    from app.services.notebook_agent_loop import _revision_markdown
    from app.services.notebook_agent_loop import _revision_playbook_markdown

    decision = NotebookRevisionDecision(
        revision_key="top_product_profit_bridge",
        reason="头部商品销售额高，但利润贡献需要继续核查。",
    )

    combined = "\n\n".join(
        [_revision_markdown(decision), _revision_playbook_markdown(decision)]
    )

    assert "业务问题" in combined
    assert "关键证据" in combined
    assert "经营含义" in combined
    assert "建议动作" in combined
    assert "下一步需要的数据" in combined
    assert "补图" not in combined


def test_notebook_agent_loop_degrades_when_llm_is_disabled() -> None:
    from app.services.notebook_agent_loop import build_notebook_revision_plan_with_trace

    plan, trace = build_notebook_revision_plan_with_trace(
        report=_sample_report(),
        schema_mapping=_sample_schema_mapping(),
        chart_contexts=[],
        llm_client=None,
    )

    assert plan.decisions == []
    assert isinstance(trace, LLMStageTrace)
    assert trace.status == "disabled"


def test_notebook_agent_loop_appends_revision_trace_and_safe_cells(tmp_path: Path) -> None:
    from app.schemas.notebook_revision import NotebookRevisionDecision, NotebookRevisionPlan
    from app.services.notebook_agent_loop import apply_notebook_revision_plan

    notebook = new_notebook(
        cells=[
            new_code_cell(
                "\n".join(
                    [
                        "import pandas as pd",
                        "import plotly.express as px",
                        "clean_df = pd.DataFrame({",
                        "    'Discount': [0.0, 0.1, 0.2, 0.3, 0.4],",
                        "    'Profit': [10.0, 8.0, -5.0, -8.0, -20.0],",
                        "    'Sales': [100.0, 120.0, 80.0, 90.0, 110.0],",
                        "    'Category': ['A', 'A', 'B', 'B', 'C'],",
                        "    'Product Name': ['P1', 'P2', 'P1', 'P3', 'P4'],",
                        "    'Segment': ['Consumer', 'Consumer', 'Home Office', 'Home Office', 'Corporate'],",
                        "    'Region': ['East', 'West', 'East', 'West', 'Central'],",
                        "})",
                        "clean_df.head()",
                    ]
                )
            )
        ]
    )
    notebook_path = save_notebook(notebook, tmp_path / "analysis.ipynb")

    revision_plan = NotebookRevisionPlan(
        decisions=[
            NotebookRevisionDecision(
                revision_key="discount_bucket_loss_rate",
                reason="高折扣负利润明显，需要补充折扣区间亏损率分析。",
            )
        ]
    )

    apply_notebook_revision_plan(
        notebook_path=notebook_path,
        output_path=notebook_path,
        revision_plan=revision_plan,
        report=_sample_report(),
        schema_mapping=_sample_schema_mapping(),
    )

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    code_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    ]

    combined_markdown = "\n\n".join(markdown_cells)
    combined_code = "\n\n".join(code_cells)

    assert "## Agent 追加分析记录" in combined_markdown
    assert "## 追加分析：折扣区间亏损率" in combined_markdown
    assert "高折扣负利润明显" in combined_markdown
    assert "discount_loss_rate" in combined_code
    assert "loss_rate" in combined_code
