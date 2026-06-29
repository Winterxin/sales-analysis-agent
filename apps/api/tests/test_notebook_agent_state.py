from __future__ import annotations

from app.schemas.notebook_revision import NotebookRevisionDecision, NotebookRevisionPlan
from app.services.notebook_agent_loop import build_notebook_agent_loop_state


def test_notebook_agent_loop_builds_visible_agent_state() -> None:
    state = build_notebook_agent_loop_state(
        task_id="task-1",
        dataset_type="sales_transaction",
        outline_section_ids=["sales_trends", "discount_and_profit", "conclusions"],
        initial_chart_contexts=[
            {"section_id": "sales_trends", "chart_title": "按月销售额趋势"},
            {"section_id": "discount_and_profit", "chart_title": "各折扣区间利润分布"},
        ],
        final_chart_contexts=[
            {"section_id": "sales_trends", "chart_title": "按月销售额趋势"},
            {"section_id": "discount_and_profit", "chart_title": "各折扣区间利润分布"},
            {"section_id": "discount_and_profit", "chart_title": "各折扣区间亏损率与平均利润"},
        ],
        revision_plan=NotebookRevisionPlan(
            decisions=[
                NotebookRevisionDecision(
                    revision_key="discount_bucket_loss_rate",
                    reason="高折扣负利润明显，需要补充折扣区间亏损率分析。",
                )
            ]
        ),
        llm_trace_payload={
            "notebook_outline": {"status": "disabled"},
            "notebook_content": {"status": "disabled"},
            "notebook_revision_decision": {"status": "llm_applied"},
            "postrun_chart_reflection": {"status": "disabled"},
        },
    )

    assert state.task_id == "task-1"
    assert state.revision_count == 1
    assert [stage.stage for stage in state.stages] == [
        "plan",
        "compose",
        "execute",
        "inspect",
        "revise",
        "finalize",
    ]
    assert state.stages[3].metadata["chart_count"] == 2
    assert state.stages[4].metadata["selected_revision_keys"] == ["discount_bucket_loss_rate"]
    assert state.stages[5].metadata["final_chart_count"] == 3
