from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient


def test_full_analysis_flow_creates_report_and_notebook(
    client: TestClient, sample_csv_path
) -> None:
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )

    assert upload_response.status_code == 200
    upload_body = upload_response.json()
    assert upload_body["status"] == "uploaded"
    assert "Order Date" in upload_body["ingestion"]["columns"]
    assert upload_body["schema_mapping"]["dataset_type"] == "sales_transaction"
    assert upload_body["analysis_plan"]["analysis_plan"][0] == "data_quality_check"
    assert upload_body["llm_trace"]["schema_mapping"]["status"] == "disabled"

    run_response = client.post(f"/api/v1/analysis/tasks/{task_id}/run")
    assert run_response.status_code == 202
    run_body = run_response.json()
    assert run_body["status"] == "completed"
    assert run_body["output_language"] == "en"
    assert run_body["report"]["task_id"] == task_id
    assert len(run_body["report"]["modules"]) >= 4
    assert run_body["business_review"]["filename"] == "business_review.md"
    assert run_body["notebook"]["filename"] == "sales_orders.ipynb"
    assert run_body["llm_trace"]["notebook_outline"]["status"] == "disabled"
    assert run_body["llm_trace"]["notebook_narrative"]["status"] == "disabled"
    assert run_body["llm_trace"]["notebook_content"]["status"] == "disabled"
    assert run_body["llm_trace"]["report_summary"]["status"] == "disabled"
    assert run_body["llm_trace"]["notebook_revision_decision"]["status"] == "disabled"
    assert run_body["llm_trace"]["postrun_chart_reflection"]["status"] == "disabled"

    task_response = client.get(f"/api/v1/analysis/tasks/{task_id}")
    assert task_response.status_code == 200
    task_body = task_response.json()
    assert task_body["status"] == "completed"
    assert task_body["artifact_manifest"]["files"]["analysis_notebook"].endswith("sales_orders.ipynb")
    assert task_body["artifact_manifest"]["files"]["analysis_notebook_canonical"].endswith(
        "analysis.executed.ipynb"
    )
    assert task_body["artifact_manifest"]["files"]["analysis_source_notebook"].endswith(
        "analysis.ipynb"
    )
    assert task_body["artifact_manifest"]["files"]["business_review_md"].endswith(
        "business_review.md"
    )
    files = task_body["artifact_manifest"]["files"]
    assert "deliverables_dir" not in files
    assert all(not key.startswith("deliverable_") for key in files)
    assert not (Path(files["analysis_notebook"]).parent / "deliverables").exists()
    assert files["client_report_html"].endswith("client_report.html")
    assert files["client_report_json"].endswith("client_report.json")
    assert Path(files["client_report_html"]).exists()
    assert Path(files["client_report_json"]).exists()
    assert task_body["artifact_manifest"]["files"]["notebook_outline_json"].endswith(
        "notebook_outline.json"
    )
    assert task_body["artifact_manifest"]["files"]["notebook_narrative_json"].endswith(
        "notebook_narrative.json"
    )
    assert task_body["artifact_manifest"]["files"]["notebook_content_json"].endswith(
        "notebook_content.json"
    )
    assert task_body["artifact_manifest"]["files"]["llm_trace_json"].endswith("llm_trace.json")
    assert task_body["artifact_manifest"]["files"]["llm_evidence_pack_json"].endswith(
        "llm_evidence_pack.json"
    )
    assert task_body["artifact_manifest"]["files"]["chart_intent_plan_json"].endswith(
        "chart_intent_plan.json"
    )
    assert task_body["artifact_manifest"]["files"]["chart_selection_plan_pre_intent_json"].endswith(
        "chart_selection_plan_pre_intent.json"
    )
    assert Path(task_body["artifact_manifest"]["files"]["llm_evidence_pack_json"]).exists()
    assert Path(task_body["artifact_manifest"]["files"]["chart_intent_plan_json"]).exists()
    assert Path(task_body["artifact_manifest"]["files"]["chart_selection_plan_pre_intent_json"]).exists()
    assert task_body["artifact_manifest"]["chart_intent_planning_mode"] == "shadow"
    assert task_body["artifact_manifest"]["chart_intent_planning_status"] == "skipped"
    assert "summary" not in run_body["llm_trace"]
    llm_trace_payload = json.loads(
        Path(task_body["artifact_manifest"]["files"]["llm_trace_json"]).read_text(encoding="utf-8")
    )
    assert "summary" in llm_trace_payload
    assert llm_trace_payload["summary"]["llm_profile"] == "full"
    assert llm_trace_payload["summary"]["llm_profile_policy"]["profile"] == "full"
    assert llm_trace_payload["summary"]["llm_profile_policy"]["allow_postrun_fallback_reflections"] is False
    assert llm_trace_payload["summary"]["allow_postrun_fallback_reflections"] is False
    assert "remote_call_count" in llm_trace_payload["summary"]
    assert "cache_hit_count" in llm_trace_payload["summary"]
    assert "cache_miss_count" in llm_trace_payload["summary"]
    assert "evidence_pack_enabled" in llm_trace_payload["summary"]
    assert "total_prompt_chars" in llm_trace_payload["summary"]
    assert "total_llm_remote_elapsed_ms" in llm_trace_payload["summary"]
    assert "total_llm_saved_ms_by_cache" in llm_trace_payload["summary"]
    assert task_body["artifact_manifest"]["files"]["revision_decisions_json"].endswith(
        "revision_decisions.json"
    )
    assert task_body["artifact_manifest"]["files"]["notebook_agent_state_json"].endswith(
        "notebook_agent_state.json"
    )
    assert task_body["artifact_manifest"]["files"]["postrun_chart_contexts_json"].endswith(
        "postrun_chart_contexts.json"
    )
    assert task_body["artifact_manifest"]["files"]["postrun_chart_reflections_json"].endswith(
        "postrun_chart_reflections.json"
    )
    assert task_body["artifact_manifest"]["llm_trace"]["schema_mapping"]["status"] == "disabled"
    assert task_body["artifact_manifest"]["llm_profile"] == "full"
    assert task_body["artifact_manifest"]["output_language"] == "en"
    assert task_body["artifact_manifest"]["llm_trace"]["notebook_content"]["status"] == "disabled"
    assert (
        task_body["artifact_manifest"]["llm_trace"]["notebook_revision_decision"]["status"]
        == "disabled"
    )
    assert (
        task_body["artifact_manifest"]["llm_trace"]["postrun_chart_reflection"]["status"]
        == "disabled"
    )
    assert llm_trace_payload["summary"]["output_language"] == "en"

    notebook_text = Path(files["analysis_notebook_canonical"]).read_text(encoding="utf-8")
    business_review_text = Path(files["business_review_md"]).read_text(encoding="utf-8")
    client_report_html = Path(files["client_report_html"]).read_text(encoding="utf-8")
    client_report_payload = json.loads(Path(files["client_report_json"]).read_text(encoding="utf-8"))
    report_payload = json.loads(Path(files["report_json"]).read_text(encoding="utf-8"))
    assert "Sales Data Analysis Notebook" in notebook_text
    assert "Business Review" in business_review_text
    assert "Client-ready Sales Analysis Report" in client_report_html
    assert client_report_payload["title"] == "Client-ready Sales Analysis Report"
    assert report_payload["output_language"] == "en"


def test_run_accepts_quick_llm_profile_and_records_policy(
    client: TestClient, sample_csv_path
) -> None:
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )
    assert upload_response.status_code == 200

    run_response = client.post(f"/api/v1/analysis/tasks/{task_id}/run?llm_profile=quick&output_language=en")

    assert run_response.status_code == 202
    task_body = client.get(f"/api/v1/analysis/tasks/{task_id}").json()
    manifest = task_body["artifact_manifest"]
    trace_payload = json.loads(Path(manifest["files"]["llm_trace_json"]).read_text(encoding="utf-8"))

    assert manifest["llm_profile"] == "quick"
    assert manifest["output_language"] == "en"
    assert manifest["llm_profile_policy"]["profile"] == "quick"
    assert manifest["llm_profile_policy"]["report_summary_enabled"] is True
    assert manifest["llm_profile_policy"]["notebook_narrative_enabled"] is True
    assert manifest["llm_profile_policy"]["final_synthesis_enabled"] is True
    assert manifest["llm_profile_policy"]["client_report_enabled"] is True
    assert manifest["llm_profile_policy"]["modeling_interpretation_enabled"] is True
    assert manifest["llm_profile_policy"]["modeling_opportunity_decision_enabled"] is True
    assert manifest["llm_profile_policy"]["modeling_outcome_interpretation_enabled"] is True
    assert manifest["llm_profile_policy"]["chart_selection_enabled"] is True
    assert manifest["llm_profile_policy"]["chart_intent_planning_enabled"] is False
    assert manifest["llm_profile_policy"]["intent_guided_chart_selection_enabled"] is False
    assert manifest["llm_profile_policy"]["notebook_revision_enabled"] is False
    assert manifest["llm_profile_policy"]["max_revision_decisions"] == 0
    assert manifest["chart_intent_planning_status"] == "skipped"
    assert manifest["chart_selection_plan"]["selection_mode"] in {
        "deterministic",
        "llm_fallback",
        "llm_sanitized",
    }
    assert manifest["chart_selection_plan"]["intent_guided_selection_trace"]["status"] == "skipped"
    assert trace_payload["summary"]["llm_profile"] == "quick"
    assert trace_payload["summary"]["output_language"] == "en"
    assert trace_payload["summary"]["llm_profile_policy"]["allow_postrun_fallback_reflections"] is False
    assert trace_payload["summary"]["allow_postrun_fallback_reflections"] is False


def test_run_accepts_zh_cn_output_language_without_breaking_chinese_artifacts(
    client: TestClient, sample_csv_path
) -> None:
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )
    assert upload_response.status_code == 200

    run_response = client.post(f"/api/v1/analysis/tasks/{task_id}/run?output_language=zh-CN")

    assert run_response.status_code == 202
    assert run_response.json()["output_language"] == "zh-CN"
    task_body = client.get(f"/api/v1/analysis/tasks/{task_id}").json()
    manifest = task_body["artifact_manifest"]
    trace_payload = json.loads(Path(manifest["files"]["llm_trace_json"]).read_text(encoding="utf-8"))
    notebook_text = Path(manifest["files"]["analysis_notebook_canonical"]).read_text(encoding="utf-8")
    business_review_text = Path(manifest["files"]["business_review_md"]).read_text(encoding="utf-8")
    client_report_payload = json.loads(Path(manifest["files"]["client_report_json"]).read_text(encoding="utf-8"))

    assert manifest["output_language"] == "zh-CN"
    assert trace_payload["summary"]["output_language"] == "zh-CN"
    assert "销售数据分析 Notebook" in notebook_text
    assert "目录" in notebook_text or "分析目标" in notebook_text
    assert "销售经营复盘报告" in business_review_text
    assert client_report_payload["title"] == "客户经营分析简报"


def test_zh_client_report_keeps_profit_metrics_when_discount_field_is_missing(
    client: TestClient, tmp_path: Path
) -> None:
    csv_path = tmp_path / "profit_no_discount_sales.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Order Date,Product Name,Category,Sales,Quantity,Profit",
                "2024-01-01,A,Tech,1000,2,120",
                "2024-01-02,B,Tech,800,1,-40",
                "2024-01-03,C,Office,500,3,80",
                "2024-01-04,D,Office,300,4,-20",
                "2024-01-05,E,Furniture,700,1,60",
                "2024-01-06,F,Furniture,200,2,-50",
            ]
        ),
        encoding="utf-8",
    )
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    with csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("profit_no_discount_sales.csv", csv_file, "text/csv")},
        )
    assert upload_response.status_code == 200
    assert "Discount" not in upload_response.json()["schema_mapping"]["field_mapping"]

    run_response = client.post(f"/api/v1/analysis/tasks/{task_id}/run?output_language=zh-CN")
    assert run_response.status_code == 202
    manifest = client.get(f"/api/v1/analysis/tasks/{task_id}").json()["artifact_manifest"]
    client_report_payload = json.loads(Path(manifest["files"]["client_report_json"]).read_text(encoding="utf-8"))
    report_payload = json.loads(Path(manifest["files"]["report_json"]).read_text(encoding="utf-8"))

    module_ids = {module["module_id"] for module in report_payload["modules"]}
    labels = {card["label"]: card["value"] for card in client_report_payload["kpi_cards"]}

    assert "discount_profit_analysis" in module_ids
    assert labels["总利润"] != "暂无"
    assert labels["负利润记录占比"] != "暂无"
    assert "最高风险折扣阈值" not in labels
    assert "what-if 估算利润改善" not in labels
    assert client_report_payload["discount_what_if"] == {}


def test_quick_english_artifacts_do_not_expose_common_chinese_headings(
    client: TestClient, sample_csv_path
) -> None:
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )
    assert upload_response.status_code == 200

    run_response = client.post(
        f"/api/v1/analysis/tasks/{task_id}/run?llm_profile=quick&output_language=en"
    )

    assert run_response.status_code == 202
    manifest = client.get(f"/api/v1/analysis/tasks/{task_id}").json()["artifact_manifest"]
    files = manifest["files"]
    source_notebook_text = Path(files["analysis_source_notebook"]).read_text(encoding="utf-8")
    executed_notebook_text = Path(files["analysis_notebook_canonical"]).read_text(encoding="utf-8")
    notebook_text = source_notebook_text + "\n" + executed_notebook_text
    business_review_text = Path(files["business_review_md"]).read_text(encoding="utf-8")
    client_report_html = Path(files["client_report_html"]).read_text(encoding="utf-8")
    client_report_payload = json.loads(Path(files["client_report_json"]).read_text(encoding="utf-8"))
    client_report_text = json.dumps(client_report_payload, ensure_ascii=False)
    client_report_combined = "\n".join([client_report_text, client_report_html])
    client_report_headings = re.findall(r"<h[13]>(.*?)</h[13]>", client_report_html)
    combined = "\n".join([notebook_text, business_review_text, client_report_combined])

    for chinese_heading in (
        "数据概览",
        "数据质量",
        "经营复盘",
        "关键发现",
        "建议",
        "客户报告",
        "销售趋势",
        "结论",
        "本节结论",
        "图表解读",
        "总销售额",
        "样本量",
        "字段数",
    ):
        assert chinese_heading not in combined
    for forbidden_notebook_text in (
        "销售数据分析 Notebook",
        "数据概览",
        "数据质量",
        "销售趋势",
        "商品与类目",
        "客群与区域",
        "折扣与利润",
        "图表解读",
        "本节结论",
        "建模目标",
        "模型表现",
        "混淆矩阵分析",
        "风险信号解释",
        "建模综合Conclusion",
        "行动建议",
        "使用边界",
        "后续应",
        "建议优先",
        "高风险样例",
        "漏判亏损样例",
        "亏损率",
        "利润率",
        "平均Profit",
        "Discount区间",
        "设为Discount审批红线",
        "不代表真实需求",
        "销量或客户行为变化",
        "Section Takeaway：### Section Takeaway",
        "mapped fields and module outputs define the current review scope",
        "This section identifies the records and business objects that need follow-up review",
        "This section provides supporting evidence for",
        "noting any unc。",
        "revenue dy。",
        "guiding the focus of dee。",
        "数据不足",
        "缺少",
        "跳过",
        "阈值取舍",
        "默认阈值下的混淆矩阵",
        "亏损风险预测信号组重要性",
        "预测类别",
        "真实类别",
        "指标值",
        "复核量占比",
        "精确率",
        "召回率",
        "特征重要性",
        "可复现建模代码",
        "误差分析",
        "综合结论",
        "销售结构",
        "销售热力图",
        "销售额贡献",
        "销售额占比",
        "月度销售趋势",
        "平均订单金额",
        "客户数",
        "订单数",
        "产品线",
        "交易规模",
        "销量",
        "贡献",
        "图表选择依据",
        "验证当前 section 的关键业务判断",
        "折扣收紧",
        "高风险折扣区间",
        "策略层跳过",
        "订单金额分布",
        "单价与数量关系",
        "年月销售热力图",
        "滚动均线",
        "平均日销售额",
        "长尾分布",
        "截尾展示",
        "低频高额客户",
        "客群销售与利润对比",
        "客群/区域销售额对比",
        "客群 x Discount Tier Loss Rate Heatmap",
        "区域 x Discount Tier Loss Rate Heatmap",
        "客群 x 折扣区间亏损率热力图",
        "区域 x 折扣区间亏损率热力图",
    ):
        assert forbidden_notebook_text not in notebook_text
    rationale_blocks = re.findall(
        r"### Chart Selection Rationale.*?(?=\\n### |\\n# |$)",
        notebook_text,
        flags=re.DOTALL,
    )
    for rationale_block in rationale_blocks:
        for bullet in rationale_block.splitlines():
            if bullet.lstrip().startswith("- "):
                assert not bullet.rstrip().endswith("...")
                assert not bullet.rstrip().endswith("…")
    for expected_notebook_text in (
        "Sales Data Analysis Notebook",
        "Product and Category Analysis",
        "Segment and Region Analysis",
        "Modeling Analysis",
        "Recommended Actions",
        "Usage Limits",
    ):
        assert expected_notebook_text in notebook_text
    for forbidden_client_report_text in (
        "The primary…",
        "A strong…",
        "Overall, 18.65% of…",
        "最终Conclusion",
        "最终结论",
        "基线改善",
        "相对基线改善",
        "经营观察",
        "核心摘要",
        "高风险区间",
        "亏损率",
        "核心指标",
        "亏损复核召回",
        "模型召回率仪表盘",
        "可以辅助复核排序",
        "误报亏损数",
        "漏判亏损数",
        "受影响",
        "Profit率",
        "平均Profit",
        "Discount区间",
        "折扣区间",
        "影响记录",
        "主要经营观察",
        "核心信号",
        "趋势波动",
        "集中度观察",
        "万",
        "亿",
        "catastrophic",
        "confirms",
        "systemic issue",
        "proves",
        "causal",
    ):
        assert forbidden_client_report_text not in client_report_combined
    assert all(not heading.endswith("…") for heading in client_report_headings)
    assert all(
        not heading.startswith(("The primary", "A strong", "Overall,"))
        for heading in client_report_headings
    )
    for english_marker in (
        "Sales Data Analysis Notebook",
        "Business Review",
        "Client-ready Sales Analysis Report",
        "Key Findings",
        "Recommendations",
    ):
        assert english_marker in combined
    for expected_client_report_text in (
        "Client-ready Sales Analysis Report",
        "Dataset Overview",
        "Key Risk Spotlight",
        "Key Findings",
        "Recommended Actions",
        "Usage Limits",
    ):
        assert expected_client_report_text in client_report_combined


def test_demo_safe_run_skips_low_priority_llm_stages(
    client: TestClient, sample_csv_path, monkeypatch
) -> None:
    monkeypatch.setenv("SALES_AGENT_DEMO_SAFE", "true")
    monkeypatch.setenv("SALES_AGENT_DEMO_SAFE_RUN_BUDGET_SECONDS", "85")
    monkeypatch.setenv("SALES_AGENT_DEMO_SAFE_MIN_STAGE_BUDGET_SECONDS", "8")
    monkeypatch.setenv("SALES_AGENT_MAX_POSTRUN_REFLECTIONS", "1")

    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )
    assert upload_response.status_code == 200

    run_response = client.post(f"/api/v1/analysis/tasks/{task_id}/run")

    assert run_response.status_code == 202
    trace = run_response.json()["llm_trace"]
    assert trace["notebook_outline"]["status"] == "disabled"
    assert trace["notebook_content"]["status"] == "disabled"
    assert trace["postrun_chart_reflection"]["status"] == "disabled"
    assert trace["notebook_narrative"]["status"] == "skipped_by_demo_safe"
    assert trace["report_summary"]["status"] == "skipped_by_demo_safe"
    assert trace["notebook_revision_decision"]["status"] == "disabled"
    assert trace["notebook_narrative"]["fallback_type"] == "demo_safe"
    assert trace["report_summary"]["budget_remaining_ms"] is not None
