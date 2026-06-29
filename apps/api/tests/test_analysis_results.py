from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.services.artifact_store import ArtifactStore
from app.services.analysis_results import build_analysis_results_payload


def test_analysis_results_prefers_notebook_markdown_conclusions(tmp_path: Path) -> None:
    task_id = "tatest-api-key"
    report_path = tmp_path / "report.json"
    notebook_path = tmp_path / "analysis.executed.ipynb"
    report_path.write_text(
        """
        {
          "task_id": "tatest-api-key",
          "dataset_type": "sales_transaction",
          "module_count": 2,
          "summary": ["发现 0 条时间缺失记录。"],
          "modules": [
            {
              "module_id": "sales_trend_analysis",
              "title": "销售趋势分析",
              "chart_type": "line",
              "summary_metrics": {"total_sales_amount": 2326534.35, "year_count": 4},
              "tables": {"daily_totals": [{"period": "2026-01-01", "Sales": 10}]},
              "chart_payload": {"x": ["2026-01-01"], "y": [10], "series_name": "Sales"},
              "findings": ["共覆盖 1242 个日度时间粒度。"],
              "warnings": []
            },
            {
              "module_id": "loss_risk_modeling",
              "title": "亏损风险建模",
              "chart_type": "modeling",
              "summary_metrics": {"best_model": "LogisticRegression", "best_recall": 0.96},
              "tables": {},
              "chart_payload": {},
              "findings": ["最佳模型为 LogisticRegression。"],
              "warnings": []
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    notebook_path.write_text(
        """
        {
          "cells": [
            {"cell_type": "markdown", "source": ["## 销售趋势分析\\n", "本节分析销售额。\\n"]},
            {"cell_type": "markdown", "source": ["本节结论：累计销售额约2,326,534.35，覆盖48个月。月度波动率高达51.89%，近期增长率为-28.09%。后续应拆分峰值月份的商品、渠道和大单。"]},
            {"cell_type": "markdown", "source": ["## 建模分析：亏损风险识别\\n", "### 建模分析小结\\n", "本次建模旨在预警利润为负的亏损订单，召回率达到96%，可用于人工复核优先级排序。"]},
            {"cell_type": "markdown", "source": ["## Action Plan\\n", "| priority | issue | action |\\n", "| --- | --- | --- |\\n", "| P1 | 折扣和利润质量需要先收敛风险口径 | 设置审批阈值 |"]},
            {"cell_type": "markdown", "source": ["## 结论与行动建议\\n", "### 核心判断\\n", "- 利润质量是首要问题。\\n", "- 关键风险切片：折扣区间30%+、Furniture类目。\\n", "### 行动方向\\n", "当前分析已明确指向折扣管理。"]}
          ]
        }
        """,
        encoding="utf-8",
    )

    payload = build_analysis_results_payload(
        task_id,
        {
            "uploaded_filename": "samplesuperstore.csv",
            "files": {
                "report_json": str(report_path),
                "analysis_notebook": str(notebook_path),
            },
        },
    )

    assert payload["summary"] == [
        "利润质量是首要问题。",
        "关键风险切片：折扣区间30%+、Furniture类目。",
    ]
    trend = payload["sections"][0]
    assert trend["section_id"] == "sales_trends"
    assert "累计销售额约2,326,534.35" in trend["takeaway"]
    assert trend["is_long_text"] is False
    modeling = payload["modeling"]
    assert modeling["narrative"].startswith("本次建模旨在预警利润为负")
    assert payload["action_plan"]["rows"][0]["priority"] == "P1"


def test_analysis_results_endpoint_builds_customer_reading_payload(
    client: TestClient,
) -> None:
    task_id = client.post("/api/v1/analysis/tasks").json()["task_id"]
    store = ArtifactStore()
    report_path = store.save_json(
        task_id,
        "report.json",
        {
            "task_id": task_id,
            "dataset_type": "sales_transaction",
            "module_count": 2,
            "summary": [
                "累计销售额约 2,326,534，覆盖 48 个月。",
                "最佳模型为 LogisticRegression，Recall=0.96，F1=0.8187，ROC AUC=0.9867。",
            ],
            "modules": [
                {
                    "module_id": "sales_trend_analysis",
                    "title": "销售趋势分析",
                    "chart_type": "line",
                    "summary_metrics": {
                        "total_sales_amount": 2326534.35,
                        "monthly_grain_count": 48,
                        "recent_growth_rate": -0.2809,
                    },
                    "tables": {
                        "monthly_totals": [
                            {"period": "2026-01", "Sales": 100.0},
                            {"period": "2026-02", "Sales": 180.0},
                        ]
                    },
                    "chart_payload": {
                        "x": ["2026-01", "2026-02"],
                        "y": [100.0, 180.0],
                        "series_name": "Sales",
                    },
                    "findings": [
                        "月度波动率较高，需要拆分峰值月份来源。",
                        "这是一段比较长的图表分析，应该默认折叠在完整分析里，避免结果页面被大段 LLM 文字撑开。"
                        * 8,
                    ],
                    "warnings": [],
                },
                {
                    "module_id": "loss_risk_modeling",
                    "title": "亏损风险建模",
                    "chart_type": "modeling",
                    "summary_metrics": {
                        "best_model": "LogisticRegression",
                        "best_recall": 0.96,
                        "best_f1": 0.8187,
                        "best_roc_auc": 0.9867,
                    },
                    "tables": {
                        "confusion_matrix": [
                            {"actual": "not_loss", "predicted": "not_loss", "count": 700},
                            {"actual": "not_loss", "predicted": "loss", "count": 183},
                            {"actual": "loss", "predicted": "not_loss", "count": 19},
                            {"actual": "loss", "predicted": "loss", "count": 210},
                        ]
                    },
                    "chart_payload": {},
                    "findings": ["模型只用于人工复核优先级排序，不代表因果关系。"],
                    "warnings": [],
                },
            ],
        },
    )
    profile_path = store.save_json(
        task_id,
        "dataset_profile.json",
        {"row_count": 10194, "column_count": 21, "date_span_days": 1460},
    )
    narrative_path = store.save_json(
        task_id,
        "notebook_narrative.json",
        {
            "sections": [
                {
                    "section_id": "sales_trends",
                    "business_takeaway": "时间趋势用于判断促销时点和阶段性经营表现。",
                }
            ]
        },
    )
    manifest = store.load_manifest(task_id)
    manifest["uploaded_filename"] = "samplesuperstore.csv"
    manifest["files"] = {
        **manifest.get("files", {}),
        "report_json": str(report_path),
        "dataset_profile_json": str(profile_path),
        "notebook_narrative_json": str(narrative_path),
        "analysis_notebook": str(Path("samplesuperstore.ipynb")),
        "client_report_html": str(Path("client_report.html")),
        "client_report_json": str(Path("client_report.json")),
    }
    store.save_manifest(task_id, manifest)

    response = client.get(f"/api/v1/analysis/tasks/{task_id}/results")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["filename"] == "samplesuperstore.csv"
    assert body["dataset"]["row_count"] == 10194
    assert body["downloads"]["notebook"].endswith("/artifacts/notebook")
    assert body["kpis"][0]["label"] == "累计销售额"
    assert body["sections"][0]["section_id"] == "sales_trends"
    assert body["sections"][0]["chart"]["type"] == "line"
    assert body["sections"][0]["chart"]["payload"]["x"] == ["2026-01", "2026-02"]
    assert body["sections"][0]["is_long_text"] is True
    assert body["sections"][0]["takeaway"] == "时间趋势用于判断促销时点和阶段性经营表现。"
    assert body["modeling"]["best_model"] == "LogisticRegression"
    assert body["modeling"]["metrics"]["Recall"] == "96%"


def test_analysis_results_endpoint_returns_404_without_report(
    client: TestClient,
) -> None:
    task_id = client.post("/api/v1/analysis/tasks").json()["task_id"]

    response = client.get(f"/api/v1/analysis/tasks/{task_id}/results")

    assert response.status_code == 404
    assert response.json()["detail"] == "Analysis results are not available"
