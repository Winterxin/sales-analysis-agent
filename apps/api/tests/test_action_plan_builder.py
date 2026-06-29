from __future__ import annotations

from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook.action_plan_builder import action_plan_rows


def _mapping(fields: dict[str, str]) -> SchemaMapping:
    return SchemaMapping(dataset_type="sales_transaction", field_mapping=fields, confidence=1.0)


def _row_by_issue(rows: list[dict[str, str]], term: str) -> dict[str, str]:
    return next(row for row in rows if term in row["issue"] or term in row["action"])


def test_action_plan_evidence_matches_issue_type_for_order_sales_fields() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        modules=[
            ModuleReport(
                module_id="country_market_analysis",
                title="国家市场分析",
                chart_type="bar",
                summary_metrics={"country_count": 7, "top_country_sales_share": 0.34},
                tables={
                    "country_market": [
                        {"country": "市场 A", "sales_amount": 120000, "order_count": 41, "avg_order_value": 2926.8}
                    ]
                },
            ),
            ModuleReport(
                module_id="order_structure_analysis",
                title="订单结构分析",
                chart_type="bar",
                summary_metrics={
                    "productline_count": 5,
                    "distinct_deal_size_count": 3,
                    "distinct_order_status_count": 4,
                    "line_per_order_avg": 2.4,
                },
                tables={
                    "productline_sales": [
                        {"productline": "产品线 A", "sales_amount": 88000, "quantity": 320, "order_count": 19}
                    ],
                    "deal_size_sales": [
                        {"deal_size": "交易规模 A", "sales_amount": 53000, "order_count": 16, "avg_order_value": 3312.5}
                    ],
                    "order_status_sales": [
                        {"order_status": "状态 A", "sales_amount": 91000, "order_count": 29}
                    ],
                },
            ),
        ],
    )

    rows = action_plan_rows(
        report=report,
        schema_mapping=_mapping(
            {
                "ORDERNUMBER": "order_id",
                "PRODUCTLINE": "productline",
                "DEALSIZE": "deal_size",
                "STATUS": "order_status",
                "COUNTRY": "country",
                "QUANTITYORDERED": "quantity",
                "SALES": "sales_amount",
            }
        ),
        dataset_profile={
            "country_count": 7,
            "top_country_sales_share": 0.34,
            "productline_count": 5,
            "distinct_deal_size_count": 3,
            "distinct_order_status_count": 4,
            "line_per_order_avg": 2.4,
        },
        analysis_focus={
            "selected_focuses": [
                "country_market_focus",
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "customer_order_structure_focus",
            ]
        },
    )

    productline = _row_by_issue(rows, "产品线")
    deal_size = _row_by_issue(rows, "DEALSIZE")
    status = _row_by_issue(rows, "STATUS")
    country = _row_by_issue(rows, "国家")

    assert "产品线" in productline["evidence"] or "PRODUCTLINE" in productline["evidence"]
    assert "交易规模" in deal_size["evidence"] or "DEALSIZE" in deal_size["evidence"]
    assert "订单状态" in status["evidence"] or "STATUS" in status["evidence"]
    assert "国家" in country["evidence"]
    assert len({row["evidence"] for row in rows}) == len(rows)


def test_action_plan_uses_specific_business_evidence_before_profile_counts() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        modules=[
            ModuleReport(
                module_id="country_market_analysis",
                title="国家市场分析",
                chart_type="bar",
                tables={
                    "country_market_totals": [
                        {"country": "市场 A", "sales_amount": 120000, "sales_share": 0.42, "order_count": 41}
                    ]
                },
            ),
            ModuleReport(
                module_id="order_structure_analysis",
                title="订单结构分析",
                chart_type="bar",
                tables={
                    "productline_sales": [
                        {"productline": "产品线 A", "sales_amount": 88000, "sales_share": 0.37, "quantity": 320}
                    ],
                    "deal_size_sales": [
                        {
                            "deal_size": "交易规模 A",
                            "sales_amount": 53000,
                            "order_count": 16,
                            "avg_order_value": 3312.5,
                        }
                    ],
                    "order_status_breakdown": [
                        {"order_status": "状态 A", "sales_amount": 91000, "sales_share": 0.64, "order_count": 29}
                    ],
                },
            ),
        ],
    )
    weak_evidence_pack = {
        "focus_evidence": {
            "country_market_focus": [
                {"evidence_id": "country_count", "business_meaning": "国家数为 19，可作为当前分析主线的证据。"}
            ],
            "productline_performance_focus": [
                {"evidence_id": "productline_count", "business_meaning": "PRODUCTLINE 数为 7，可作为当前分析主线的证据。"}
            ],
            "deal_size_focus": [
                {"evidence_id": "distinct_deal_size_count", "business_meaning": "DEALSIZE 类型数为 3，可作为当前分析主线的证据。"}
            ],
            "order_status_focus": [
                {"evidence_id": "distinct_order_status_count", "business_meaning": "STATUS 类型数为 6，可作为当前分析主线的证据。"}
            ],
        }
    }

    rows = action_plan_rows(
        report=report,
        schema_mapping=_mapping(
            {
                "ORDERNUMBER": "order_id",
                "PRODUCTLINE": "productline",
                "DEALSIZE": "deal_size",
                "STATUS": "order_status",
                "COUNTRY": "country",
                "SALES": "sales_amount",
            }
        ),
        dataset_profile={"country_count": 19, "productline_count": 7, "distinct_deal_size_count": 3, "distinct_order_status_count": 6},
        analysis_focus={
            "selected_focuses": [
                "country_market_focus",
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
            ]
        },
        evidence_pack=weak_evidence_pack,
    )

    country = _row_by_issue(rows, "国家")
    productline = _row_by_issue(rows, "产品线")
    deal_size = _row_by_issue(rows, "DEALSIZE")
    status = _row_by_issue(rows, "STATUS")

    for row in (country, productline, deal_size, status):
        assert "可作为当前分析主线的证据" not in row["evidence"]
    assert "市场 A" in country["evidence"] and "120,000" in country["evidence"]
    assert "产品线 A" in productline["evidence"] and "88,000" in productline["evidence"]
    assert "交易规模 A" in deal_size["evidence"] and "53,000" in deal_size["evidence"]
    assert "状态 A" in status["evidence"] and "91,000" in status["evidence"]
    assert len({row["evidence"] for row in rows}) == len(rows)


def test_action_plan_discount_evidence_does_not_leak_into_product_rows() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣利润分析",
                chart_type="bar",
                summary_metrics={"negative_profit_rate": 0.18, "profit_margin_spread": 0.52},
                tables={
                    "discount_profit_risk_buckets": [
                        {"discount_bucket": "折扣区间 A", "avg_profit": -12.5, "negative_profit_rate": 0.61}
                    ]
                },
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品类目分析",
                chart_type="bar",
                tables={
                    "category_profit_quality": [
                        {"category": "类目 A", "sales_amount": 92000, "profit": 1800, "profit_margin": 0.019}
                    ],
                    "high_sales_low_profit_products": [
                        {"product_name": "商品 A", "sales_amount": 18000, "profit": -240, "profit_margin": -0.013}
                    ],
                },
            ),
        ],
    )

    rows = action_plan_rows(
        report=report,
        schema_mapping=_mapping(
            {
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Category": "category",
                "Product Name": "product_name",
            }
        ),
        dataset_profile={"has_discount": True, "has_profit": True, "negative_profit_rate": 0.18},
        analysis_focus={
            "selected_focuses": [
                "discount_erosion_focus",
                "profit_quality_focus",
                "product_concentration_focus",
            ]
        },
    )

    discount = _row_by_issue(rows, "折扣")
    category = _row_by_issue(rows, "类目")
    product = _row_by_issue(rows, "商品")

    assert "折扣" in discount["evidence"] or "亏损" in discount["evidence"]
    assert "类目" in category["evidence"]
    assert "商品" in product["evidence"]
    assert category["evidence"] != discount["evidence"]
    assert product["evidence"] != discount["evidence"]


def test_action_plan_prefers_concrete_business_objects_for_product_category_and_slice() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=3,
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣利润分析",
                chart_type="bar",
                tables={
                    "discount_profit_risk_buckets": [
                        {
                            "discount_bucket": "30%+",
                            "negative_profit_rate": 0.98,
                            "avg_profit": -105.91,
                            "profit_margin": -0.48,
                            "sales_amount": 190000,
                        }
                    ]
                },
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品类目分析",
                chart_type="bar",
                tables={
                    "category_profit_quality": [
                        {
                            "Category": "Furniture",
                            "Sales": 300000,
                            "Profit": -12000,
                            "profit_margin": -0.04,
                            "negative_profit_rate": 0.42,
                        }
                    ],
                    "high_sales_low_profit_products": [
                        {
                            "Product Name": "Cubify CubeX 3D Printer",
                            "Sales": 62000,
                            "Profit": -5200,
                            "profit_margin": -0.084,
                            "negative_profit_rate": 0.55,
                        }
                    ],
                },
            ),
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="组合切片分析",
                chart_type="bar",
                tables={
                    "weak_segment_region": [
                        {
                            "Segment": "Consumer",
                            "Region": "Central",
                            "Sales": 150000,
                            "Profit": -5100,
                            "profit_margin": -0.034,
                            "negative_profit_rate": 0.35,
                        }
                    ]
                },
            ),
        ],
    )

    rows = action_plan_rows(
        report=report,
        schema_mapping=_mapping(
            {
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Category": "category",
                "Product Name": "product_name",
                "Segment": "segment",
                "Region": "region",
            }
        ),
        dataset_profile={"has_discount": True, "has_profit": True, "negative_profit_rate": 0.32},
        analysis_focus={
            "selected_focuses": [
                "discount_erosion_focus",
                "profit_quality_focus",
                "product_concentration_focus",
                "segment_region_focus",
            ]
        },
    )

    category = _row_by_issue(rows, "类目")
    product = _row_by_issue(rows, "商品")
    slice_row = _row_by_issue(rows, "组合切片")

    assert "Furniture" in category["evidence"]
    assert "Cubify CubeX 3D Printer" in product["evidence"]
    assert "Consumer" in slice_row["evidence"] or "Central" in slice_row["evidence"]
    assert "30%+" not in category["evidence"]
    assert "30%+" not in product["evidence"]
    assert "类目维度" not in category["evidence"]
    assert "商品维度" not in product["evidence"]
    assert "组合切片维度" not in slice_row["evidence"]


def test_discount_risk_action_uses_high_risk_bucket_instead_of_first_low_risk_row() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣利润分析",
                chart_type="bar",
                summary_metrics={"negative_profit_rate": 0.32, "profit_margin_spread": 0.71},
                tables={
                    "discount_buckets": [
                        {
                            "discount_bucket": "0%",
                            "negative_profit_rate": 0.0,
                            "avg_profit": 66.34,
                            "profit_margin": 0.18,
                            "sales_amount": 120000,
                        },
                        {
                            "discount_bucket": "30%+",
                            "negative_profit_rate": 0.75,
                            "avg_profit": -46.25,
                            "profit_margin": -0.08,
                            "sales_amount": 93000,
                        },
                    ]
                },
            )
        ],
    )

    rows = action_plan_rows(
        report=report,
        schema_mapping=_mapping(
            {
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Segment": "segment",
                "Region": "region",
                "Category": "category",
            }
        ),
        dataset_profile={"has_discount": True, "has_profit": True, "negative_profit_rate": 0.32},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
    )

    discount = _row_by_issue(rows, "折扣")

    assert "30%+" in discount["evidence"]
    assert "75.00%" in discount["evidence"]
    assert "0.00%%" not in discount["evidence"]
    assert "折扣利润「0%」" not in discount["evidence"]


def test_slice_action_evidence_uses_combined_segment_region_label() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="客群与区域分析",
                chart_type="bar",
                tables={
                    "weak_performance_cuts": [
                        {
                            "Segment": "Consumer",
                            "Region": "Central",
                            "Sales": 125000.0,
                            "Profit": -1200.0,
                            "profit_margin": -0.0096,
                        }
                    ]
                },
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Segment": "segment",
            "Region": "region",
            "Sales": "sales_amount",
            "Profit": "profit",
        },
        confidence=0.9,
    )

    rows = action_plan_rows(report, schema_mapping)

    slice_row = next(row for row in rows if "组合切片" in row["issue"])
    assert "Consumer / Central" in slice_row["evidence"]
    assert "Consumer" in slice_row["evidence"]
    assert "Central" in slice_row["evidence"]


def test_discount_erosion_action_uses_high_risk_threshold_evidence() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                tables={
                    "discount_threshold_candidates": [
                        {
                            "discount_bucket": "0%",
                            "row_count": 4,
                            "sales_amount": 1600.0,
                            "avg_profit": 80.0,
                            "profit_margin": 0.2,
                            "negative_profit_rate": 0.0,
                            "risk_level": "low",
                        },
                        {
                            "discount_bucket": "30%+",
                            "row_count": 5,
                            "sales_amount": 2200.0,
                            "avg_profit": -46.25,
                            "profit_margin": -0.1051,
                            "negative_profit_rate": 0.8,
                            "risk_level": "high",
                        },
                    ]
                },
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Discount": "discount", "Profit": "profit", "Sales": "sales_amount"},
        confidence=0.9,
    )

    rows = action_plan_rows(
        report,
        schema_mapping,
        dataset_profile={"has_discount": True, "has_profit": True},
        analysis_focus={"selected_focuses": ["discount_erosion_focus"]},
    )

    discount_row = next(row for row in rows if "折扣" in row["issue"])
    assert "30%+" in discount_row["evidence"]
    assert "80.00%" in discount_row["evidence"]
    assert "折扣利润「0%」" not in discount_row["evidence"]


def test_discount_action_prefers_threshold_candidates_over_category_risk() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                tables={
                    "discount_threshold_candidates": [
                        {
                            "threshold": "审批层级 A",
                            "row_count": 9,
                            "sales_amount": 900.0,
                            "avg_profit": 20.0,
                            "profit_margin": 0.18,
                            "negative_profit_rate": 0.0,
                            "risk_level": "low",
                        },
                        {
                            "threshold": "审批层级 B",
                            "row_count": 8,
                            "sales_amount": 800.0,
                            "avg_profit": -12.0,
                            "profit_margin": -0.04,
                            "negative_profit_rate": 0.4,
                            "risk_level": "medium",
                        },
                        {
                            "threshold": "审批层级 C",
                            "row_count": 7,
                            "sales_amount": 700.0,
                            "avg_profit": -8.0,
                            "profit_margin": -0.02,
                            "negative_profit_rate": 0.7,
                            "risk_level": "high",
                        },
                    ],
                    "category_discount_risk": [
                        {
                            "Category": "Generic Category",
                            "discount_bucket": "Category Risk Bucket",
                            "sales_amount": 999999.0,
                            "avg_profit": -500.0,
                            "profit_margin": -0.5,
                            "negative_profit_rate": 1.0,
                            "risk_level": "high",
                        }
                    ],
                },
            )
        ],
    )

    rows = action_plan_rows(
        report,
        _mapping({"Discount": "discount", "Profit": "profit", "Sales": "sales_amount"}),
        dataset_profile={"has_discount": True, "has_profit": True},
        analysis_focus={"selected_focuses": ["discount_erosion_focus"]},
    )

    discount_row = next(row for row in rows if "折扣" in row["issue"])
    assert "审批层级 C" in discount_row["evidence"]
    assert "70.00%" in discount_row["evidence"]
    assert "风险等级为 high" in discount_row["evidence"]
    assert "Generic Category" not in discount_row["evidence"]
    assert "Category Risk Bucket" not in discount_row["evidence"]
    assert "审批层级 A" not in discount_row["evidence"]


def _profit_quality_report() -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品与类目分析",
                chart_type="bar",
                tables={
                    "category_profit_quality": [
                        {"category": "Bikes", "sales_amount": 120000, "profit": -8000, "profit_margin": -0.0667}
                    ],
                    "high_sales_low_profit_products": [
                        {"product_name": "Road Bike", "sales_amount": 88000, "profit": -4200, "profit_margin": -0.0477}
                    ],
                },
                findings=["Bikes 类目销售额较高但利润质量偏弱。"],
            )
        ],
    )


def _profit_quality_focus() -> dict[str, list[str]]:
    return {"selected_focuses": ["discount_erosion_focus", "profit_quality_focus", "product_concentration_focus"]}


def test_action_plan_without_discount_blocks_discount_actions_but_keeps_profit_quality_actions() -> None:
    rows = action_plan_rows(
        report=_profit_quality_report(),
        schema_mapping=_mapping(
            {
                "Category": "category",
                "Product": "product_name",
                "Sales": "sales_amount",
                "Profit": "profit",
            }
        ),
        dataset_profile={"has_discount": False, "has_profit": True, "negative_profit_rate": 0.18},
        analysis_focus=_profit_quality_focus(),
        evidence_pack={
            "dataset_signature": {"dominant_story": "discount_loss"},
            "focus_evidence": {
                "discount_erosion_focus": [
                    {"business_meaning": "折扣风险故事来自历史缓存，但当前字段不支持。"}
                ]
            },
        },
    )

    visible_text = " ".join(" ".join(row.values()) for row in rows).lower()

    assert "折扣" not in visible_text
    assert "discount" not in visible_text
    assert "审批阈值" not in visible_text
    assert "高折扣" not in visible_text
    assert any("类目利润质量" in row["issue"] or "商品" in row["issue"] for row in rows)
    assert "成本和定价结构" in visible_text or "成本、定价和履约费用" in visible_text


def test_action_plan_with_discount_preserves_category_and_product_action_wording() -> None:
    rows = action_plan_rows(
        report=_profit_quality_report(),
        schema_mapping=_mapping(
            {
                "Category": "category",
                "Product": "product_name",
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
            }
        ),
        dataset_profile={"has_discount": True, "has_profit": True, "negative_profit_rate": 0.18},
        analysis_focus=_profit_quality_focus(),
    )
    actions = [row["action"] for row in rows]

    assert "按类目复盘销售额、利润率和折扣结构，优先修正利润率偏低的类目策略。" in actions
    assert "下钻头部商品的折扣、成本和履约费用，调整低利润商品的定价或促销资源。" in actions
