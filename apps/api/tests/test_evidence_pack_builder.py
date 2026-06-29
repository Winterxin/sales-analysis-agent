from __future__ import annotations

from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.evidence_pack_builder import build_evidence_pack


def _mapping(fields: dict[str, str]) -> SchemaMapping:
    return SchemaMapping(dataset_type="sales_transaction", field_mapping=fields, confidence=1.0)


def _report() -> AnalysisReport:
    return AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["总销售额为 1,250,000。"],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润",
                chart_type="bar",
                findings=["30%+ 折扣桶亏损率为 97.77%。"],
                summary_metrics={"negative_profit_rate": 0.1872},
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献",
                chart_type="bar",
                findings=["头部商品销售占比为 10.65%。"],
                summary_metrics={"top_product_sales_share": 0.1065},
            ),
        ],
    )


def _leaf_text(payload: object) -> str:
    if isinstance(payload, dict):
        return " ".join(_leaf_text(value) for value in payload.values())
    if isinstance(payload, list):
        return " ".join(_leaf_text(value) for value in payload)
    return str(payload)


def test_superstore_profile_generates_discount_profit_segment_product_evidence() -> None:
    pack = build_evidence_pack(
        report=_report(),
        schema_mapping=_mapping(
            {
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Segment": "segment",
                "Region": "region",
                "Category": "category",
                "Product Name": "product_name",
            }
        ),
        dataset_profile={
            "has_profit": True,
            "has_discount": True,
            "region_count": 4,
            "segment_count": 3,
            "category_count": 3,
            "product_count": 1850,
            "negative_profit_rate": 0.1872,
            "profit_margin_spread": 0.77,
            "top_product_sales_share": 0.1065,
        },
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        section_priority={"core_sections": ["discount_and_profit", "product_and_category"]},
    )

    assert pack["dataset_signature"]["shape_type"] == "superstore_like"
    assert pack["dataset_signature"]["dominant_story"] == "discount_loss"
    assert pack["focus_evidence"]["discount_erosion_focus"][0]["formatted"] == "18.72%"
    assert any(item["fact_id"] == "product_count" for item in pack["distinctive_facts"])
    text = _leaf_text(pack).lower()
    assert "unknown" not in text
    assert "none" not in text


def test_online_retail_profile_generates_country_order_evidence_without_profit_discount() -> None:
    pack = build_evidence_pack(
        report=_report(),
        schema_mapping=_mapping(
            {
                "InvoiceNo": "order_id",
                "CustomerID": "customer_id",
                "Country": "country",
                "Quantity": "quantity",
                "UnitPrice": "unit_price",
                "__sales_amount": "sales_amount",
            }
        ),
        dataset_profile={
            "has_country": True,
            "has_profit": False,
            "has_discount": False,
            "country_count": 38,
            "top_country_sales_share": 0.84,
            "order_count": 25900,
            "customer_count": 4372,
            "repeat_customer_rate": 0.6997,
            "line_per_order_avg": 20.9231,
        },
        analysis_focus={"selected_focuses": ["country_market_focus", "customer_order_structure_focus"]},
        section_priority={"core_sections": ["country_market", "order_structure"]},
    )

    text = str(pack)
    assert pack["dataset_signature"]["shape_type"] == "retail_like"
    assert pack["dataset_signature"]["dominant_story"] == "country_market"
    assert "country_market_focus" in pack["focus_evidence"]
    assert "customer_order_structure_focus" in pack["focus_evidence"]
    assert "discount_erosion_focus" not in pack["focus_evidence"]
    assert "profit_quality_focus" not in pack["focus_evidence"]
    assert "69.97%" in text


def test_sample_sales_profile_generates_productline_deal_status_evidence() -> None:
    pack = build_evidence_pack(
        report=_report(),
        schema_mapping=_mapping(
            {
                "PRODUCTLINE": "productline",
                "DEALSIZE": "deal_size",
                "STATUS": "order_status",
                "COUNTRY": "country",
                "SALES": "sales_amount",
            }
        ),
        dataset_profile={
            "productline_count": 7,
            "distinct_deal_size_count": 3,
            "distinct_order_status_count": 6,
            "country_count": 19,
            "monthly_volatility": 0.647,
        },
        analysis_focus={
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
            ]
        },
        section_priority={"core_sections": ["product_and_category", "order_structure"]},
    )

    assert pack["dataset_signature"]["shape_type"] == "productline_order_like"
    assert pack["dataset_signature"]["dominant_story"] == "productline_mix"
    assert "productline_performance_focus" in pack["focus_evidence"]
    assert "deal_size_focus" in pack["focus_evidence"]
    assert "order_status_focus" in pack["focus_evidence"]


def test_missing_fields_enter_limitations_without_unknown_values() -> None:
    pack = build_evidence_pack(
        report=_report(),
        schema_mapping=_mapping({"Sales": "sales_amount"}),
        dataset_profile={"has_profit": False, "has_discount": False},
        analysis_focus={"selected_focuses": ["data_limitation_focus"]},
        section_priority={"core_sections": []},
    )

    limitations = pack["limitations"]
    assert any(item["field"] == "profit" for item in limitations)
    assert any(item["field"] == "discount" for item in limitations)
    assert "数据限制说明" in str(limitations)
    text = _leaf_text(pack).lower()
    assert "unknown" not in text
    assert "none" not in text
