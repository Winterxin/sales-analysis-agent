from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.analysis.contracts import ModuleResult
from app.analysis.modules import (
    data_quality,
    country_market,
    dimension_breakdown,
    discount_profit,
    forecast,
    loss_risk_modeling,
    metric_distribution,
    order_structure,
    product_contribution,
    sales_trend,
)
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.canonical_fields import canonical_columns_for_frame, clean_numeric_series
from app.services.time_parser import parse_datetime_series


def _safe_loss_risk_modeling_run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    try:
        return loss_risk_modeling.run(frame, canonical_columns)
    except Exception as exc:
        return ModuleResult(
            module_id="loss_risk_modeling",
            title="亏损风险建模",
            chart_type="modeling",
            warnings=[f"modeling_module_error: {type(exc).__name__}: {exc}"],
        )


MODULE_RUNNERS = {
    "data_quality_check": data_quality.run,
    "metric_distribution_analysis": metric_distribution.run,
    "sales_trend_analysis": sales_trend.run,
    "product_contribution_analysis": product_contribution.run,
    "dimension_breakdown_analysis": dimension_breakdown.run,
    "country_market_analysis": country_market.run,
    "order_structure_analysis": order_structure.run,
    "discount_profit_analysis": discount_profit.run,
    "loss_risk_modeling": _safe_loss_risk_modeling_run,
    "forecast_analysis": forecast.run,
}


def _canonical_columns(schema_mapping: SchemaMapping) -> dict[str, str]:
    return {
        canonical: original for original, canonical in schema_mapping.field_mapping.items()
    }


def _read_frame(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin1")


def _prepare_datetime_fields(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> pd.DataFrame:
    canonical_columns = _canonical_columns(schema_mapping)
    if "order_datetime" in canonical_columns:
        frame[canonical_columns["order_datetime"]] = parse_datetime_series(
            frame[canonical_columns["order_datetime"]]
        )
    return frame


def _load_raw_frame(csv_path: Path, schema_mapping: SchemaMapping) -> pd.DataFrame:
    return _prepare_datetime_fields(_read_frame(csv_path), schema_mapping)


def _clean_numeric_fields(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> pd.DataFrame:
    canonical_columns = canonical_columns_for_frame(frame, schema_mapping)
    for numeric_field in ("quantity", "sales_amount", "unit_price", "discount", "profit"):
        if numeric_field in canonical_columns:
            frame[canonical_columns[numeric_field]] = clean_numeric_series(
                frame[canonical_columns[numeric_field]]
            ).fillna(0)
    return frame


def _load_frame(csv_path: Path, schema_mapping: SchemaMapping) -> pd.DataFrame:
    return _clean_numeric_fields(_load_raw_frame(csv_path, schema_mapping), schema_mapping)


def run_analysis(
    task_id: str, csv_path: Path, schema_mapping: SchemaMapping, plan: AnalysisPlan
) -> AnalysisReport:
    raw_frame = _load_raw_frame(csv_path, schema_mapping)
    frame = _clean_numeric_fields(raw_frame.copy(), schema_mapping)
    canonical_columns = canonical_columns_for_frame(frame, schema_mapping)
    module_results: list[ModuleResult] = []

    for module_id in plan.analysis_plan:
        runner = MODULE_RUNNERS.get(module_id)
        if runner is None:
            continue
        module_frame = raw_frame if module_id == "loss_risk_modeling" else frame
        module_results.append(runner(module_frame, canonical_columns))

    summary = [finding for module in module_results for finding in module.findings[:1]]
    return AnalysisReport(
        task_id=task_id,
        dataset_type=schema_mapping.dataset_type,
        module_count=len(module_results),
        summary=summary,
        modules=[ModuleReport.model_validate(module.model_dump()) for module in module_results],
    )
