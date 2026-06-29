from __future__ import annotations

from app.services.notebook.modeling_renderer import render_modeling_section_cells
from apps.api.tests.test_modeling_display import (
    _combined,
    _loss_module,
    _outcome,
    _sales_regression_module,
)


def _sources(cells: list) -> list[str]:
    return [str(cell.source) for cell in cells]


def _code_sources(cells: list) -> list[str]:
    return [str(cell.source) for cell in cells if cell.cell_type == "code"]


def _index_containing(sources: list[str], term: str) -> int:
    return next(index for index, source in enumerate(sources) if term in source)


def test_sales_regression_usable_is_code_first_without_static_markdown_tables() -> None:
    cells = render_modeling_section_cells(
        None,
        forecast_module=_sales_regression_module(status="usable", best_model="Ridge Regression"),
        modeling_outcome=_outcome(
            task="sales_amount_regression",
            status="regression_usable",
            value="medium",
        ),
        modeling_outcome_interpretation={
            "model_selection_takeaway": "Ridge Regression 在 holdout 中 MAE 更低，但仍需要结合 R2 判断解释力。",
            "cv_stability_takeaway": "TimeSeriesSplit 显示 Ridge Regression 的 cv_mae_mean 更低，优势不是只来自最后一段。",
            "prediction_fit_takeaway": "Actual vs Best Model 可以观察趋势，峰值和低谷仍可能漏掉。",
            "feature_importance_takeaway": "lag_1 和 rolling_mean_4 说明模型主要依赖历史销售惯性。",
            "error_analysis_takeaway": "误差最大的周期可能缺少促销、节假日或异常订单变量。",
            "final_regression_synthesis": "综合来看，该模型可作为监控参照，不是生产级预测，也不用于自动决策。",
        },
    )

    markdown, code = _combined(cells)
    sources = _sources(cells)

    assert "## Modeling Analysis: Sales Regression" in markdown
    assert "| fold | train_start | train_end | valid_start | valid_end | model | mae | rmse | mape | r2 |" not in markdown
    assert "| model | cv_mae_mean | cv_mae_std | cv_mape_mean | cv_mape_std | cv_r2_mean | cv_r2_std |" not in markdown
    assert "| model | model_type | mae | rmse | mape | r2 | improvement_vs_baseline |" not in markdown
    assert "| period | actual | predicted | absolute_error | percentage_error |" not in markdown

    assert "feature_df = sales_series.copy()" in code
    assert "target.shift(lag)" in code
    assert "rolling_source = target.shift(1)" in code
    assert "display(feature_df.head())" in code
    assert "feature_summary = pd.DataFrame" in code
    assert "train_df = feature_df.iloc[:-backtest_periods]" in code
    assert "split_summary = pd.DataFrame" in code
    assert "TimeSeriesSplit(n_splits=3)" in code
    assert "cv_folds = pd.DataFrame" in code
    assert "cv_summary = " in code
    assert "display(cv_summary)" in code
    assert "display(cv_folds.head(12))" not in code
    assert "TimeSeriesSplit CV Model Stability Comparison" in code
    assert "model_comparison = pd.DataFrame" in code
    assert "holdout_mae_rank" in code
    assert "cv_mae_rank" in code
    assert "cv_stability_note" in code
    assert "selected_reason" in code
    assert "selection_basis = pd.DataFrame" in code
    assert "cv_best_model" in code
    assert "display(model_comparison)" in code
    assert "backtest_df = pd.DataFrame" in code
    assert "display(backtest_df.head())" not in code
    assert "Actual vs Best Model" in code
    assert "feature_importance_df = " in code
    assert "display(feature_importance_df)" not in code
    assert "Sales Regression Feature Importance Top 8" in code
    assert "error_top5 = " in code
    assert "error_direction" in code
    assert "display(error_top5)" in code
    assert "train_test_split" not in code

    assert _index_containing(sources, "feature_df = sales_series.copy()") < _index_containing(
        sources, "train_df = feature_df.iloc[:-backtest_periods]"
    )
    assert _index_containing(sources, "train_df = feature_df.iloc[:-backtest_periods]") < _index_containing(
        sources, "TimeSeriesSplit(n_splits=3)"
    )
    assert _index_containing(sources, "TimeSeriesSplit(n_splits=3)") < _index_containing(
        sources, "model_comparison = pd.DataFrame"
    )
    assert _index_containing(sources, "model_comparison = pd.DataFrame") < _index_containing(
        sources, "backtest_df = pd.DataFrame"
    )
    assert _index_containing(sources, "backtest_df = pd.DataFrame") < _index_containing(
        sources, "feature_importance_df = "
    )
    assert _index_containing(sources, "feature_importance_df = ") < _index_containing(sources, "error_top5 = ")


def test_sales_regression_weak_stays_code_first_but_omits_feature_importance() -> None:
    cells = render_modeling_section_cells(
        None,
        forecast_module=_sales_regression_module(status="weak", best_model="naive_last_value"),
        modeling_outcome=_outcome(
            task="sales_amount_regression",
            status="regression_weak",
            value="low",
        ),
        modeling_outcome_interpretation={
            "model_selection_takeaway": "best model 仍接近 baseline，复杂模型没有带来稳定增益。",
            "cv_stability_takeaway": "TimeSeriesSplit 显示各折误差波动较大。",
            "prediction_fit_takeaway": "Actual vs Best Model 显示模型难以跟上突发峰值或低谷。",
            "error_analysis_takeaway": "Top 误差集中在销售尖峰，可能缺少促销变量。",
            "final_regression_synthesis": "当前结果更适合作为预测难度证据，不建议用于销售计划、库存、补货或经营目标制定。",
        },
    )

    markdown, code = _combined(cells)

    assert "## Modeling Analysis: Sales Forecast Attempt" in markdown
    assert "feature_df = sales_series.copy()" in code
    assert "TimeSeriesSplit(n_splits=3)" in code
    assert "model_comparison = pd.DataFrame" in code
    assert "selection_basis = pd.DataFrame" in code
    assert "display(cv_folds.head(12))" not in code
    assert "backtest_df = pd.DataFrame" in code
    assert "display(backtest_df.head())" not in code
    assert "error_top5 = " in code
    assert "feature_importance_df" not in code
    assert "Sales Regression Feature Importance Top 8" not in code
    assert "Feature Importance" not in markdown


def test_loss_risk_branch_does_not_gain_sales_regression_code() -> None:
    cells = render_modeling_section_cells(
        _loss_module(quality="strong"),
        modeling_outcome=_outcome(
            task="loss_risk_classification",
            status="strong_model",
            value="high",
        ),
    )

    markdown, code = _combined(cells)

    assert "## Modeling Analysis: Loss Risk Review" in markdown
    assert "Sales Regression" not in markdown
    assert "feature_df = sales_series.copy()" not in code
    assert "TimeSeriesSplit(n_splits=3)" not in code
