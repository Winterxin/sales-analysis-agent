from __future__ import annotations

from collections.abc import Callable, Iterable
import inspect
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.analysis.contracts import ModuleResult
from app.analysis.runner import MODULE_RUNNERS


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrendArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    granularity: Literal["day", "month"] = "month"


class ProductContributionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_n: int = Field(default=10, ge=5, le=20)


class DimensionBreakdownArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal["segment", "region", "state", "city", "country", "category"]
    metric: Literal["sales_amount", "profit"] = "sales_amount"
    top_n: int = Field(default=10, ge=5, le=20)


ToolExecutor = Callable[[pd.DataFrame, dict[str, str], dict[str, object]], ModuleResult]


class AnalysisToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    required_fields: frozenset[str] = Field(default_factory=frozenset)
    optional_fields: frozenset[str] = Field(default_factory=frozenset)
    required_any_of: tuple[frozenset[str], ...] = ()
    minimum_rows: int | None = None
    argument_model: type[BaseModel] = Field(
        default=EmptyArguments, exclude=True, repr=False
    )
    allow_distinct_argument_calls: bool = False
    executor: ToolExecutor = Field(exclude=True, repr=False)

    def missing_fields(
        self,
        available_fields: set[str],
        dataset_profile: dict[str, object] | None = None,
    ) -> list[str]:
        missing = sorted(self.required_fields - available_fields)
        for alternatives in self.required_any_of:
            if not alternatives & available_fields:
                missing.append("one_of:" + "|".join(sorted(alternatives)))
        if self.minimum_rows is not None and dataset_profile is not None:
            row_count = int(dataset_profile.get("row_count") or 0)
            if row_count < self.minimum_rows:
                missing.append(f"minimum_rows:{self.minimum_rows}")
        return missing


class AnalysisToolRegistry:
    def __init__(self, specs: Iterable[AnalysisToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> AnalysisToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def describe(self, name: str) -> dict[str, object] | None:
        spec = self.get(name)
        if spec is None:
            return None
        description = spec.model_dump(
            mode="json", exclude={"executor", "argument_model"}
        )
        description["argument_schema"] = spec.argument_model.model_json_schema()
        return description

    def is_available(
        self,
        name: str,
        available_fields: set[str],
        dataset_profile: dict[str, object] | None = None,
    ) -> tuple[bool, list[str]]:
        spec = self.get(name)
        if spec is None:
            return False, ["unknown_tool"]
        missing = spec.missing_fields(available_fields, dataset_profile)
        return not missing, missing

    def available_specs(
        self,
        available_fields: set[str],
        dataset_profile: dict[str, object] | None = None,
    ) -> list[AnalysisToolSpec]:
        return [
            spec
            for spec in self._specs.values()
            if not spec.missing_fields(available_fields, dataset_profile)
        ]

    def execute(
        self,
        name: str,
        frame: pd.DataFrame,
        canonical_columns: dict[str, str],
        arguments: dict[str, object] | None = None,
    ) -> ModuleResult:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"Unknown analysis tool: {name}")
        if len(inspect.signature(spec.executor).parameters) == 2:
            return spec.executor(frame, canonical_columns)  # type: ignore[call-arg]
        return spec.executor(frame, canonical_columns, arguments or {})


def _adapt_executor(executor: Callable[..., ModuleResult]) -> ToolExecutor:
    def adapted(
        frame: pd.DataFrame,
        canonical_columns: dict[str, str],
        arguments: dict[str, object],
    ) -> ModuleResult:
        return executor(frame, canonical_columns, **arguments)

    return adapted


def _spec(
    name: str,
    description: str,
    *,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
    required_any_of: tuple[Iterable[str], ...] = (),
    minimum_rows: int | None = None,
    argument_model: type[BaseModel] = EmptyArguments,
    allow_distinct_argument_calls: bool = False,
) -> AnalysisToolSpec:
    return AnalysisToolSpec(
        name=name,
        description=description,
        required_fields=frozenset(required),
        optional_fields=frozenset(optional),
        required_any_of=tuple(frozenset(group) for group in required_any_of),
        minimum_rows=minimum_rows,
        argument_model=argument_model,
        allow_distinct_argument_calls=allow_distinct_argument_calls,
        executor=_adapt_executor(MODULE_RUNNERS[name]),
    )


_REGISTRY = AnalysisToolRegistry(
    [
        _spec(
            "data_quality_check",
            "Inspect missing values, duplicates, field validity, and dataset quality risks.",
        ),
        _spec(
            "metric_distribution_analysis",
            "Profile the distribution and outliers of core numeric business metrics.",
            required_any_of=(("sales_amount", "quantity", "discount", "profit", "unit_price"),),
        ),
        _spec(
            "sales_trend_analysis",
            "Analyze sales movement and time-period changes.",
            required=("order_datetime", "sales_amount"),
            optional=("profit",),
            argument_model=TrendArguments,
            allow_distinct_argument_calls=True,
        ),
        _spec(
            "product_contribution_analysis",
            "Measure product or SKU contribution and concentration.",
            required=("sales_amount",),
            required_any_of=(("product_name", "sku", "category", "productline"),),
            optional=("profit",),
            argument_model=ProductContributionArguments,
            allow_distinct_argument_calls=True,
        ),
        _spec(
            "dimension_breakdown_analysis",
            "Compare performance across customer, product, or geographic dimensions.",
            required=("sales_amount",),
            required_any_of=(("segment", "region", "state", "city", "country", "category"),),
            optional=("profit",),
            argument_model=DimensionBreakdownArguments,
            allow_distinct_argument_calls=True,
        ),
        _spec(
            "country_market_analysis",
            "Analyze country-level market contribution and performance.",
            required=("country", "sales_amount"),
            optional=("profit",),
        ),
        _spec(
            "order_structure_analysis",
            "Analyze order, customer, basket, status, and deal-size structure.",
            required=("order_id",),
            required_any_of=(("customer_id", "quantity", "unit_price", "deal_size", "order_status"),),
            optional=("sales_amount",),
        ),
        _spec(
            "discount_profit_analysis",
            "Inspect profit performance and, when present, discount-related erosion signals.",
            required=("profit",),
            optional=("discount", "sales_amount"),
        ),
        _spec(
            "loss_risk_modeling",
            "Attempt bounded deterministic modeling of loss risk with graceful skip behavior.",
            required=("profit",),
            optional=("discount", "sales_amount", "quantity", "unit_price"),
        ),
        _spec(
            "forecast_analysis",
            "Produce a basic deterministic sales forecast when time coverage is sufficient.",
            required=("order_datetime", "sales_amount"),
            minimum_rows=30,
        ),
    ]
)


def get_analysis_tool_registry() -> AnalysisToolRegistry:
    return _REGISTRY
