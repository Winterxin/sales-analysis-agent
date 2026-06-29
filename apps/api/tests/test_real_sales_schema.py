from __future__ import annotations

import pandas as pd

from app.services.schema_rules import map_schema_fields
from app.services.time_parser import parse_datetime_series


def test_schema_rules_map_superstore_geography_and_business_dimensions() -> None:
    mapping = map_schema_fields(
        [
            "Order ID",
            "Order Date",
            "Ship Date",
            "Customer ID",
            "Country",
            "City",
            "State",
            "Region",
            "Segment",
            "Category",
            "Sub-Category",
            "Product ID",
            "Product Name",
            "Sales",
            "Quantity",
            "Discount",
            "Profit",
        ]
    )

    assert mapping.field_mapping["Country"] == "country"
    assert mapping.field_mapping["City"] == "city"
    assert mapping.field_mapping["State"] == "state"
    assert mapping.field_mapping["Region"] == "region"
    assert mapping.field_mapping["Segment"] == "segment"
    assert mapping.field_mapping["Sub-Category"] == "sub_category"
    assert mapping.field_mapping["Product ID"] == "sku"
    assert mapping.field_mapping["Discount"] == "discount"
    assert mapping.field_mapping["Profit"] == "profit"


def test_parse_datetime_series_handles_superstore_date_format() -> None:
    series = pd.Series(["17-Jul-16", "21-Dec-17", "19-Apr-16"])

    parsed = parse_datetime_series(series)

    assert parsed.notna().all()
    assert parsed.dt.year.tolist() == [2016, 2017, 2016]
