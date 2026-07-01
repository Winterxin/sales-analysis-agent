from __future__ import annotations

import importlib
import inspect


def test_fallback_section_content_is_delegated_to_boundary_module() -> None:
    boundary = importlib.import_module("app.services.notebook_content_fallback")

    assert hasattr(boundary, "FallbackSectionHelpers")
    assert hasattr(boundary, "build_fallback_section_content")

    planner = importlib.import_module("app.services.notebook_content_planner")
    source = inspect.getsource(planner._fallback_section_content)

    assert "build_fallback_section_content" in source
    assert len(source.splitlines()) <= 50


def test_fallback_boundary_dispatches_to_section_builders() -> None:
    boundary = importlib.import_module("app.services.notebook_content_fallback")
    source = inspect.getsource(boundary.build_fallback_section_content)

    assert len(source.splitlines()) <= 250
    for name in [
        "_build_dataset_and_schema_fallback_section",
        "_build_metric_distributions_fallback_section",
        "_build_sales_trends_fallback_section",
        "_build_order_structure_fallback_section",
    ]:
        assert hasattr(boundary, name)
        assert name in source
