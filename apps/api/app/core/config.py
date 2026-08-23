from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SALES_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    runtime_dir: Path = Field(default=Path("runtime"))
    database_url: str = Field(default="sqlite:///./sales_analysis_agent.db")
    max_upload_mb: int = Field(default=25)
    llm_enabled: bool = Field(default=True)
    llm_provider: str = Field(default="openai_compatible")
    llm_base_url: str | None = Field(default=None)
    llm_api_key: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)
    llm_completion_timeout_seconds: float | None = Field(default=None)
    llm_completion_wall_timeout_seconds: float | None = Field(default=None)
    llm_max_attempts: int = Field(default=3)
    llm_cache_enabled: bool = Field(default=True)
    llm_cache_dir: Path = Field(default=Path("runtime") / "llm_cache")
    llm_cache_version: str = Field(default="g4_llm_cache_v1")
    llm_profile: str = Field(default="full")
    llm_chart_selection_enabled: bool = Field(default=False)
    llm_chart_intent_planning_enabled: bool = Field(default=False)
    llm_intent_guided_chart_selection_enabled: bool = Field(default=False)
    max_postrun_reflections: int = Field(default=6)
    analysis_agent_max_rounds: int = Field(default=2)
    analysis_agent_max_tools_per_round: int = Field(default=4)
    analysis_agent_max_plan_corrections: int = Field(default=1)
    notebook_output_mode: str = Field(default="compact")
    demo_safe: bool = Field(default=False)
    demo_safe_run_budget_seconds: float = Field(default=85.0)
    demo_safe_min_stage_budget_seconds: float = Field(default=8.0)
    cliproxy_config_path: Path | None = Field(default=None)


def get_settings() -> Settings:
    return Settings()
