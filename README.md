# Sales Analysis Agent

Sales Analysis Agent is an open-source AI application for turning sales CSV data into reproducible analysis artifacts, executed notebooks, and business reports.

The system combines a bounded Agent workflow with deterministic pandas-based analysis. The Agent plans registered Tool Calls from the user goal, the Harness validates tool selection and parameters, deterministic modules compute the actual metrics, and the workflow inspects results before finalizing or performing a limited replan.

The project is designed around traceable outputs and explicit field-aware degradation:

- With Profit and Discount fields: profit quality and discount analysis are enabled.
- With Profit but no Discount: profit analysis stays enabled while discount-specific conclusions are excluded.
- Without Profit or Discount: those conclusions are explicitly limited instead of inferred.

## Project Overview

The application uses FastAPI for the service layer and LangGraph to coordinate the bounded analysis workflow. LLMs are used for planning and narrative generation, while numeric results remain owned by deterministic pandas modules.

The generated notebook is a primary reproducible artifact, so each result can be reviewed, re-run, and traced back to the uploaded data. When the model is unavailable, produces invalid decisions, or exceeds its budget, the system falls back to a deterministic analysis plan instead of blocking the whole run.

## What It Generates

A completed run can produce:

- `analysis.executed.ipynb`
- `analysis.ipynb`
- `business_review.md`
- `report.html`
- `report.json`
- `client_report.html`
- `client_report.json`
- `artifact_manifest.json`
- `llm_trace.json`
- `analysis_agent_state.json`
- `analysis_agent_trace.json`

## Key Features

- Bounded Agent loop with planning, validated Tool Calls, inspection, limited replan, and fallback
- LangGraph-based workflow with structured Agent state and per-node trace
- Tool whitelist and Harness guards for argument schemas, dataset capabilities, duplicate prevention, round limits, and budget
- Deterministic pandas modules for all numeric metrics, tables, and chart-ready data
- FastAPI service for CSV upload, task execution, status, artifacts, and UI routes
- Runtime task status with progress heartbeat, refresh recovery, LLM call state, and cooperative cancellation
- Schema mapping for flexible column names
- Sales trend, product/category, regional, discount, profit, and data-quality analysis
- Field-aware limits when optional columns are missing
- Executed notebook output with charts and supporting tables
- Business review and client report artifacts
- Reproducible scripted Golden Eval for the Agent workflow
- OpenAI-compatible Chat Completions configuration

## Supported Data Shape

The system does not require fixed column names. It maps available columns to
business concepts where possible.

Recommended field semantics:

- order date
- product or SKU
- sales amount
- quantity
- category / sub-category
- region / customer segment
- profit
- discount

## Quick Start

From the repository root:

```bash
cd apps/api
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pip install plotly seaborn matplotlib nbconvert nbclient ipykernel jupyter_client
```

Create local configuration:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Start the API and web UI:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010/app
```

## LLM Configuration

LLM enrichment is optional. Configure any OpenAI-compatible Chat Completions
endpoint with:

```env
SALES_AGENT_LLM_BASE_URL=
SALES_AGENT_LLM_API_KEY=
SALES_AGENT_LLM_MODEL=
```

The LLM is used for planning and narrative generation. Incomplete configuration falls back safely, and the deterministic analysis pipeline remains runnable when LLM enrichment is disabled.

The run endpoint accepts an optional `user_goal` query parameter. When omitted,
the Agent uses a general business-analysis goal.

See [docs/llm-configuration.md](docs/llm-configuration.md) for details.

## Output Artifacts

Artifacts are written under the configured runtime directory. The default local
runtime path is `runtime`.

Typical files include executed notebooks, HTML reports, JSON reports, trace
metadata, and manifest metadata. Runtime outputs are intentionally ignored by
Git.

Cooperative cancellation: the current stage is allowed to finish before the run
stops.

## Example Dataset

A small demonstration dataset is available at:

```text
examples/data/ecommerce_sales_sample.csv
```

It is intended for trying the upload-to-artifact workflow. It is not a benchmark
and does not represent production data.

## Architecture Overview

- FastAPI serves the upload, task, artifact, and static UI routes.
- The ingestion layer validates CSV files and builds a dataset profile.
- Schema mapping converts flexible column names into canonical sales fields.
- A LangGraph subgraph plans Tool Calls (`name + arguments`), validates, executes,
  inspects, applies a deterministic sufficiency floor, and can replan only the
  analysis-module selection stage.
- The LLM proposes registered tools and arguments; deterministic pandas modules
  remain the only component that computes tables, metrics, and chart-ready data.
- Harness guards enforce argument schemas, dataset capabilities, signature-based
  duplicate prevention, tool/round limits, no-progress termination, and budget.
- Disabled, failed, invalid, or budget-blocked Agent decisions fall back to the
  deterministic analysis plan.
- Notebook assembly produces executable and executed notebook artifacts.
- Report builders produce business and client-facing output files.

See [docs/analysis-agent-architecture.md](docs/analysis-agent-architecture.md)
for the analysis subgraph boundary and state contract.

This is a bounded, constrained agentic workflow, not a fully autonomous Agent.
The model proposes registered Tool Calls; the Harness owns validation and
termination, and deterministic modules own every numeric result. The model has
no arbitrary Python or code-execution permission. Identical calls are blocked by
a stable tool-name-plus-normalized-arguments signature; distinct arguments are
allowed only where a ToolSpec explicitly permits them.

## Failure Behavior

| Failure | Behavior |
|---|---|
| LLM disabled or budget exhausted | Run the deterministic fallback plan |
| Planner exception or invalid payload | Run the deterministic fallback plan |
| Unknown tool or missing required fields | Harness rejects the selection |
| Initial plan has no valid tool | One bounded correction, then fallback |
| Replan has no valid new tool | Finalize the evidence already collected |
| No new content-based evidence facts | Terminate with `no_progress` |
| Maximum rounds reached | Finalize at the configured bound |
| Tool execution failure | Trace the error and keep any valid sibling evidence |

Run the offline scripted Agent evaluation without an external LLM:

```bash
cd apps/api
python evals/run_analysis_agent_eval.py
```

## Limitations

- The project is designed for exploratory sales analysis, not financial, pricing, or credit decision automation.
- LLM-generated narrative should be reviewed before operational use.
- Model outputs are for manual review prioritization, not automatic decisions.
- Native Anthropic Messages API support is not included in v0.1.0.

## Development and Tests

Run the focused LLM client tests:

```bash
python -m pytest apps/api/tests/test_llm_client.py -q
```

Compile the application package:

```bash
python -m compileall apps/api/app
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
