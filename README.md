# Sales Analysis Agent

Sales Analysis Agent is an open-source, notebook-first sales analysis application.
Upload a sales CSV and generate a reproducible analysis package: an executed
notebook, charts, a business review, and client-ready reports.

The project is designed around traceable outputs and explicit field-aware
degradation:

- With Profit and Discount fields: profit quality and discount analysis are enabled.
- With Profit but no Discount: profit analysis stays enabled while discount-specific conclusions are excluded.
- Without Profit or Discount: those conclusions are explicitly limited instead of inferred.

## Project Overview

The application combines deterministic pandas-based analysis with optional LLM
enrichment for planning and narrative generation. The generated notebook is the
primary artifact, so each result can be reviewed, re-run, and traced back to the
uploaded data.

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

## Key Features

- CSV upload workflow for sales transaction data
- Schema mapping for flexible column names
- Sales trend, product/category, regional, discount, profit, and data-quality analysis
- Field-aware limits when optional columns are missing
- Executed notebook output with charts and supporting tables
- Business review and client report artifacts
- Runtime task status with progress heartbeat, refresh recovery, LLM call state, and cooperative cancellation
- Optional LLM enrichment through an OpenAI-compatible Chat Completions endpoint

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

The LLM enriches planning and narrative generation. It is optional: incomplete
configuration falls back safely, and the deterministic analysis pipeline remains
runnable when LLM enrichment is disabled.

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
- Deterministic analysis modules compute tables, metrics, and chart-ready data.
- Optional LLM stages enrich analysis planning and narrative text.
- Notebook assembly produces executable and executed notebook artifacts.
- Report builders produce business and client-facing output files.

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
