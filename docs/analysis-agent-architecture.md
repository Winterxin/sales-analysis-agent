# Bounded Analysis Agent

The application remains an outer deterministic workflow:

```text
upload -> validation -> schema mapping -> dataset profile
       -> analysis Agent subgraph -> notebook/report pipeline -> artifacts
```

Only analysis-tool selection is a LangGraph subgraph. This is a bounded,
constrained agentic workflow, not a fully autonomous Agent:

```text
goal -> Planner -> Tool Calls (name + arguments) -> Harness
     -> deterministic tools -> evidence -> Inspector
     -> deterministic Sufficiency Guard -> replan/finalize
```

## Responsibility boundary

- The LLM may propose only structured calls to tools advertised by the Analysis
  Tool Registry. It has no arbitrary Python or code-execution capability.
- The Harness validates the tool whitelist, Pydantic argument schema, required
  fields, dataset capabilities, normalized call signature, limits, and budget.
- Existing pandas module runners perform every numeric computation.
- Evidence is derived from the existing `ModuleReport` contract; the Agent does
  not create a second report system.
- The outer Notebook, revision, reflection, client-report, persistence, runtime,
  and cancellation stages are unchanged.

## State

`SalesAnalysisAgentState` carries the goal, dataset profile, available and
requested/validated tool calls, execution records with arguments and signatures,
compact evidence, round, decision, reason,
errors, and termination reason. Full DataFrames never enter graph state. Module
reports are internal execution outputs and are excluded from the public Agent
state artifact.

## Trace and observability

`analysis_agent_trace.json` is separate from final State. Every event has a
monotonic sequence, node, round, structured decision metadata, remaining budget,
and only concise reasoning summaries—not hidden chain-of-thought.

- `plan`, `plan_correction`, and `replan` record requested tool calls, decision reason,
  LLM call count, and the existing completion metrics window.
- `validate_plan` records requested, accepted, rejected calls and exact argument
  or capability validation errors.
- `execute_tools` records validated arguments, successful calls, and per-tool
  exceptions.
- `build_evidence` records before/after fact counts and exact new fact keys.
- `inspect` records enough/need-more, missing questions, suggested tools, metrics,
  and termination decisions.
- `sufficiency_guard` records recognized, satisfied, missing, and unavailable goal
  capabilities, suggestions, deterministic action, and termination reason.
- `fallback` and `finalize` record the final path and termination reason.

The trace summary reports rounds, Planner/Inspector calls, requested/accepted/
rejected/successful/failed tool calls, replans, no-progress, fallback use, LLM
latency and character counts, and cache hits. It does not estimate tokens when
the provider does not return token usage.

## Invalid-plan correction

If the first Planner response requests tools but the Harness rejects all of them,
the Planner receives the structured rejection reasons and gets one correction.
A second all-invalid plan terminates through `initial_plan_invalid_fallback`. An
empty initial plan with no evidence uses `initial_plan_empty_fallback` directly.

If a later RePlan has no valid new tool, the Agent preserves its existing report
and evidence and finalizes with `replan_no_valid_tools`; it does not replace useful
work with a new full fallback run.

## Parameterized Tool Calls

The Planner's primary protocol is `tool_calls`, where every call contains
`tool_name` and `arguments`. A compatibility adapter accepts legacy
`selected_tools` fixtures, but graph state, validation, execution, and trace use
Tool Calls only.

Three tools expose deliberately small schemas:

- `sales_trend_analysis`: `granularity = day | month` (default `month`).
- `product_contribution_analysis`: `top_n` from 5 through 20 (default `10`).
- `dimension_breakdown_analysis`: required mapped `dimension`, `metric =
  sales_amount | profit`, and `top_n` from 5 through 20.

All other tools use an `EmptyArguments` schema with extra fields forbidden.
Argument validity alone is insufficient: the Harness also rejects a requested
dimension or metric that is not mapped in the current dataset. Validated
arguments are normalized to stable JSON and SHA-256 hashed with the tool name.
Identical calls therefore share a signature regardless of key order. The three
parameterized tools permit distinct legal signatures, while fixed-capability
tools cannot use arguments to bypass duplicate protection. Global round and
per-round limits still apply.

## Inspector and deterministic sufficiency

The Inspector remains the LLM's soft semantic judgment. An `enough` result must
then pass a deterministic floor: at least one successful execution, non-error
content evidence, and evidence for any transparently recognized goal capability
(profit, discount, trend, product, region, or customer). Unknown wording uses
the generic evidence floor and is not rejected merely because no keyword matched.

If a mapped capability is missing and another round is available, the Guard
forces RePlan and supplies a concrete Tool Call suggestion. If the underlying
field is unmapped, it finalizes existing evidence with `capability_unavailable`.
At the round limit it finalizes with `sufficiency_guard_unresolved`; without any
valid result it takes deterministic fallback. The Guard is rule-based and never
calls an LLM.

## Content-based progress

Progress is a deterministic set difference over normalized metric key/value
facts, finding text, and result/warning signals. Evidence IDs, rounds, timestamps,
and tool names are excluded, so repeating the same facts under new metadata does
not count as progress. A successful execution with no new facts terminates rather
than spending another LLM round. With no facts at all it uses deterministic
fallback; with existing facts it finalizes them with `no_progress`.

## Termination and fallback

The loop defaults to two rounds and at most four tools per round. It terminates
when evidence is sufficient, the round limit is reached, or a round produces no
new valid evidence. If the LLM is disabled, planning/inspection fails, a payload
is invalid, or the shared run budget is unavailable, the existing deterministic
`AnalysisPlan` runs instead.

## Golden Eval and failure injection

`apps/api/evals/analysis_agent_golden_cases.json` contains 33 scripted cases: the
original 18 plus eight argument/signature cases and seven sufficiency cases.
`run_analysis_agent_eval.py` creates deterministic synthetic CSV inputs and uses
a scripted Planner/Inspector, so it is fast, repeatable, and requires no external
model. It measures pass rate, expected-tool matching, forbidden-tool violations,
fallback/termination/RePlan correctness, argument validation, duplicate-signature
correctness, sufficiency/capability-unavailable correctness, guard activations,
and average tool calls and rounds.

The Golden cases and focused tests inject Planner and Inspector exceptions,
invalid payload/status, unknown and missing-field tools, duplicates, budget
exhaustion, identical evidence, maximum rounds, complete tool failure, and
partial tool failure. This separates Planner, Harness, Executor, Evidence, and
Inspector failures in both tests and trace.
