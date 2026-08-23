# Bounded Analysis Agent

The application remains an outer deterministic workflow:

```text
upload -> validation -> schema mapping -> dataset profile
       -> analysis Agent subgraph -> notebook/report pipeline -> artifacts
```

Only analysis-tool selection is a LangGraph subgraph. This is a bounded,
constrained agentic workflow, not a fully autonomous Agent:

```text
goal -> plan -> Harness validate -> deterministic tool execute
                                  -> evidence -> inspect -> replan/finalize
```

## Responsibility boundary

- The LLM may select only names advertised by the Analysis Tool Registry.
- The Harness rejects unknown, unavailable, repeated, over-limit, or
  budget-blocked selections.
- Existing pandas module runners perform every numeric computation.
- Evidence is derived from the existing `ModuleReport` contract; the Agent does
  not create a second report system.
- The outer Notebook, revision, reflection, client-report, persistence, runtime,
  and cancellation stages are unchanged.

## State

`SalesAnalysisAgentState` carries the goal, dataset profile, available and
selected tools, execution records, compact evidence, round, decision, reason,
errors, and termination reason. Full DataFrames never enter graph state. Module
reports are internal execution outputs and are excluded from the public Agent
state artifact.

## Trace and observability

`analysis_agent_trace.json` is separate from final State. Every event has a
monotonic sequence, node, round, structured decision metadata, remaining budget,
and only concise reasoning summaries—not hidden chain-of-thought.

- `plan`, `plan_correction`, and `replan` record requested tools, decision reason,
  LLM call count, and the existing completion metrics window.
- `validate_plan` records requested, accepted, rejected tools and exact guard
  errors.
- `execute_tools` records successful tools and per-tool exceptions.
- `build_evidence` records before/after fact counts and exact new fact keys.
- `inspect` records enough/need-more, missing questions, suggested tools, metrics,
  and termination decisions.
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

`apps/api/evals/analysis_agent_golden_cases.json` contains 18 scripted cases.
`run_analysis_agent_eval.py` creates deterministic synthetic CSV inputs and uses
a scripted Planner/Inspector, so it is fast, repeatable, and requires no external
model. It measures pass rate, expected-tool matching, forbidden-tool violations,
fallback/termination/RePlan correctness, guard activations, and average tool calls
and rounds.

The Golden cases and focused tests inject Planner and Inspector exceptions,
invalid payload/status, unknown and missing-field tools, duplicates, budget
exhaustion, identical evidence, maximum rounds, complete tool failure, and
partial tool failure. This separates Planner, Harness, Executor, Evidence, and
Inspector failures in both tests and trace.
