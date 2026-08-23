# Bounded Analysis Agent

The application remains an outer deterministic workflow:

```text
upload -> validation -> schema mapping -> dataset profile
       -> analysis Agent subgraph -> notebook/report pipeline -> artifacts
```

Only analysis-tool selection is a LangGraph subgraph:

```text
plan -> validate_plan -> execute_tools -> build_evidence -> inspect
  ^                                                        |
  |---------------- replan when evidence is insufficient --|
                                      otherwise -> finalize
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

## Termination and fallback

The loop defaults to two rounds and at most four tools per round. It terminates
when evidence is sufficient, the round limit is reached, or a round produces no
new valid evidence. If the LLM is disabled, planning/inspection fails, a payload
is invalid, or the shared run budget is unavailable, the existing deterministic
`AnalysisPlan` runs instead.
