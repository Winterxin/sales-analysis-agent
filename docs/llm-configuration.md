# LLM Configuration

Sales Analysis Agent can optionally enrich planning and narrative generation
with an LLM.

1. The application uses an OpenAI-compatible Chat Completions interface.
2. Required values are Base URL, API key, and model ID.
3. The service may query `/models` before sending requests to `/chat/completions`.
4. If `/models` is unavailable, explicitly configure `SALES_AGENT_LLM_MODEL`.
5. Any model available through an OpenAI-compatible endpoint can use the same configuration pattern.
6. LLM enrichment is optional.
7. Deterministic analysis remains available when LLM is disabled.
8. Never commit `.env` files, API keys, runtime artifacts, or LLM cache files.

Configure these values in `apps/api/.env`:

```env
SALES_AGENT_LLM_ENABLED=true
SALES_AGENT_LLM_PROVIDER=openai_compatible
SALES_AGENT_LLM_BASE_URL=
SALES_AGENT_LLM_API_KEY=
SALES_AGENT_LLM_MODEL=
```

When `SALES_AGENT_LLM_BASE_URL` and `SALES_AGENT_LLM_API_KEY` are not both set,
the LLM client is disabled and the deterministic analysis pipeline remains
available.
