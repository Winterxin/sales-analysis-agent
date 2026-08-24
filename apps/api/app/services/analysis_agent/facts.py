from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from app.services.analysis_agent.state import AgentEvidence


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _normalize_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value).lower()
        return format(value, ".12g")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, str):
        return _normalize_text(value)
    return str(value)


def evidence_fact_set(evidence: list[AgentEvidence]) -> set[str]:
    facts: set[str] = set()
    for item in evidence:
        for key, value in sorted(item.summary_metrics.items()):
            facts.add(f"metric:{key.casefold()}={_normalize_value(value)}")
        for finding in item.findings:
            normalized = _normalize_text(finding)
            if normalized:
                facts.add(f"finding:{normalized}")
        for signal in item.result_signals:
            normalized = _normalize_text(signal)
            if normalized:
                facts.add(f"signal:{normalized}")
    return facts


def evidence_fact_fingerprint(facts: set[str]) -> str:
    encoded = "\n".join(sorted(facts)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
