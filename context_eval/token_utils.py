"""Shared helpers for context management strategies."""
from typing import List, Dict, Any

CRITICAL_KEYWORDS = [
    "vip", "sla", "compliance", "penalty", "breach", "critical", "outage",
    "downtime", "regulatory", "mandate", "escalation", "root cause",
    "firmware", "recall", "failover",
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate, consistent with memory/short_term.py."""
    return len(text.split()) * 4 // 3


def transcript_tokens(turns: List[Dict[str, Any]]) -> int:
    return sum(estimate_tokens(t.get("content", "")) for t in turns)


def is_high_signal(turn: Dict[str, Any]) -> bool:
    content = turn.get("content", "").lower()
    return any(kw in content for kw in CRITICAL_KEYWORDS)


def render(turns: List[Dict[str, Any]]) -> str:
    """Flatten a turn list to a single string, as it would be sent to the LLM."""
    lines = []
    for t in turns:
        lines.append(f"[{t.get('role', 'unknown')}/{t.get('type', 'msg')}] {t.get('content', '')}")
    return "\n".join(lines)
