"""Nexlink Telecom NOC — Sub-task Router.

Decides which planning algorithm and self-correction method to use for each
sub-task based on lightweight heuristics (tool readiness, description keywords,
dependency structure).
"""

from __future__ import annotations

import re
from planning.dag import SubTask

# ---------------------------------------------------------------------------
# Keyword sets used by the routing heuristics
# ---------------------------------------------------------------------------

_TEXT_GENERATION_KEYWORDS: set[str] = {
    "report", "draft", "summarize", "notify", "document",
    "summary", "write", "compose", "describe", "explain",
}

_DECISION_KEYWORDS: set[str] = {
    "choose", "decide", "select", "compare", "evaluate",
    "which", "best", "recommend", "rank", "prioritize",
}

_ACTION_KEYWORDS: set[str] = {
    "migrate", "fix", "remediate", "resolve", "upgrade",
    "change", "patch", "restart", "rollback", "reconfigure",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _has_unresolved_args(tool_args: dict | None) -> bool:
    """Return *True* if any argument value contains a ``FROM:`` reference."""
    if not tool_args:
        return False
    for value in tool_args.values():
        if isinstance(value, str) and value.startswith("FROM:"):
            return True
    return False


def _description_matches(description: str, keywords: set[str]) -> bool:
    """Case-insensitive check whether *description* contains any keyword."""
    desc_lower = description.lower()
    # Use word-boundary matching to avoid false positives (e.g. "notify"
    # should not match inside "notification" — but we intentionally allow
    # stem-level matches for flexibility).
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}", desc_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_subtask(sub_task: SubTask) -> tuple[str, str]:
    """Determine the best planning and self-correction strategy for *sub_task*.

    The function applies a priority-ordered set of heuristics:

    1. **Direct tool call** — tool is known and all args are concrete.
    2. **Text generation** — description suggests a report / summary.
    3. **Decision making** — description suggests choosing among options.
    4. **State-changing action with deps** — needs external validation.
    5. **Fallback** — structured planning, no self-correction.

    Returns:
        A ``(planning_method, self_correction_method)`` tuple.

    Side-effects:
        Updates ``sub_task.planning_method`` and ``sub_task.self_correction``
        in-place so downstream code can inspect the routing decision directly
        on the task object.
    """

    description: str = sub_task.description or ""
    tool_name: str | None = getattr(sub_task, "tool_name", None)
    tool_args: dict | None = getattr(sub_task, "tool_args", None)
    dependencies: list = getattr(sub_task, "dependencies", []) or []

    # Heuristic 1 — fully-specified direct tool call
    if tool_name and tool_args is not None and not _has_unresolved_args(tool_args):
        planning_method, self_correction = "direct", "none"

    # Heuristic 2 — text-generation / reporting tasks
    elif _description_matches(description, _TEXT_GENERATION_KEYWORDS):
        planning_method, self_correction = "plan_and_solve", "self_refine"

    # Heuristic 3 — decision / evaluation tasks
    elif _description_matches(description, _DECISION_KEYWORDS):
        planning_method, self_correction = "tot", "self_refine"

    # Heuristic 4 — state-changing action *with* upstream dependencies
    elif _description_matches(description, _ACTION_KEYWORDS) and len(dependencies) > 0:
        planning_method, self_correction = "lats", "reflexion"

    # Heuristic 5 — safe default
    else:
        planning_method, self_correction = "plan_and_solve", "none"

    # Persist the decision on the SubTask object itself
    sub_task.planning_method = planning_method
    sub_task.self_correction = self_correction

    return planning_method, self_correction
