"""Nexlink Telecom NOC — Decomposition-First Planner.

Generates the **entire** task plan in a single LLM call by asking the model to
decompose a user request into an ordered JSON array of sub-tasks. Each
sub-task is then routed (planning method + self-correction) and assembled into
a :class:`~planning.dag.TaskDAG`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from planning.dag import SubTask, TaskDAG
from planning.metrics import PlanningMetrics, timed_llm_call
from planning.router import route_subtask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Nexlink Telecom NOC planning engine. Given a user request and a set
of available tools, decompose the request into an ordered list of sub-tasks.

Respond with **only** a JSON object in the following format (no explanation):

{{"tasks": [
  {{
    "id": "t1",
    "description": "Human-readable description of this step",
    "dependencies": [],
    "tool_name": "<tool to call or null>",
    "tool_args": {{"arg_name": "value"}}
  }},
  {{
    "id": "t2",
    "description": "...",
    "dependencies": ["t1"],
    "tool_name": "<tool or null>",
    "tool_args": {{"node_id": "FROM:t1"}}
  }}
]}}

Rules:
- Each task must have a unique id (t1, t2, …).
- dependencies lists IDs of tasks that MUST complete before this task can start.
- Avoid circular dependencies.
- If a tool argument depends on a prior task's result, use "FROM:<task_id>".
- Keep plans focused: 2 to 5 sub-tasks is usually optimal.
- Respond with valid JSON only.
"""

_USER_PROMPT_TEMPLATE = """\
### Available Tools
{tool_descriptions}

### User Request
{request}

Decompose this request into sub-tasks. Respond with JSON only.
"""

_RETRY_PROMPT = """\
Your previous response could not be parsed as JSON.
Please re-generate the decomposition plan for the following request.
Respond with **ONLY** a valid JSON object matching {{"tasks": [...]}} — no markdown fences, no explanatory text.

### User Request
{request}
"""


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove optional markdown ```json … ``` wrappers."""
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction from an LLM response."""
    cleaned = _strip_code_fences(raw)

    # Attempt 1 — direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2 — find the first { … } block via brace matching
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    depth = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    raise ValueError(f"Failed to parse JSON from LLM response: {cleaned[:200]!r}")


# ---------------------------------------------------------------------------
# Tool description formatter
# ---------------------------------------------------------------------------


def _format_tool_descriptions(available_tools: list[dict]) -> str:
    """Render a concise textual description of each tool for the prompt."""
    lines: list[str] = []
    for tool in available_tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", tool.get("params", {}))
        param_parts: list[str] = []
        for pname, pinfo in params.items():
            ptype = pinfo if isinstance(pinfo, str) else pinfo.get("type", "any")
            param_parts.append(f"{pname}: {ptype}")
        sig = ", ".join(param_parts) if param_parts else ""
        lines.append(f"- **{name}**({sig}): {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_upfront(
    request: str,
    available_tools: list[dict],
    llm: Any,
    metrics: PlanningMetrics,
) -> TaskDAG:
    """Decompose *request* into a full :class:`TaskDAG` in one LLM call."""

    tool_descriptions = _format_tool_descriptions(available_tools)
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        tool_descriptions=tool_descriptions,
        request=request,
    )
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"

    # ---- First LLM call (tracked) -----------------------------------------
    raw_text: str = timed_llm_call(llm, full_prompt, metrics)

    # ---- Parse JSON --------------------------------------------------------
    try:
        plan = _extract_json(raw_text)
    except ValueError:
        logger.warning("First decomposition attempt returned invalid JSON — retrying.")
        retry_prompt = _RETRY_PROMPT.format(request=request)
        raw_text = timed_llm_call(llm, retry_prompt, metrics)
        plan = _extract_json(raw_text)  # let it raise on second failure

    # ---- Build DAG ---------------------------------------------------------
    tasks_raw: list[dict] = plan.get("tasks", [])
    if not tasks_raw:
        raise ValueError("LLM returned a plan with no tasks")

    dag = TaskDAG()

    for t in tasks_raw:
        sub_task = SubTask(
            id=t["id"],
            description=t.get("description", ""),
            tool_name=t.get("tool_name"),
            tool_args=t.get("tool_args", {}),
            dependencies=t.get("dependencies", []),
        )

        # Route the sub-task (sets planning_method & self_correction in-place)
        route_subtask(sub_task)

        dag.add_task(sub_task)

    logger.info(
        "Decomposition complete: %d tasks, execution order: %s",
        len(tasks_raw),
        dag.get_execution_order(),
    )

    return dag
