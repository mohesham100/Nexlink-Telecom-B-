"""Nexlink Telecom NOC — Dynamic (Interleaved) Decomposer.

Instead of generating the full plan upfront, this module produces **one
sub-task at a time**. After each task executes, the LLM is shown the real
results so it can adapt the plan — choosing different tools, skipping steps,
or adding new ones as the situation evolves.
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
You are a Nexlink Telecom NOC step-by-step planner.

You will be given:
1. The original user request.
2. A list of completed tasks with their results (may be empty on the first call).
3. The set of available tools.

Your job: decide the **single next step** to make progress on the request.

Respond with **only** a JSON object in one of two formats:

If there is more work to do:
{{"id": "t_{N}", "description": "...", "tool_name": "<tool or null>", "tool_args": {{...}}}}

If the request is fully addressed:
{{"status": "DONE", "summary": "Brief summary of what was accomplished"}}

Rules:
- Only use tools from the provided list.
- tool_args values must be concrete (use results from completed tasks).
- Do not repeat a step that has already been completed.
- Respond with JSON only — no explanation, no markdown fences.
"""

_USER_PROMPT_TEMPLATE = """\
### Available Tools
{tool_descriptions}

### Original Request
{request}

### Completed Tasks
{completed_section}

What is the next step? Respond with JSON only.
"""

# ---------------------------------------------------------------------------
# Helpers
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


def _format_completed(completed_results: dict[str, Any]) -> str:
    """Format the completed-tasks dict into a readable prompt section."""
    if not completed_results:
        return "(none yet)"
    lines: list[str] = []
    for task_id, result in completed_results.items():
        result_str = str(result)
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "…(truncated)"
        lines.append(f"- **{task_id}**: {result_str}")
    return "\n".join(lines)


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

    # Attempt 2 — find the first { … } block
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
# Public API
# ---------------------------------------------------------------------------


class DynamicDecomposer:
    """Generates sub-tasks one at a time, re-planning after each result."""

    def __init__(self) -> None:
        self.dag = TaskDAG()
        self.task_counter: int = 0

    def get_next_task(
        self,
        request: str,
        completed_results: dict[str, Any],
        available_tools: list[dict],
        llm: Any,
        metrics: PlanningMetrics,
    ) -> SubTask | None:
        """Generate the next sub-task or return ``None`` when done."""

        tool_descriptions = _format_tool_descriptions(available_tools)
        completed_section = _format_completed(completed_results)

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            tool_descriptions=tool_descriptions,
            request=request,
            completed_section=completed_section,
        )
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"

        # ---- LLM call (tracked) -------------------------------------------
        raw_text: str = timed_llm_call(llm, full_prompt, metrics)

        # ---- Parse ---------------------------------------------------------
        try:
            data = _extract_json(raw_text)
        except ValueError:
            logger.warning(
                "Dynamic decomposer received invalid JSON — retrying once."
            )
            retry_prompt = (
                "Your previous response was not valid JSON. Respond with ONLY "
                "a JSON object — no fences, no prose.\n\n" + user_prompt
            )
            raw_text = timed_llm_call(llm, retry_prompt, metrics)
            data = _extract_json(raw_text)

        # ---- Check for DONE -----------------------------------------------
        if data.get("status", "").upper() == "DONE":
            summary = data.get("summary", "Plan complete.")
            logger.info("Dynamic decomposition complete: %s", summary)
            return None

        # ---- Build SubTask -------------------------------------------------
        self.task_counter += 1
        task_id = data.get("id", f"t_{self.task_counter}")

        # Dependencies: all previously completed tasks (linear chain)
        dependencies = list(completed_results.keys())

        sub_task = SubTask(
            id=task_id,
            description=data.get("description", ""),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args", {}),
            dependencies=dependencies,
        )

        # Route the sub-task
        route_subtask(sub_task)

        # Add to the internal DAG for traceability
        self.dag.add_task(sub_task)

        logger.info(
            "Dynamic step %d: %s → %s (planning=%s, correction=%s)",
            self.task_counter,
            sub_task.id,
            sub_task.tool_name,
            sub_task.planning_method,
            sub_task.self_correction,
        )

        return sub_task
