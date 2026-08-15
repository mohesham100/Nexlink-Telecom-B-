"""
Plan-and-Solve planning algorithm for Nexlink Telecom NOC.

Single-pass algorithm with three phases:
  1. **Plan** — One LLM call generates an explicit numbered step-by-step plan
     for the sub-task, incorporating context from completed dependencies.
  2. **Execute** — Each plan step is executed sequentially.  If a step
     references a known MCP tool, it is dispatched via the session; otherwise
     it is treated as a reasoning step handled by the LLM.
  3. **Synthesize** — One final LLM call merges all step results into a
     coherent answer.

No branching, no retries — a lean baseline planner.
"""

import re
from typing import Dict, List, Tuple

from planning.dag import SubTask
from planning.metrics import PlanningMetrics, timed_llm_call, timed_tool_call


# ── Prompt templates ────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a Nexlink Telecom NOC planning assistant.

## Task
{description}

## Context from completed predecessor tasks
{context_block}

## Instructions
Generate a clear, numbered step-by-step plan to accomplish this task.
Each step should be a single concrete action.
If a step requires calling an external tool, prefix it with [TOOL:<tool_name>].
For example: "1. [TOOL:get_customer_network_status] Retrieve the current network status for customer #405."

Output ONLY the numbered list — no preamble, no summary.
"""

_SYNTHESIZE_PROMPT = """\
You are a Nexlink Telecom NOC analyst.

## Original task
{description}

## Step-by-step execution results
{step_results}

## Instructions
Synthesize the results above into a single, coherent, actionable answer that
fully addresses the original task.  Be specific — include IDs, metrics, and
recommendations where relevant.
"""


# ── Helpers ─────────────────────────────────────────────────────────

def _format_context(context: Dict[str, str]) -> str:
    """Format predecessor context into a readable block."""
    if not context:
        return "(no prior context)"
    lines: List[str] = []
    for task_id, result in context.items():
        lines.append(f"- **{task_id}**: {result}")
    return "\n".join(lines)


def _parse_plan_steps(plan_text: str) -> List[str]:
    """Extract numbered steps from the LLM-generated plan.

    Accepts formats like ``1. …``, ``1) …``, or bare ``1 …``.
    """
    steps: List[str] = []
    for line in plan_text.strip().splitlines():
        line = line.strip()
        # Match lines starting with a number followed by . or ) or a space
        match = re.match(r"^\d+[\.\)]\s*(.*)", line)
        if match:
            steps.append(match.group(1).strip())
        elif line and steps:
            # Continuation line — append to the last step
            steps[-1] += " " + line
    return steps


_TOOL_TAG_RE = re.compile(r"\[TOOL:(\S+)\]", re.IGNORECASE)


def _extract_tool_from_step(step: str, default_tool: str | None) -> str | None:
    """Return the tool name mentioned in a step, or None."""
    m = _TOOL_TAG_RE.search(step)
    if m:
        return m.group(1)
    # Fall back: if the default tool name appears literally in the step text
    if default_tool and default_tool.lower() in step.lower():
        return default_tool
    return None


def _extract_tool_args_from_step(step: str) -> Dict[str, object]:
    """Best-effort extraction of tool arguments from natural-language step text.

    Looks for common NOC identifiers (node IDs, customer IDs, bandwidth).
    """
    args: Dict[str, object] = {}

    node_match = re.search(r"[Nn]ode[_# ]*(\d+)", step)
    if node_match:
        args["node_id"] = int(node_match.group(1))

    customer_match = re.search(r"[Cc]ustomer[_# ]*(\d+)", step)
    if customer_match:
        args["customer_id"] = int(customer_match.group(1))

    bw_match = re.search(r"(\d+\.?\d*)\s*[Gg]bps", step)
    if bw_match:
        args["bandwidth"] = float(bw_match.group(1))

    return args


# ── Main entry point ────────────────────────────────────────────────

async def plan_and_solve(
    sub_task: SubTask,
    context: Dict[str, str],
    llm,
    session,
    metrics: PlanningMetrics,
) -> str:
    """Execute a sub-task using the Plan-and-Solve algorithm.

    Parameters
    ----------
    sub_task : SubTask
        The DAG node to execute.
    context : dict
        Mapping of ``task_id`` → result string from completed predecessors.
    llm : ChatOllama
        Synchronous LLM (``llm.invoke(prompt).content``).
    session : mcp.ClientSession
        Async MCP session for tool calls.
    metrics : PlanningMetrics
        Accumulator for cost/latency tracking.

    Returns
    -------
    str
        The synthesised final answer for this sub-task.
    """
    # ── Phase 1: Plan ───────────────────────────────────────────────
    plan_prompt = _PLAN_PROMPT.format(
        description=sub_task.description,
        context_block=_format_context(context),
    )
    plan_text: str = timed_llm_call(llm, plan_prompt, metrics)
    steps = _parse_plan_steps(plan_text)

    # Safeguard: if the parser found nothing, treat the whole plan as one step
    if not steps:
        steps = [plan_text.strip()]

    # ── Phase 2: Execute each step ──────────────────────────────────
    step_results: List[str] = []

    for idx, step in enumerate(steps, start=1):
        tool_name = _extract_tool_from_step(step, sub_task.tool_name)

        if tool_name:
            # Merge extracted args with any pre-configured args on the sub_task
            extracted_args = _extract_tool_args_from_step(step)
            merged_args = {**sub_task.tool_args, **extracted_args}
            try:
                result = await timed_tool_call(
                    session, tool_name, merged_args, metrics,
                )
            except Exception as exc:
                result = f"[Tool error: {exc}]"
        else:
            # Reasoning step — ask the LLM
            reasoning_prompt = (
                f"You are a Nexlink Telecom NOC analyst.\n\n"
                f"Overall task: {sub_task.description}\n\n"
                f"Current step ({idx}/{len(steps)}): {step}\n\n"
                f"Prior step results:\n"
                + "\n".join(
                    f"  Step {i}: {r}" for i, r in enumerate(step_results, 1)
                )
                + "\n\nProvide a concise result for this step."
            )
            result = timed_llm_call(llm, reasoning_prompt, metrics)

        step_results.append(f"Step {idx} ({step}): {result}")

    # ── Phase 3: Synthesize ─────────────────────────────────────────
    synth_prompt = _SYNTHESIZE_PROMPT.format(
        description=sub_task.description,
        step_results="\n".join(step_results),
    )
    final_answer: str = timed_llm_call(llm, synth_prompt, metrics)
    return final_answer
