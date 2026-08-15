"""
LATS (Language Agent Tree Search) planning algorithm for Nexlink Telecom NOC.

An MCTS-inspired search that distinguishes itself from Tree of Thoughts by
using **external** feedback rather than LLM self-evaluation:

  1. **Select** — Pick the most promising unexplored branch (highest UCB-style
     score among untried candidates, or expand from the current best).
  2. **Expand** — Generate a candidate solution for the sub-task via the LLM,
     incorporating accumulated reflections from failed branches.
  3. **Simulate / Evaluate** — Validate the candidate by calling real MCP
     tools (e.g. ``run_network_diagnostic``, ``get_customer_network_status``)
     to check whether the proposed solution's claims hold against live
     database state.  The score comes from *external reality*, not the LLM.
  4. **Backpropagate** — Update the branch's score with the external feedback.
  5. **Reflect** — If a branch fails, the LLM generates a verbal reflection
     ("I failed because …") that is injected into subsequent prompts to steer
     the search away from repeated mistakes.
  6. **Repeat** — Up to ``max_iterations`` rounds.

This produces solutions that are grounded in verifiable data, not
self-assessed plausibility.
"""

import re
from typing import Any, Dict, List, Optional

from planning.dag import SubTask
from planning.metrics import PlanningMetrics, timed_llm_call, timed_tool_call


# ── Prompt templates ────────────────────────────────────────────────

_GENERATE_SOLUTION_PROMPT = """\
You are a Nexlink Telecom NOC problem-solver.

## Task
{description}

## Context from completed predecessor tasks
{context_block}

{reflection_section}

## Instructions
Propose a concrete, specific solution for this task.
Include exact node IDs, customer IDs, bandwidth values, or any other
quantitative details you believe are correct.
Your solution MUST be verifiable against the live NOC database.

Respond with your solution only — no preamble.
"""

_REFLECTION_PROMPT = """\
You are a Nexlink Telecom NOC analyst reflecting on a failed solution attempt.

## Task
{description}

## Attempted solution
{solution}

## External feedback
{feedback}

## Instructions
Explain in 2-3 sentences why this solution failed and what should be done
differently in the next attempt.  Be specific — mention concrete IDs,
metrics, or logical errors.

Start your response with "I failed because".
"""

_EXTRACT_SCORE_PROMPT = """\
You are a strict grading assistant.

## Task description
{description}

## Proposed solution
{solution}

## External tool feedback
{feedback}

## Instructions
Based ONLY on the external feedback (not your own judgement), assign a score
from 0.0 to 1.0 reflecting how well the proposed solution aligns with the
verified data.

- 1.0 = solution fully consistent with external data
- 0.5 = partially correct, some claims unverified or wrong
- 0.0 = solution contradicts external data entirely

Respond with EXACTLY one line:
SCORE: <float between 0.0 and 1.0>
"""

_FINAL_ANSWER_PROMPT = """\
You are a Nexlink Telecom NOC analyst.

## Task
{description}

## Best verified solution
{solution}

## External validation feedback
{feedback}

## Instructions
Produce a final, polished answer incorporating the verified data.
Be specific — include IDs, metrics, and actionable recommendations.
"""


# ── Branch data structure ───────────────────────────────────────────

class _Branch:
    """A single candidate branch in the LATS search tree."""

    __slots__ = ("solution", "score", "feedback", "reflection", "iteration")

    def __init__(
        self,
        solution: str,
        score: float = 0.0,
        feedback: str = "",
        reflection: str = "",
        iteration: int = 0,
    ) -> None:
        self.solution = solution
        self.score = score
        self.feedback = feedback
        self.reflection = reflection
        self.iteration = iteration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution": self.solution,
            "score": self.score,
            "feedback": self.feedback,
            "reflection": self.reflection,
        }


# ── Helpers ─────────────────────────────────────────────────────────

def _format_context(context: Dict[str, str]) -> str:
    if not context:
        return "(no prior context)"
    return "\n".join(f"- **{tid}**: {res}" for tid, res in context.items())


def _build_reflection_section(reflections: List[str]) -> str:
    """Format accumulated reflections for injection into the prompt."""
    if not reflections:
        return ""
    numbered = "\n".join(
        f"  {i}. {r}" for i, r in enumerate(reflections, 1)
    )
    return (
        "## Reflections from previous failed attempts\n"
        f"{numbered}\n\n"
        "Avoid repeating the same mistakes described above."
    )


_NODE_RE = re.compile(r"[Nn]ode[_# ]*(\d+)")
_CUSTOMER_RE = re.compile(r"[Cc]ustomer[_# ]*(\d+)")
_SCORE_RE = re.compile(r"SCORE:\s*([\d.]+)", re.IGNORECASE)


def _extract_ids(text: str) -> Dict[str, List[int]]:
    """Extract node and customer IDs from text for validation calls."""
    return {
        "node_ids": [int(m) for m in _NODE_RE.findall(text)],
        "customer_ids": [int(m) for m in _CUSTOMER_RE.findall(text)],
    }


def _parse_score(text: str) -> float:
    """Extract a float score from the grading LLM output."""
    m = _SCORE_RE.search(text)
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    return 0.0


async def _external_validate(
    solution: str,
    sub_task: SubTask,
    session,
    metrics: PlanningMetrics,
) -> str:
    """Call MCP tools to externally validate claims in the solution.

    Strategy:
      1. If the sub_task already has a ``tool_name``, call it directly.
      2. Otherwise parse the solution for node/customer IDs and call
         standard NOC tools to verify.

    Returns a combined feedback string from all tool outputs.
    """
    feedback_parts: List[str] = []

    # ── Direct tool on the sub-task ─────────────────────────────────
    if sub_task.tool_name:
        try:
            result = await timed_tool_call(
                session, sub_task.tool_name, sub_task.tool_args, metrics,
            )
            feedback_parts.append(
                f"[{sub_task.tool_name}] {result}"
            )
        except Exception as exc:
            feedback_parts.append(
                f"[{sub_task.tool_name}] Error: {exc}"
            )

    # ── ID-based verification probes ────────────────────────────────
    ids = _extract_ids(solution)

    for nid in ids["node_ids"][:3]:  # cap to avoid spamming
        for tool in ("run_network_diagnostic", "get_node_status"):
            try:
                result = await timed_tool_call(
                    session, tool, {"node_id": nid}, metrics,
                )
                feedback_parts.append(f"[{tool} node={nid}] {result}")
                break  # one successful tool per node is enough
            except Exception:
                continue  # tool may not exist; try next

    for cid in ids["customer_ids"][:3]:
        try:
            result = await timed_tool_call(
                session,
                "get_customer_network_status",
                {"customer_id": cid},
                metrics,
            )
            feedback_parts.append(
                f"[get_customer_network_status cust={cid}] {result}"
            )
        except Exception:
            pass  # tool may not be available

    if not feedback_parts:
        feedback_parts.append(
            "(no external validation tools available — treating as unverified)"
        )

    return "\n\n".join(feedback_parts)


# ── Main entry point ────────────────────────────────────────────────

async def lats_search(
    sub_task: SubTask,
    context: Dict[str, str],
    llm,
    session,
    metrics: PlanningMetrics,
    max_iterations: int = 4,
) -> str:
    """Execute a sub-task using LATS (Language Agent Tree Search).

    Parameters
    ----------
    sub_task : SubTask
        The DAG node to execute.
    context : dict
        Mapping of ``task_id`` → result from completed predecessors.
    llm : ChatOllama
        Synchronous LLM.
    session : mcp.ClientSession
        Async MCP session for tool calls.
    metrics : PlanningMetrics
        Accumulator for cost/latency tracking.
    max_iterations : int
        Maximum MCTS-style search iterations.

    Returns
    -------
    str
        The final answer derived from the best externally-validated branch.
    """
    context_block = _format_context(context)
    branches: List[_Branch] = []
    reflections: List[str] = []

    for iteration in range(max_iterations):
        # ── 1. Select & Expand ──────────────────────────────────────
        gen_prompt = _GENERATE_SOLUTION_PROMPT.format(
            description=sub_task.description,
            context_block=context_block,
            reflection_section=_build_reflection_section(reflections),
        )
        solution_text: str = timed_llm_call(llm, gen_prompt, metrics)

        # ── 2. Simulate / Evaluate (external) ──────────────────────
        feedback = await _external_validate(
            solution_text, sub_task, session, metrics,
        )

        # ── 3. Score based on external feedback ─────────────────────
        score_prompt = _EXTRACT_SCORE_PROMPT.format(
            description=sub_task.description,
            solution=solution_text,
            feedback=feedback,
        )
        score_text = timed_llm_call(llm, score_prompt, metrics)
        score = _parse_score(score_text)

        branch = _Branch(
            solution=solution_text,
            score=score,
            feedback=feedback,
            iteration=iteration,
        )

        # ── 4. Backpropagate ────────────────────────────────────────
        branches.append(branch)

        # Early exit: perfect score
        if score >= 0.9:
            break

        # ── 5. Reflect on failure ───────────────────────────────────
        reflect_prompt = _REFLECTION_PROMPT.format(
            description=sub_task.description,
            solution=solution_text,
            feedback=feedback,
        )
        reflection = timed_llm_call(llm, reflect_prompt, metrics)
        branch.reflection = reflection
        reflections.append(reflection)

    # ── Select the best branch ──────────────────────────────────────
    best_branch = max(branches, key=lambda b: b.score) if branches else None

    if best_branch is None:
        # Fallback: no branches at all (should not happen)
        return timed_llm_call(
            llm,
            f"Solve this Nexlink NOC task:\n{sub_task.description}\n"
            f"Context:\n{context_block}",
            metrics,
        )

    # ── Produce final polished answer ───────────────────────────────
    final_prompt = _FINAL_ANSWER_PROMPT.format(
        description=sub_task.description,
        solution=best_branch.solution,
        feedback=best_branch.feedback,
    )
    final_answer: str = timed_llm_call(llm, final_prompt, metrics)
    return final_answer
