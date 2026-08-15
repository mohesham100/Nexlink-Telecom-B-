"""
Tree of Thoughts (BFS variant) planning algorithm for Nexlink Telecom NOC.

At each depth level the algorithm:
  1. **Branch** — Generates ``breadth`` distinct candidate approaches for the
     current sub-task (or for extending the best approach from the prior level).
  2. **Evaluate** — Each candidate is self-scored by the LLM on a 1-10 scale
     with a justification.
  3. **Prune** — Only candidates that score ≥ 6 survive to the next level.
  4. **Deepen** — The surviving candidates are expanded for the next depth
     level, up to ``max_depth``.
  5. **Select** — The highest-scoring leaf across the entire tree is chosen
     and executed to produce the final answer.

This allows deliberate exploration of multiple reasoning paths before
committing to a solution.
"""

import re
from typing import Any, Dict, List, Optional

from planning.dag import SubTask
from planning.metrics import PlanningMetrics, timed_llm_call, timed_tool_call


# ── Prompt templates ────────────────────────────────────────────────

_GENERATE_CANDIDATES_PROMPT = """\
You are a Nexlink Telecom NOC planning expert.

## Task
{description}

## Context from completed predecessor tasks
{context_block}

{parent_section}

## Instructions
Generate exactly {breadth} DISTINCT candidate approaches for solving this task.
Each approach should be a concrete, actionable plan — not a vague idea.

Format your response as a numbered list:
1. <approach 1>
2. <approach 2>
...
"""

_EVALUATE_CANDIDATE_PROMPT = """\
You are a Nexlink Telecom NOC quality evaluator.

## Task
{description}

## Candidate approach
{approach}

## Instructions
Evaluate how likely this approach is to correctly and completely solve the task.
Consider accuracy, completeness, and feasibility in a NOC environment.

Respond with EXACTLY two lines:
SCORE: <integer 1-10>
JUSTIFICATION: <one-sentence reason>
"""

_EXECUTE_BEST_PROMPT = """\
You are a Nexlink Telecom NOC analyst.

## Task
{description}

## Context from completed predecessor tasks
{context_block}

## Chosen approach
{approach}

## Instructions
Execute this approach fully.  Provide a detailed, actionable answer with
specific IDs, metrics, and recommendations where relevant.
"""


# ── Data types ──────────────────────────────────────────────────────

class _Candidate:
    """A single node in the thought tree."""

    __slots__ = ("approach", "score", "justification", "depth", "parent_approach")

    def __init__(
        self,
        approach: str,
        score: int = 0,
        justification: str = "",
        depth: int = 0,
        parent_approach: Optional[str] = None,
    ) -> None:
        self.approach = approach
        self.score = score
        self.justification = justification
        self.depth = depth
        self.parent_approach = parent_approach

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approach": self.approach,
            "score": self.score,
            "justification": self.justification,
        }


# ── Helpers ─────────────────────────────────────────────────────────

def _format_context(context: Dict[str, str]) -> str:
    if not context:
        return "(no prior context)"
    return "\n".join(f"- **{tid}**: {res}" for tid, res in context.items())


def _parse_candidates(text: str) -> List[str]:
    """Parse a numbered list of candidate approaches from LLM output."""
    candidates: List[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        m = re.match(r"^\d+[\.\)]\s*(.*)", line)
        if m:
            candidates.append(m.group(1).strip())
        elif line and candidates:
            candidates[-1] += " " + line
    return candidates


_SCORE_RE = re.compile(r"SCORE:\s*(\d+)", re.IGNORECASE)
_JUSTIFICATION_RE = re.compile(r"JUSTIFICATION:\s*(.*)", re.IGNORECASE)


def _parse_evaluation(text: str) -> tuple[int, str]:
    """Extract score and justification from evaluator output."""
    score = 0
    justification = ""

    m_score = _SCORE_RE.search(text)
    if m_score:
        score = min(max(int(m_score.group(1)), 1), 10)

    m_just = _JUSTIFICATION_RE.search(text)
    if m_just:
        justification = m_just.group(1).strip()

    return score, justification


# ── Main entry point ────────────────────────────────────────────────

async def tree_of_thoughts(
    sub_task: SubTask,
    context: Dict[str, str],
    llm,
    session,
    metrics: PlanningMetrics,
    breadth: int = 3,
    max_depth: int = 2,
) -> str:
    """Execute a sub-task using BFS Tree of Thoughts.

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
    breadth : int
        Number of candidate approaches generated at each depth level.
    max_depth : int
        Maximum number of expansion levels (0-indexed).

    Returns
    -------
    str
        The final answer produced by executing the best thought path.
    """
    context_block = _format_context(context)
    best_overall: Optional[_Candidate] = None

    # Seed: the current level starts empty (root)
    current_parents: List[Optional[str]] = [None]

    for depth in range(max_depth):
        level_candidates: List[_Candidate] = []

        for parent_approach in current_parents:
            # ── Generate candidates ─────────────────────────────────
            if parent_approach:
                parent_section = (
                    f"## Parent approach (depth {depth - 1})\n{parent_approach}\n\n"
                    "Expand on this approach — generate {breadth} refined or "
                    "alternative sub-approaches."
                )
            else:
                parent_section = ""

            gen_prompt = _GENERATE_CANDIDATES_PROMPT.format(
                description=sub_task.description,
                context_block=context_block,
                parent_section=parent_section,
                breadth=breadth,
            )
            gen_text = timed_llm_call(llm, gen_prompt, metrics)
            raw_approaches = _parse_candidates(gen_text)

            # Fallback: if parsing found nothing, treat entire output as one
            if not raw_approaches:
                raw_approaches = [gen_text.strip()]

            # ── Evaluate each candidate ─────────────────────────────
            for approach_text in raw_approaches[:breadth]:
                eval_prompt = _EVALUATE_CANDIDATE_PROMPT.format(
                    description=sub_task.description,
                    approach=approach_text,
                )
                eval_text = timed_llm_call(llm, eval_prompt, metrics)
                score, justification = _parse_evaluation(eval_text)

                candidate = _Candidate(
                    approach=approach_text,
                    score=score,
                    justification=justification,
                    depth=depth,
                    parent_approach=parent_approach,
                )
                level_candidates.append(candidate)

        # ── Prune: keep candidates scoring ≥ 6 ─────────────────────
        survivors = [c for c in level_candidates if c.score >= 6]

        # If nothing survives, keep the single best candidate to avoid dead end
        if not survivors and level_candidates:
            survivors = [max(level_candidates, key=lambda c: c.score)]

        # Track the global best
        level_best = max(survivors, key=lambda c: c.score)
        if best_overall is None or level_best.score > best_overall.score:
            best_overall = level_best

        # Prepare parents for next depth
        current_parents = [c.approach for c in survivors]

    # ── Execute the best path ───────────────────────────────────────
    if best_overall is None:
        # Edge case: no candidates were generated at all
        return timed_llm_call(
            llm,
            f"Solve this Nexlink NOC task:\n{sub_task.description}\n"
            f"Context:\n{context_block}",
            metrics,
        )

    # If the sub_task has a tool, call it first for grounding data
    tool_context = ""
    if sub_task.tool_name:
        try:
            tool_result = await timed_tool_call(
                session, sub_task.tool_name, sub_task.tool_args, metrics,
            )
            tool_context = f"\n## Tool output ({sub_task.tool_name})\n{tool_result}\n"
        except Exception as exc:
            tool_context = f"\n## Tool call failed: {exc}\n"

    exec_prompt = _EXECUTE_BEST_PROMPT.format(
        description=sub_task.description,
        context_block=context_block + tool_context,
        approach=best_overall.approach,
    )
    final_answer: str = timed_llm_call(llm, exec_prompt, metrics)
    return final_answer
