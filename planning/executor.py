"""
DAG Executor — walks the task DAG in topological order, dispatching
each sub-task to the appropriate planning algorithm and self-correction
method, collecting metrics throughout.
"""

import json
import re
from typing import Dict, Optional, Any
from planning.dag import SubTask, TaskDAG
from planning.metrics import PlanningMetrics, timed_llm_call, timed_tool_call


async def execute_direct_tool(
    sub_task: SubTask,
    context: Dict[str, str],
    session,
    metrics: PlanningMetrics,
) -> str:
    """Execute a simple direct tool call — no planning needed."""
    tool_name = sub_task.tool_name
    if not tool_name:
        return f"No tool specified for task {sub_task.id}"

    # Resolve any FROM: references in tool_args
    resolved_args = _resolve_args(sub_task.tool_args, context)
    return await timed_tool_call(session, tool_name, resolved_args, metrics)


def _resolve_args(args: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
    """
    Resolve FROM:task_id references in tool arguments.
    E.g., {"node_id": "FROM:t1"} -> extracts node_id from t1's result.
    """
    resolved = {}
    for key, value in args.items():
        if isinstance(value, str) and value.startswith("FROM:"):
            source_task_id = value[5:]
            source_result = context.get(source_task_id, "")
            resolved[key] = _extract_value(key, source_result)
        else:
            resolved[key] = value
    return resolved


def _extract_value(param_name: str, result_text: str) -> Any:
    """
    Extract a parameter value from a prior task's result text.
    Looks for common patterns like 'Node #10', 'Customer #405', etc.
    """
    if "node_id" in param_name.lower():
        match = re.search(r'[Nn]ode[_# ]*(\d+)', result_text)
        if match:
            return int(match.group(1))
    if "customer_id" in param_name.lower():
        match = re.search(r'[Cc]ustomer[_# ]*(\d+)', result_text)
        if match:
            return int(match.group(1))
    if "bandwidth" in param_name.lower():
        match = re.search(r'(\d+\.?\d*)\s*[Gg]bps', result_text)
        if match:
            return float(match.group(1))
    # Fallback: return the raw text
    return result_text


async def execute_with_algorithm(
    sub_task: SubTask,
    context: Dict[str, str],
    llm,
    session,
    metrics: PlanningMetrics,
) -> str:
    """Execute a sub-task using its assigned planning algorithm."""
    method = sub_task.planning_method

    if method == "direct":
        return await execute_direct_tool(sub_task, context, session, metrics)

    elif method == "plan_and_solve":
        from planning.plan_and_solve import plan_and_solve
        return await plan_and_solve(sub_task, context, llm, session, metrics)

    elif method == "tot":
        from planning.tree_of_thoughts import tree_of_thoughts
        return await tree_of_thoughts(sub_task, context, llm, session, metrics)

    elif method == "lats":
        from planning.lats import lats_search
        return await lats_search(sub_task, context, llm, session, metrics)

    else:
        # Fallback: use plan_and_solve
        from planning.plan_and_solve import plan_and_solve
        return await plan_and_solve(sub_task, context, llm, session, metrics)


async def apply_self_correction(
    output: str,
    sub_task: SubTask,
    llm,
    session,
    metrics: PlanningMetrics,
) -> str:
    """Apply the assigned self-correction method to a sub-task's output."""
    method = sub_task.self_correction

    if method == "none":
        return output

    elif method == "self_refine":
        from planning.self_refine import self_refine
        rubric = (
            f"The output should fully address: {sub_task.description}\n"
            f"It should contain accurate, specific information from the NOC system.\n"
            f"It should not contain fabricated data or unverified claims."
        )
        return await self_refine(output, rubric, sub_task, llm, session, metrics)

    elif method == "reflexion":
        from planning.reflexion import reflexion
        from planning.critique import grounded_critique

        async def task_fn(st, reflections):
            reflection_text = "\n".join(reflections) if reflections else ""
            prompt = (
                f"Task: {st.description}\n"
                f"Previous attempt reflections:\n{reflection_text}\n\n"
                f"Generate an improved solution:"
            )
            return timed_llm_call(llm, prompt, metrics)

        async def eval_fn(out, st):
            return await grounded_critique(out, st, session, metrics)

        return await reflexion(sub_task, task_fn, eval_fn, llm, metrics)

    return output


class DAGExecutor:
    """
    Executes a TaskDAG by walking tasks in topological order,
    dispatching each to the appropriate planning algorithm
    and applying self-correction.
    """

    def __init__(self, llm, session):
        self.llm = llm
        self.session = session

    async def execute(self, dag: TaskDAG, metrics: PlanningMetrics) -> Dict[str, Any]:
        """
        Execute all tasks in the DAG in dependency order.
        Returns a dict with final results and the DAG state.
        """
        max_iterations = len(dag.nodes) * 2  # safety limit
        iteration = 0

        while not dag.all_done() and iteration < max_iterations:
            iteration += 1
            ready = dag.get_ready_tasks()

            if not ready:
                # No tasks ready but not all done — likely a failed dependency
                break

            for task in ready:
                dag.mark_running(task.id)
                context = dag.get_completed_results()

                try:
                    # Step 1: Execute with planning algorithm
                    result = await execute_with_algorithm(
                        task, context, self.llm, self.session, metrics
                    )

                    # Step 2: Apply self-correction
                    corrected = await apply_self_correction(
                        result, task, self.llm, self.session, metrics
                    )

                    dag.mark_completed(task.id, corrected)

                except Exception as e:
                    dag.mark_failed(task.id, str(e))

        # Build final summary
        results = dag.get_completed_results()
        failed = {
            tid: t.error
            for tid, t in dag.nodes.items()
            if t.status == "failed"
        }

        return {
            "results": results,
            "failed": failed,
            "dag_summary": dag.summary(),
            "metrics": metrics.to_dict(),
            "all_completed": len(failed) == 0,
        }

    async def execute_dynamic(
        self,
        request: str,
        available_tools: list,
        metrics: PlanningMetrics,
    ) -> Dict[str, Any]:
        """
        Execute using dynamic/interleaved decomposition.
        Generates one sub-task at a time, executes it, then asks for the next.
        """
        from planning.decompose_dynamic import DynamicDecomposer

        decomposer = DynamicDecomposer()

        while True:
            completed_results = decomposer.dag.get_completed_results()

            next_task = decomposer.get_next_task(
                request, completed_results, available_tools,
                self.llm, metrics,
            )

            if next_task is None:
                break

            decomposer.dag.mark_running(next_task.id)
            context = decomposer.dag.get_completed_results()

            try:
                result = await execute_with_algorithm(
                    next_task, context, self.llm, self.session, metrics
                )
                corrected = await apply_self_correction(
                    result, next_task, self.llm, self.session, metrics
                )
                decomposer.dag.mark_completed(next_task.id, corrected)
            except Exception as e:
                decomposer.dag.mark_failed(next_task.id, str(e))
                # Dynamic decomposition can adapt after failure
                continue

        results = decomposer.dag.get_completed_results()
        failed = {
            tid: t.error
            for tid, t in decomposer.dag.nodes.items()
            if t.status == "failed"
        }

        return {
            "results": results,
            "failed": failed,
            "dag_summary": decomposer.dag.summary(),
            "metrics": metrics.to_dict(),
            "all_completed": len(failed) == 0,
        }
