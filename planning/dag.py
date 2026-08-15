"""
DAG (Directed Acyclic Graph) data structure for task decomposition.

Enforces acyclicity at construction time — a plan that can deadlock
is a bug, not an edge case.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import json
import time


@dataclass
class SubTask:
    """A single node in the task DAG."""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    # Which planning algorithm to use: direct, plan_and_solve, tot, lats
    planning_method: str = "direct"
    # Which self-correction to apply: none, self_refine, reflexion
    self_correction: str = "none"
    # Execution state
    status: str = "pending"   # pending, running, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "dependencies": self.dependencies,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "planning_method": self.planning_method,
            "self_correction": self.self_correction,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


class TaskDAG:
    """
    Directed Acyclic Graph of SubTasks with topological execution ordering.
    Enforces acyclicity on every add_task call.
    """

    def __init__(self):
        self.nodes: Dict[str, SubTask] = {}
        self._execution_order: List[str] = []

    def add_task(self, task: SubTask) -> None:
        """Add a task to the DAG. Raises ValueError if it would create a cycle."""
        # Validate dependencies exist or will exist
        for dep_id in task.dependencies:
            if dep_id == task.id:
                raise ValueError(f"Task '{task.id}' cannot depend on itself.")

        # Temporarily add the task to check for cycles
        self.nodes[task.id] = task

        if self._has_cycle():
            del self.nodes[task.id]
            raise ValueError(
                f"Adding task '{task.id}' with dependencies {task.dependencies} "
                f"would create a cycle in the DAG."
            )

        # Recompute execution order
        self._execution_order = self._topological_sort()

    def _has_cycle(self) -> bool:
        """DFS-based cycle detection."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {nid: WHITE for nid in self.nodes}

        def dfs(node_id: str) -> bool:
            color[node_id] = GRAY
            task = self.nodes[node_id]
            for dep_id in task.dependencies:
                if dep_id not in color:
                    continue  # dependency not yet added — allowed
                if color[dep_id] == GRAY:
                    return True  # back edge → cycle
                if color[dep_id] == WHITE and dfs(dep_id):
                    return True
            color[node_id] = BLACK
            return False

        for nid in self.nodes:
            if color[nid] == WHITE:
                if dfs(nid):
                    return True
        return False

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological ordering."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        for nid, task in self.nodes.items():
            for dep_id in task.dependencies:
                if dep_id in self.nodes:
                    adj[dep_id].append(nid)
                    in_degree[nid] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for neighbor in adj[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def get_execution_order(self) -> List[str]:
        """Return the topological execution order."""
        return list(self._execution_order)

    def get_ready_tasks(self) -> List[SubTask]:
        """Return tasks whose dependencies are all completed and that are still pending."""
        ready = []
        for task_id in self._execution_order:
            task = self.nodes[task_id]
            if task.status != "pending":
                continue
            deps_done = all(
                self.nodes[dep_id].status == "completed"
                for dep_id in task.dependencies
                if dep_id in self.nodes
            )
            if deps_done:
                ready.append(task)
        return ready

    def mark_running(self, task_id: str) -> None:
        """Mark a task as currently running."""
        self.nodes[task_id].status = "running"
        self.nodes[task_id].start_time = time.time()

    def mark_completed(self, task_id: str, result: str) -> None:
        """Mark a task as successfully completed with its result."""
        self.nodes[task_id].status = "completed"
        self.nodes[task_id].result = result
        self.nodes[task_id].end_time = time.time()

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message."""
        self.nodes[task_id].status = "failed"
        self.nodes[task_id].error = error
        self.nodes[task_id].end_time = time.time()

    def get_task(self, task_id: str) -> SubTask:
        """Get a task by ID."""
        return self.nodes[task_id]

    def get_completed_results(self) -> Dict[str, str]:
        """Return a dict of task_id -> result for all completed tasks."""
        return {
            tid: task.result
            for tid, task in self.nodes.items()
            if task.status == "completed" and task.result is not None
        }

    def all_done(self) -> bool:
        """True if every task is completed or failed."""
        return all(
            t.status in ("completed", "failed") for t in self.nodes.values()
        )

    def summary(self) -> str:
        """Human-readable summary of the DAG state."""
        lines = ["=== Task DAG ==="]
        for tid in self._execution_order:
            t = self.nodes[tid]
            deps = ", ".join(t.dependencies) if t.dependencies else "none"
            status_icon = {
                "pending": "⏳", "running": "🔄",
                "completed": "✅", "failed": "❌"
            }.get(t.status, "?")
            lines.append(
                f"  {status_icon} [{t.id}] {t.description} "
                f"(deps: {deps}, method: {t.planning_method})"
            )
            if t.result:
                preview = t.result[:100] + "..." if len(t.result) > 100 else t.result
                lines.append(f"      → {preview}")
            if t.error:
                lines.append(f"      ✖ {t.error}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the DAG to a dictionary."""
        return {
            "execution_order": self._execution_order,
            "tasks": {tid: t.to_dict() for tid, t in self.nodes.items()},
        }
