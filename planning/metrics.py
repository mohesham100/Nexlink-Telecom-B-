"""
Metrics tracking for planning operations.

Accumulates LLM calls, token usage, tool calls, and latency
across all sub-tasks in a DAG execution for comparison tables.
"""

from dataclasses import dataclass, field
import time
from typing import Optional


@dataclass
class PlanningMetrics:
    """Tracks cost and performance metrics for a planning run."""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_s: float = 0.0
    tool_calls: int = 0

    def record_llm_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency: float = 0.0,
    ) -> None:
        """Record a single LLM call's costs."""
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_latency_s += latency

    def record_tool_call(self, latency: float = 0.0) -> None:
        """Record a single MCP tool call."""
        self.tool_calls += 1
        self.total_latency_s += latency

    def merge(self, other: "PlanningMetrics") -> None:
        """Merge another metrics object into this one."""
        self.llm_calls += other.llm_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_latency_s += other.total_latency_s
        self.tool_calls += other.tool_calls

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency_s, 2),
            "tool_calls": self.tool_calls,
        }

    def summary(self) -> str:
        return (
            f"LLM calls: {self.llm_calls} | "
            f"Tokens: {self.total_tokens} (in={self.input_tokens}, out={self.output_tokens}) | "
            f"Tool calls: {self.tool_calls} | "
            f"Latency: {self.total_latency_s:.1f}s"
        )


def timed_llm_call(llm, prompt: str, metrics: PlanningMetrics) -> str:
    """
    Call the LLM with timing and metrics tracking.
    Returns the response text content.
    """
    start = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - start

    content = str(response.content) if hasattr(response, "content") else str(response)

    # Estimate token counts (rough: 1 token ≈ 4 chars)
    est_input = len(prompt) // 4
    est_output = len(content) // 4

    metrics.record_llm_call(
        input_tokens=est_input,
        output_tokens=est_output,
        latency=elapsed,
    )
    return content


async def timed_tool_call(session, tool_name: str, args: dict, metrics: PlanningMetrics) -> str:
    """
    Call an MCP tool with timing and metrics tracking.
    Returns the result text.
    """
    start = time.time()
    result = await session.call_tool(tool_name, args)
    elapsed = time.time() - start
    metrics.record_tool_call(latency=elapsed)

    # Extract text from MCP result
    if result is not None and hasattr(result, "content") and result.content:
        text_parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else "No result content."
    elif isinstance(result, str):
        return result
    elif result is not None:
        return str(result)
    return "No result."
