"""
Nexlink NOC Planning Agent — Decomposition & Planning Enhanced Client (v3.0)

A NEW agent (separate from memory_rag_client.py) that decomposes complex
multi-step NOC requests into a DAG of sub-tasks, plans each using the
appropriate algorithm, and self-corrects before delivering results.

Reuses the SAME mcp_server/ and db/ as the existing agents.
"""

import asyncio
import json
import os
import sys
import time
import uuid

from langchain_ollama import ChatOllama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add project root to path
PROJECT_ROOT = str(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, PROJECT_ROOT)

from planning.dag import TaskDAG, SubTask
from planning.metrics import PlanningMetrics, timed_llm_call
from planning.executor import DAGExecutor
from planning.decompose_first import decompose_upfront
from planning.decompose_dynamic import DynamicDecomposer


# ── Banner ───────────────────────────────────────────────────────────────

BANNER = """
======================================================================
🧠 NEXLINK NOC PLANNING AGENT — Decomposition & Planning (v3.0)
======================================================================
"""

HELP_TEXT = """
Commands:
  /plan <request>     Decompose-first: build full DAG upfront, then execute
  /dynamic <request>  Dynamic: generate one step at a time, adapt after each
  /compare <request>  Run BOTH methods and show comparison metrics
  /tools              List available MCP tools
  /help               Show this help message
  exit                Quit the agent

Or just type a natural-language request (defaults to /plan mode).
"""


def get_tool_descriptions(tools) -> list[dict]:
    """Convert MCP tool list into simplified dicts for LLM prompts."""
    descriptions = []
    for tool in tools:
        desc = {
            "name": tool.name,
            "description": tool.description or "",
        }
        if tool.inputSchema and "properties" in tool.inputSchema:
            params = {}
            for pname, pinfo in tool.inputSchema["properties"].items():
                params[pname] = pinfo.get("type", "string")
            desc["parameters"] = params
        descriptions.append(desc)
    return descriptions


async def run_upfront(request, llm, session, tools_desc, metrics):
    """Run decomposition-first mode."""
    print("\n📐 [Decompose-First] Building full DAG upfront...")
    dag = decompose_upfront(request, tools_desc, llm, metrics)

    print(f"\n{dag.summary()}")
    print(f"\n▶ Executing {len(dag.nodes)} tasks in topological order...\n")

    executor = DAGExecutor(llm, session)
    result = await executor.execute(dag, metrics)
    return result, dag


async def run_dynamic(request, llm, session, tools_desc, metrics):
    """Run dynamic/interleaved decomposition mode."""
    print("\n🔄 [Dynamic Decomposition] Generating steps one at a time...\n")

    executor = DAGExecutor(llm, session)
    result = await executor.execute_dynamic(request, tools_desc, metrics)
    return result


def print_results(result, label="Results"):
    """Pretty-print execution results."""
    print(f"\n{'='*60}")
    print(f"📋 {label}")
    print(f"{'='*60}")

    if result.get("dag_summary"):
        print(result["dag_summary"])

    print(f"\n📊 Metrics: {result['metrics']}")

    if result.get("failed"):
        print(f"\n❌ Failed tasks:")
        for tid, err in result["failed"].items():
            print(f"   {tid}: {err}")

    # Generate final synthesis
    if result.get("results"):
        print(f"\n✅ All task results collected ({len(result['results'])} tasks completed)")

    print(f"{'='*60}\n")


async def synthesize_answer(request, results, llm, metrics):
    """Use the LLM to synthesize a final user-facing answer from all task results."""
    results_text = "\n\n".join(
        f"[{tid}]: {res}" for tid, res in results.items()
    )
    prompt = (
        f"You are a Nexlink Telecom NOC assistant. A user asked:\n"
        f"\"{request}\"\n\n"
        f"The following sub-task results were collected:\n{results_text}\n\n"
        f"Synthesize a clear, complete answer addressing the original request. "
        f"Include all relevant data from the sub-tasks. Be specific with numbers and statuses."
    )
    return timed_llm_call(llm, prompt, metrics)


async def run_client():
    """Main client loop."""
    print(BANNER)
    session_id = uuid.uuid4().hex[:8]
    print(f"📝 Session ID: {session_id}")

    # Initialize LLM
    llm = ChatOllama(model="qwen-accurate:9b", temperature=0)
    print(f"🤖 LLM: qwen-accurate:9b (Ollama)")

    # Connect to MCP server
    server_script = str(os.path.join(PROJECT_ROOT, "mcp_server", "server.py"))
    python_path = sys.executable

    server_params = StdioServerParameters(
        command=python_path,
        args=[server_script, "--transport", "stdio"],
    )

    print("🔌 Connecting to MCP Server over Stdio...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ [HANDSHAKE SUCCESS] MCP Session Initialized.\n")

            # Get available tools
            tools_res = await session.list_tools()
            tools_desc = get_tool_descriptions(tools_res.tools)
            print(f"🔧 {len(tools_desc)} MCP tools available")

            # Auto-authenticate as admin for planning operations
            auth_result = await session.call_tool(
                "authenticate_user", {"api_token": "token-admin-9988"}
            )
            if auth_result and auth_result.content:
                auth_text = getattr(auth_result.content[0], "text", "")
                print(f"🔒 {auth_text}")

            print(f"\n{HELP_TEXT}")

            # Interactive loop
            while True:
                try:
                    user_input = input("Planner: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "/help":
                    print(HELP_TEXT)
                    continue

                if user_input.lower() == "/tools":
                    print("\n🔧 Available MCP Tools:")
                    for t in tools_desc:
                        params = t.get("parameters", {})
                        param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
                        print(f"  • {t['name']}({param_str})")
                        if t.get("description"):
                            print(f"    {t['description'][:80]}")
                    print()
                    continue

                # Parse command
                mode = "plan"  # default
                request = user_input

                if user_input.startswith("/plan "):
                    mode = "plan"
                    request = user_input[6:].strip()
                elif user_input.startswith("/dynamic "):
                    mode = "dynamic"
                    request = user_input[9:].strip()
                elif user_input.startswith("/compare "):
                    mode = "compare"
                    request = user_input[9:].strip()

                if not request:
                    print("Please provide a request after the command.")
                    continue

                try:
                    if mode == "plan":
                        metrics = PlanningMetrics()
                        start = time.time()
                        result, dag = await run_upfront(
                            request, llm, session, tools_desc, metrics
                        )
                        elapsed = time.time() - start
                        print_results(result, "Decompose-First Results")

                        # Synthesize final answer
                        if result.get("results"):
                            print("🧠 Synthesizing final answer...\n")
                            answer = await synthesize_answer(
                                request, result["results"], llm, metrics
                            )
                            print(f"🤖 Planning Agent:\n{answer}\n")
                            print(f"⏱ Total time: {elapsed:.1f}s | {metrics.summary()}")

                    elif mode == "dynamic":
                        metrics = PlanningMetrics()
                        start = time.time()
                        result = await run_dynamic(
                            request, llm, session, tools_desc, metrics
                        )
                        elapsed = time.time() - start
                        print_results(result, "Dynamic Decomposition Results")

                        if result.get("results"):
                            print("🧠 Synthesizing final answer...\n")
                            answer = await synthesize_answer(
                                request, result["results"], llm, metrics
                            )
                            print(f"🤖 Planning Agent:\n{answer}\n")
                            print(f"⏱ Total time: {elapsed:.1f}s | {metrics.summary()}")

                    elif mode == "compare":
                        print("\n" + "="*60)
                        print("📊 COMPARISON MODE: Running both decomposition methods")
                        print("="*60)

                        # Run decompose-first
                        metrics_first = PlanningMetrics()
                        start1 = time.time()
                        result_first, _ = await run_upfront(
                            request, llm, session, tools_desc, metrics_first
                        )
                        elapsed1 = time.time() - start1
                        print_results(result_first, "Decompose-First Results")

                        # Run dynamic
                        metrics_dyn = PlanningMetrics()
                        start2 = time.time()
                        result_dyn = await run_dynamic(
                            request, llm, session, tools_desc, metrics_dyn
                        )
                        elapsed2 = time.time() - start2
                        print_results(result_dyn, "Dynamic Decomposition Results")

                        # Comparison table
                        print("\n" + "="*60)
                        print("📊 COMPARISON TABLE")
                        print("="*60)
                        print(f"{'Metric':<25} {'Decompose-First':>18} {'Dynamic':>18}")
                        print("-"*61)
                        print(f"{'LLM Calls':<25} {metrics_first.llm_calls:>18} {metrics_dyn.llm_calls:>18}")
                        print(f"{'Total Tokens':<25} {metrics_first.total_tokens:>18} {metrics_dyn.total_tokens:>18}")
                        print(f"{'  Input Tokens':<25} {metrics_first.input_tokens:>18} {metrics_dyn.input_tokens:>18}")
                        print(f"{'  Output Tokens':<25} {metrics_first.output_tokens:>18} {metrics_dyn.output_tokens:>18}")
                        print(f"{'Tool Calls':<25} {metrics_first.tool_calls:>18} {metrics_dyn.tool_calls:>18}")
                        print(f"{'Latency (s)':<25} {elapsed1:>18.1f} {elapsed2:>18.1f}")
                        t1_ok = len(result_first.get('results', {}))
                        t2_ok = len(result_dyn.get('results', {}))
                        t1_fail = len(result_first.get('failed', {}))
                        t2_fail = len(result_dyn.get('failed', {}))
                        print(f"{'Tasks Completed':<25} {t1_ok:>18} {t2_ok:>18}")
                        print(f"{'Tasks Failed':<25} {t1_fail:>18} {t2_fail:>18}")
                        print("="*61 + "\n")

                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_client())
