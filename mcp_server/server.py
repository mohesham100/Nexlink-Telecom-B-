import asyncio
import os
import sqlite3
import argparse
from typing import Dict, Any
from dataclasses import dataclass
from fastmcp import FastMCP, Context
from mcp.types import PromptMessage, TextContent
from langchain_ollama import ChatOllama

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mcp_server.rag.rag_tool import search_knowledge_base_handler, knowledge_store

mcp = FastMCP("NexlinkTelecomNOC")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'nexlink.db')
POLICIES_DIR = os.path.join(os.path.dirname(__file__), 'policies')

# Real LLM (TODO #1 from decompose_search.py: replace FakeLLM with real LLM client)
decompose_llm = ChatOllama(model="qwen-accurate:9b", temperature=0)

active_session: Dict[str, Any] = {
    "authenticated": False,
    "user_id": None,
    "username": "guest",
    "role": "Guest",
    "token": None
}

@mcp.resource("file://policies/sla_policy.txt")
def get_sla_policy() -> str:
    path = os.path.join(POLICIES_DIR, 'sla_policy.txt')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Policy file not found."

@mcp.resource("file://policies/network_runbook.txt")
def get_network_runbook() -> str:
    path = os.path.join(POLICIES_DIR, 'network_runbook.txt')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Runbook not found."

@mcp.prompt("draft_incident_report")
def draft_incident_report(incident_id: int, node_id: int) -> str:
    return f"Draft report for Incident #{incident_id} on Node #{node_id} including summary, impact, and actions."

@mcp.tool()
async def authenticate_user(api_token: str, ctx: Context) -> str:
    """Authenticate user with API token."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username, role FROM users WHERE api_token = ?", (api_token,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Authentication failed: invalid token."

    old_role = active_session["role"]
    active_session["authenticated"] = True
    active_session["user_id"], active_session["username"], active_session["role"] = row

    if old_role != active_session["role"] and ctx and ctx.session:
        try:
            await ctx.session.send_tool_list_changed()
        except Exception:
            pass

    return f"Authenticated as {active_session['username']} ({active_session['role']})."

@mcp.tool()
def get_customer_network_status(customer_id: int) -> str:
    """Get customer network status and bandwidth."""
    if customer_id <= 0:
        return "Validation Error: customer_id must be a positive integer."

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.name, c.sla_tier, n.id, n.name, n.status, s.allocated_bandwidth_gbps 
        FROM customers c
        JOIN services s ON c.id = s.customer_id
        JOIN network_nodes n ON s.node_id = n.id
        WHERE c.id = ?
    """, (customer_id,))
    res = cur.fetchone()
    conn.close()

    if not res:
        return f"Customer {customer_id} not found."
    return f"Customer: {res[0]}, SLA: {res[1]}, Node #{res[2]} ({res[3]}), Status: {res[4]}, Bandwidth: {res[5]} Gbps"

@mcp.tool()
async def run_network_diagnostic(node_id: int, ctx: Context) -> str:
    """Run diagnostic on a node."""
    if node_id <= 0:
        return "Validation Error: node_id must be positive."

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, status, current_load_gbps, max_capacity_gbps FROM network_nodes WHERE id = ?", (node_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return f"Node {node_id} not found."

    for step in range(1, 5):
        if ctx:
            await ctx.report_progress(step, 4)
        await asyncio.sleep(0.2)

    name, status, load, capacity = row
    pct = round((load / capacity) * 100, 1)
    return f"Diagnostic Node #{node_id} ({name}): Status={status}, Load={load}/{capacity} Gbps ({pct}%)."

@mcp.tool()
async def upgrade_bandwidth(customer_id: int, new_bandwidth_gbps: float, ctx: Context) -> str:
    """Upgrade customer bandwidth."""
    if customer_id <= 0 or new_bandwidth_gbps <= 0:
        return "Validation Error: ID and bandwidth must be positive numbers."

    if active_session["role"] not in ["NOC_Admin", "NOC_Engineer"]:
        return "Auth error: NOC_Admin role required."

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT c.name, c.sla_tier, s.allocated_bandwidth_gbps FROM customers c JOIN services s ON c.id = s.customer_id WHERE c.id = ?", (customer_id,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return f"Customer {customer_id} not found."

    cust_name, sla_tier, old_bw = row
    needs_approval = (new_bandwidth_gbps > 3.0 or sla_tier == "VIP")

    if needs_approval and ctx and ctx.session:
        msg = f"Approve upgrade for {cust_name} ({sla_tier}) from {old_bw} to {new_bandwidth_gbps} Gbps? (yes/no)"
        try:
            res = await ctx.elicit(message=msg)
            if not res or res.lower().strip() not in ["yes", "y"]:
                conn.close()
                return "Upgrade denied by operator."
        except Exception:
            pass

    cur.execute("UPDATE services SET allocated_bandwidth_gbps = ? WHERE customer_id = ?", (new_bandwidth_gbps, customer_id))
    cur.execute("INSERT INTO audit_logs (user_id, action, details) VALUES (?, ?, ?)",
                (active_session["user_id"], "UPGRADE_BANDWIDTH", f"Customer {customer_id} set to {new_bandwidth_gbps} Gbps"))
    conn.commit()
    conn.close()
    return f"Upgraded {cust_name} bandwidth from {old_bw} to {new_bandwidth_gbps} Gbps."

@mcp.tool()
async def analyze_incident_root_cause(node_id: int, ctx: Context) -> str:
    """Analyze incident root cause for node."""
    if node_id <= 0:
        return "Validation Error: node_id must be positive."

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, type, status, current_load_gbps FROM network_nodes WHERE id = ?", (node_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return f"Node {node_id} not found."

    analysis = "Optical signal attenuation detected. Physical fiber inspection recommended."
    if ctx and ctx.session:
        try:
            trace = f"Node {row[0]} ({row[1]}) status {row[2]}, load {row[3]} Gbps."
            res = await ctx.session.create_message(
                messages=[PromptMessage(role="user", content=TextContent(type="text", text=f"Analyze root cause for: {trace}"))],
                max_tokens=100
            )
            if res and res.content:
                analysis = res.content.text.strip()
        except Exception:
            pass

    return f"Root Cause Analysis for Node #{node_id}: {analysis}"

DECOMPOSE_PROMPT = """\
Break the following question into 2-4 simpler sub-questions that, together,
fully answer it. If the question is already simple, just return it as-is
as a single sub-question.

Question: {query}

Return ONLY a numbered list, one sub-question per line. Example:
1. ...
2. ...
"""


def decompose_query(query: str) -> list[str]:
    """Turn one (possibly compound) query into a list of sub-questions."""
    raw = str(decompose_llm.invoke(DECOMPOSE_PROMPT.format(query=query)).content)

    sub_questions = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # strip a leading "1.", "2)", "- " etc.
        for sep in [". ", ") ", "- "]:
            if sep in line[:4]:
                line = line.split(sep, 1)[1]
                break
        sub_questions.append(line.strip())

    return sub_questions or [query]  # fallback: treat as one question


# Tagged result so the model knows which sub-question each chunk answers


@dataclass
class TaggedChunk:
    sub_question: str
    chunk: str
    score: float


@mcp.tool()
def decompose_and_search(query: str, top_k: int = 3) -> str:
    """
    Break a compound question into sub-questions, search the knowledge base
    once per sub-question, and return combined tagged results.
    Sits in front of search_knowledge_base — does not replace it.
    """
    sub_questions = decompose_query(query)

    results: list[TaggedChunk] = []
    for sub_q in sub_questions:
        # TODO #2 from decompose_search.py: use real search tool
        hits = knowledge_store.query(query_text=sub_q, top_k=top_k)
        for doc in hits:
            if doc["metadata"]["role_required"] in ("any", active_session["role"]):
                results.append(TaggedChunk(sub_question=sub_q, chunk=doc["payload"], score=1.0))

    if not results:
        return "No relevant records found for any sub-question."

    output_lines = []
    for r in results:
        output_lines.append(f"[Sub-Q: {r.sub_question}] -> {r.chunk} (score={r.score})")
    return "\n".join(output_lines)


@mcp.tool()
def search_knowledge_base(query: str, entity_id: str = None, top_k: int = 3) -> str:
    """Search unstructured NOC incident post-mortems, runbooks, and maintenance notes."""
    args = {"query": query, "top_k": top_k}
    if entity_id:
        args["entity_id"] = entity_id
    return search_knowledge_base_handler(args, session_role=active_session["role"])


# ============================================================
# Planning Lab: additional tools for multi-step planning
# ============================================================

@mcp.tool()
def list_all_nodes() -> str:
    """List all network nodes with their status, type, load, and capacity."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, type, status, current_load_gbps, max_capacity_gbps, location "
        "FROM network_nodes ORDER BY id"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "No network nodes found."

    lines = []
    for r in rows:
        pct = round((r[4] / r[5]) * 100, 1) if r[5] > 0 else 0.0
        lines.append(
            f"Node #{r[0]} ({r[1]}): type={r[2]}, status={r[3]}, "
            f"load={r[4]}/{r[5]} Gbps ({pct}%), location={r[6]}"
        )
    return "\n".join(lines)


@mcp.tool()
def list_all_customers() -> str:
    """List all customers with their SLA tier, assigned node, and service status."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id, c.name, c.industry, c.sla_tier,
               s.node_id, s.allocated_bandwidth_gbps, s.status,
               n.name, n.status
        FROM customers c
        LEFT JOIN services s ON c.id = s.customer_id
        LEFT JOIN network_nodes n ON s.node_id = n.id
        ORDER BY c.id
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "No customers found."

    lines = []
    for r in rows:
        lines.append(
            f"Customer #{r[0]} ({r[1]}): industry={r[2]}, SLA={r[3]}, "
            f"node=#{r[4]} ({r[7]}, status={r[8]}), "
            f"bandwidth={r[5]} Gbps, service_status={r[6]}"
        )
    return "\n".join(lines)


@mcp.tool()
def get_node_customers(node_id: int) -> str:
    """Get all customers assigned to a specific network node."""
    if node_id <= 0:
        return "Validation Error: node_id must be positive."

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id, c.name, c.sla_tier, c.industry,
               s.allocated_bandwidth_gbps, s.status
        FROM customers c
        JOIN services s ON c.id = s.customer_id
        WHERE s.node_id = ?
        ORDER BY c.sla_tier, c.id
    """, (node_id,))
    rows = cur.fetchall()

    cur.execute("SELECT name, status FROM network_nodes WHERE id = ?", (node_id,))
    node_row = cur.fetchone()
    conn.close()

    if not node_row:
        return f"Node {node_id} not found."

    header = f"Node #{node_id} ({node_row[0]}, status={node_row[1]}) — {len(rows)} customer(s):"
    if not rows:
        return header + "\n  No customers assigned to this node."

    lines = [header]
    for r in rows:
        lines.append(
            f"  Customer #{r[0]} ({r[1]}): SLA={r[2]}, industry={r[3]}, "
            f"bandwidth={r[4]} Gbps, service={r[5]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse")