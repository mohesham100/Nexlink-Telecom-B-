"""
Nexlink Telecom NOC Agent — Memory & RAG Enhanced Client
=========================================================
Extends the original MCP agent with:
  - Short-term memory buffer + scratchpad
  - Episodic and semantic memory stores
  - Promote-or-drop routing on buffer overflow
  - Periodic semantic consolidation
  - Hybrid RAG with Self-RAG verification
  - MCP tool execution with human-in-the-loop security
"""

import asyncio
import json
import re
import sys
import os
import uuid

# --- Path setup so imports work from any working directory ---
PROJECT_ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_ollama import ChatOllama

# Memory system imports
from memory.short_term import ShortTermMemory, Scratchpad
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.router import MemoryRouter
from memory.consolidation import ConsolidationLayer

# RAG system imports
from rag.hybrid_rag import hybrid_rag_query
from rag.self_rag import verified_rag_query

# --- Constants ---
MEMORY_DB = os.path.join(PROJECT_ROOT, 'db', 'memory.db')
ROUTING_LOG = os.path.join(PROJECT_ROOT, 'memory', 'routing_log.jsonl')
MODEL_NAME = "qwen-accurate:9b"
SESSION_ID = uuid.uuid4().hex[:8]


class NOCAgentWithMemory:
    """
    Enhanced NOC Agent Engine with memory context and RAG grounding.
    """

    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)

        # ========== MEMORY COMPONENTS ==========
        # Short-term buffer (rolling window of recent interactions)
        self.stm = ShortTermMemory(max_size=20)
        # Scratchpad (NEVER pruned by buffer overflow — holds working state)
        self.scratchpad = Scratchpad()
        # Episodic memory (SQLite-backed, stores individual events)
        self.episodic = EpisodicStore(db_path=MEMORY_DB)
        # Semantic memory (SQLite-backed, stores consolidated facts)
        self.semantic = SemanticStore(db_path=MEMORY_DB)
        # Promote-or-drop router (forget vs. episodic routing)
        self.router = MemoryRouter(
            episodic_store=self.episodic,
            routing_log_path=ROUTING_LOG
        )
        # Consolidation layer (periodic episodic → semantic pass)
        self.consolidation = ConsolidationLayer(
            episodic_store=self.episodic,
            semantic_store=self.semantic,
            llm_model=MODEL_NAME
        )
        self.session_id = SESSION_ID

    def build_memory_context(self, user_input: str) -> str:
        """Query episodic + semantic memory for relevant context."""
        parts = []

        # Search episodic memory by text
        episodic_results = self.episodic.search_text(user_input, limit=3)
        if episodic_results:
            parts.append("--- Episodic Memory (Past Events) ---")
            for ep in episodic_results:
                parts.append(f"  [{ep['timestamp']}] {ep['content']}")

        # Search semantic memory by text
        semantic_results = self.semantic.search_facts(user_input)
        if semantic_results:
            parts.append("--- Semantic Memory (Known Facts) ---")
            for fact in semantic_results:
                status = f" [v{fact['version']}, {fact['status']}]"
                parts.append(f"  • {fact['fact']}{status}")

        return "\n".join(parts) if parts else "No relevant memories found."

    def handle_overflow(self):
        """Check buffer overflow and route items via promote-or-drop."""
        if self.stm.is_overflowing():
            overflow_count = 5  # Route 5 oldest items at a time
            overflow_items = self.stm.get_overflow_items(overflow_count)
            result = self.router.route(overflow_items, session_id=self.session_id)
            self.stm.remove_oldest(overflow_count)

            promoted = len(result.get("promoted", []))
            forgotten = len(result.get("forgotten", []))
            if promoted > 0 or forgotten > 0:
                print(f"   📦 Buffer overflow: {promoted} items promoted to episodic, {forgotten} forgotten")

    def get_ai_decision(self, user_input: str, available_tools, memory_context: str, rag_context: str = ""):
        """Build enriched prompt with memory + RAG context and get AI decision."""
        tools_info = []
        for t in available_tools:
            tools_info.append(f"- {t.name}: {t.description}\n  Schema: {t.inputSchema}")
        tools_text = "\n".join(tools_info)

        scratchpad_text = self.scratchpad.to_context_string()
        recent_buffer = self.stm.get_recent(6)
        recent_text = "\n".join(
            [f"  {m['role']}: {m['content'][:200]}" for m in recent_buffer]
        ) if recent_buffer else "No recent context."

        prompt = f"""You are an autonomous NOC AI Assistant for Nexlink Telecom.

=== SCRATCHPAD (Active Working State) ===
{scratchpad_text}

=== MEMORY CONTEXT ===
{memory_context}

=== KNOWLEDGE BASE (RAG) ===
{rag_context if rag_context else "No RAG context retrieved."}

=== RECENT CONVERSATION ===
{recent_text}

=== AVAILABLE TOOLS ===
{tools_text}

User Request: "{user_input}"

INSTRUCTIONS (Choose OPTION A or OPTION B):
OPTION A - If you need to use a tool to get data, output ONLY a JSON object:
{{"tool_name": "name_of_tool", "arguments": {{"key": value}}}}

OPTION B - If you can answer from memory/RAG/context, output ONLY:
{{"response": "Your message here"}}

STRICT RULES:
- NEVER write code blocks. NEVER add text outside the JSON.
- node_id and customer_id must be integers.
- Use memory context and RAG knowledge to inform your answer when available.
"""
        res = self.llm.invoke(prompt)
        return str(res.content)


async def run_client():
    print("=" * 70)
    print("🤖 NEXLINK NOC AGENT — Memory & RAG Enhanced (v2.0)")
    print("=" * 70)
    print(f"📝 Session ID: {SESSION_ID}")

    server_params = StdioServerParameters(
        command="python",
        args=[str(os.path.join(PROJECT_ROOT, "mcp_server", "server.py")), "--transport", "stdio"]
    )

    engine = NOCAgentWithMemory()
    print(f"🧠 Memory initialized: STM buffer={engine.stm.max_size}, DB={MEMORY_DB}")
    print(f"📂 Routing log: {ROUTING_LOG}")

    print("🔌 Connecting to MCP Server over Stdio...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ [HANDSHAKE SUCCESS] MCP Session Initialized.\n")

            tools_res = await session.list_tools()

            print("NOC Agent Ready. Type your query naturally.")
            print("Special commands: /auth, /resources, /memory, /consolidate, /scratchpad, /search, /help\n")

            interaction_count = 0

            while True:
                try:
                    user_input = input("Engineer: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input or user_input.lower() in ['exit', 'quit']:
                    print("🔄 Running final consolidation pass...")
                    summary = engine.consolidation.run_consolidation()
                    print(f"   ✅ Final consolidation: {summary}")
                    print("Goodbye!")
                    break

                # ============ SPECIAL COMMANDS ============
                elif user_input.startswith("/auth "):
                    token = user_input.split("/auth ")[1].strip()
                    print("🔒 Processing Authentication...")
                    t_res = await session.call_tool("authenticate_user", arguments={"api_token": token})
                    auth_msg = getattr(t_res.content[0], 'text', str(t_res.content[0])) if t_res.content else ""
                    print(f"✅ [Server]: {auth_msg}\n")
                    engine.stm.add("system", f"Authenticated: {auth_msg}", "system")
                    tools_res = await session.list_tools()
                    continue

                elif user_input == "/resources":
                    print("\n📚 [MCP RESOURCES] Reading policy files...")
                    r_list = await session.list_resources()
                    for r in r_list.resources:
                        c = await session.read_resource(r.uri)
                        res_text = getattr(c.contents[0], 'text', str(c.contents[0])) if c.contents else ""
                        print(f"📄 {r.name}:\n{res_text[:150]}...\n")
                    continue

                elif user_input == "/memory":
                    print("\n🧠 === MEMORY STATE ===")
                    recent_eps = engine.episodic.query_recent(limit=5)
                    if recent_eps:
                        print("--- Recent Episodic Memories ---")
                        for ep in recent_eps:
                            print(f"  [{ep['timestamp']}] ({ep['event_type']}) {ep['content'][:100]}")
                    else:
                        print("  No episodic memories yet.")

                    all_facts = engine.semantic.search_facts("")
                    if all_facts:
                        print("--- Semantic Facts ---")
                        for f in all_facts:
                            print(f"  [v{f['version']}|{f['status']}] {f['fact'][:100]}")
                            if f.get('conflict_resolution_note'):
                                print(f"    ⚠️ Conflict note: {f['conflict_resolution_note']}")
                    else:
                        print("  No semantic facts yet.")
                    print()
                    continue

                elif user_input == "/consolidate":
                    print("🔄 Running manual consolidation pass...")
                    summary = engine.consolidation.run_consolidation()
                    print(f"   Episodes processed: {summary['episodes_processed']}")
                    print(f"   New facts added: {summary['new_facts_added']}")
                    print(f"   Conflicts resolved: {summary['conflicts_resolved']}")
                    print("✅ Consolidation complete.\n")
                    continue

                elif user_input == "/scratchpad":
                    print(f"\n📋 === SCRATCHPAD ===\n{engine.scratchpad.to_context_string()}\n")
                    continue

                elif user_input.startswith("/search "):
                    query = user_input[8:].strip()
                    print(f"🔍 Searching knowledge base: '{query}'...")
                    result = verified_rag_query(query, hybrid_rag_query)
                    print(f"📄 Answer: {result['answer']}")
                    print(f"🔎 Verification: {result.get('verification', {})}")
                    print(f"📊 Chunks used: {len(result.get('retrieved_chunks', []))}\n")
                    continue

                elif user_input == "/help":
                    print("\nCommands:")
                    print("  /auth <token>  — Authenticate with API token")
                    print("  /resources     — List MCP policy documents")
                    print("  /memory        — Show episodic + semantic memory")
                    print("  /consolidate   — Trigger manual consolidation pass")
                    print("  /scratchpad    — Show current scratchpad state")
                    print("  /search <q>    — Direct RAG knowledge base search")
                    print("  /help          — Show this help")
                    print("  exit/quit      — Exit agent\n")
                    continue

                # ============ MAIN AGENT LOOP ============
                # 1. Add to short-term buffer
                engine.stm.add("user", user_input, "user_input")

                # 2. Handle buffer overflow
                engine.handle_overflow()

                # 3. Query memory for context
                memory_context = engine.build_memory_context(user_input)

                # 4. Optionally query RAG for knowledge-heavy questions
                rag_context = ""
                knowledge_keywords = ["what", "how", "why", "procedure", "protocol", "post-mortem",
                                      "maintenance", "policy", "sla", "escalation", "history", "bulletin"]
                if any(kw in user_input.lower() for kw in knowledge_keywords):
                    try:
                        rag_result = verified_rag_query(user_input, hybrid_rag_query)
                        rag_context = rag_result.get("answer", "")
                        verification = rag_result.get("verification", {})
                        support_status = verification.get("support_status", "UNKNOWN")
                        if support_status != "SUPPORTED":
                            rag_context = f"[⚠️ Verification: {support_status}] {rag_context}"
                    except Exception as e:
                        rag_context = f"RAG search failed: {e}"

                # 5. Get AI decision
                print("🧠 AI is thinking...")
                ai_response = engine.get_ai_decision(user_input, tools_res.tools, memory_context, rag_context)

                # 6. Parse JSON response
                json_match = re.search(r'\{.*}', ai_response, re.DOTALL)

                if json_match:
                    try:
                        decision = json.loads(json_match.group())

                        if "tool_name" in decision:
                            t_name = decision["tool_name"]
                            t_args = decision.get("arguments", {})
                            print(f"⚙️ [AI Action]: Using tool '{t_name}' with args {t_args}")

                            # Security check for sensitive tools
                            sensitive_tools = ["upgrade_bandwidth"]
                            approved = True
                            if t_name in sensitive_tools:
                                print(f"\n⚠️ [SECURITY ALERT]: AI requested sensitive operation ({t_name}).")
                                print(f"   Requested Parameters: {t_args}")
                                approval = input("   👉 Do you approve? (yes/no): ").strip().lower()
                                if approval not in ['yes', 'y']:
                                    approved = False

                            if approved:
                                # Progress tracking for diagnostics
                                if t_name == "run_network_diagnostic":
                                    print("   ⏳ Running deep network diagnostic...")
                                    for i in range(1, 5):
                                        print(f"   📊 [MCP PROGRESS] Phase {i}/4 ({i * 25}%)")
                                        await asyncio.sleep(0.2)

                                try:
                                    t_res = await session.call_tool(t_name, arguments=t_args)
                                    if t_res.content:
                                        out_text = getattr(t_res.content[0], 'text', str(t_res.content[0]))
                                    else:
                                        out_text = ""
                                except Exception as e:
                                    out_text = f"Error: {str(e)}"
                            else:
                                out_text = "Operation cancelled: Human engineer denied approval."
                                print("❌ [Action Denied by Admin]")

                            print(f"✅ [Server Response]: {out_text}")

                            # Save tool result to short-term memory
                            engine.stm.add("system", out_text, "tool_result")

                            # Update scratchpad if relevant
                            if "diagnostic" in t_name.lower():
                                engine.scratchpad.set_working_state("last_diagnostic", out_text[:200])
                            if "upgrade" in t_name.lower() and approved:
                                engine.scratchpad.set_working_state("last_upgrade", out_text[:200])

                            # Summarize for user
                            print("🧠 AI is summarizing...")
                            summary_prompt = f"The tool returned: {out_text}. Summarize this result concisely."
                            summary_res = engine.llm.invoke(summary_prompt)
                            print(f"\n🤖 AI Assistant:\n {summary_res.content}\n")
                            engine.stm.add("assistant", str(summary_res.content), "ai_response")

                        elif "response" in decision:
                            print(f"\n🤖 AI Assistant:\n {decision['response']}\n")
                            engine.stm.add("assistant", decision['response'], "ai_response")

                    except json.JSONDecodeError:
                        print(f"\n🤖 AI Assistant:\n {ai_response}\n")
                        engine.stm.add("assistant", ai_response, "ai_response")
                else:
                    print(f"\n🤖 AI Assistant:\n {ai_response}\n")
                    engine.stm.add("assistant", ai_response, "ai_response")

                # Periodic consolidation every 15 interactions
                interaction_count += 1
                if interaction_count % 15 == 0:
                    print("🔄 Running periodic consolidation...")
                    summary = engine.consolidation.run_consolidation()
                    if summary['episodes_processed'] > 0:
                        print(f"   ✅ Consolidated: {summary}")


if __name__ == "__main__":
    asyncio.run(run_client())
