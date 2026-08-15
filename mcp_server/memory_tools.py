"""
MCP Server Extension: Memory & RAG Tools
=========================================
Provides standalone tool handler functions for memory recall and RAG search.
These can be registered on the MCP server or called directly by the agent.

Does NOT modify the existing server.py — this is a separate module.
"""

import sys
import os
import json

PROJECT_ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from rag.hybrid_rag import hybrid_rag_query
from rag.self_rag import verified_rag_query

MEMORY_DB = os.path.join(PROJECT_ROOT, 'db', 'memory.db')


def recall_memory(query: str, entity_type: str = None, entity_id: str = None) -> str:
    """
    Queries both episodic and semantic memory stores, returns formatted results.
    Can filter by entity_type ('node' or 'customer') and entity_id.
    """
    episodic = EpisodicStore(db_path=MEMORY_DB)
    semantic = SemanticStore(db_path=MEMORY_DB)

    response_parts = ["🧠 Memory Recall Results:\n"]

    # --- Episodic Memory ---
    if entity_type and entity_id:
        ep_results = episodic.query_by_entity(entity_type, entity_id, limit=5)
    else:
        ep_results = episodic.search_text(query, limit=5)

    response_parts.append("--- Episodic Experiences ---")
    if ep_results:
        for ep in ep_results:
            response_parts.append(f"  [{ep['timestamp']}] ({ep['event_type']}) {ep['content'][:150]}")
    else:
        response_parts.append("  No episodic memories found.")

    # --- Semantic Memory ---
    if entity_id:
        sem_results = semantic.query_facts(entity_id=entity_id)
    else:
        sem_results = semantic.search_facts(query)

    response_parts.append("\n--- Semantic Facts ---")
    if sem_results:
        for fact in sem_results:
            status_tag = f"[v{fact['version']}|{fact['status']}]"
            response_parts.append(f"  {status_tag} {fact['fact']}")
            if fact.get('conflict_resolution_note'):
                response_parts.append(f"    ⚠️ Conflict: {fact['conflict_resolution_note']}")
    else:
        response_parts.append("  No semantic facts found.")

    return "\n".join(response_parts)


def search_documents(query: str, top_k: int = 5, doc_type: str = None) -> str:
    """
    Runs hybrid RAG query with optional doc_type metadata filter.
    Returns answer with Self-RAG verification status.
    """
    where_filter = None
    if doc_type:
        where_filter = {"doc_type": doc_type}

    try:
        result = verified_rag_query(query, hybrid_rag_query, top_k=top_k, where_filter=where_filter)

        answer = result.get("answer", "No answer found.")
        verification = result.get("verification", {})
        support = verification.get("support_status", "UNKNOWN")
        reasoning = verification.get("support_reasoning", "")
        relevant_ratio = verification.get("relevant_chunks_ratio", "N/A")

        output = f"📄 Document Search Results:\n"
        output += f"Answer: {answer}\n\n"
        output += f"🔎 Verification Status: {support}\n"
        output += f"   Reasoning: {reasoning}\n"
        output += f"   Relevant chunks: {relevant_ratio}\n"
        return output

    except Exception as e:
        return f"Error during document search: {str(e)}"


def get_memory_summary() -> str:
    """
    Returns a summary of current memory state (total episodes, total facts, recent activity).
    """
    episodic = EpisodicStore(db_path=MEMORY_DB)
    semantic = SemanticStore(db_path=MEMORY_DB)

    # Count episodes
    all_episodes = episodic.get_all(limit=10000)
    total_episodes = len(all_episodes)
    unconsolidated = len(episodic.query_unconsolidated(limit=10000))

    # Count facts
    all_facts = semantic.search_facts("")
    active_facts = [f for f in all_facts if f['status'] == 'active']
    superseded_facts = [f for f in all_facts if f['status'] == 'superseded']

    # Recent activity
    recent = episodic.query_recent(limit=3)
    recent_summary = [f"[{r['timestamp']}] {r['content'][:80]}" for r in recent]

    summary = {
        "episodic_memory": {
            "total_episodes": total_episodes,
            "unconsolidated": unconsolidated,
            "consolidated": total_episodes - unconsolidated
        },
        "semantic_memory": {
            "active_facts": len(active_facts),
            "superseded_facts": len(superseded_facts),
            "total_versions": len(all_facts)
        },
        "recent_activity": recent_summary
    }

    return json.dumps(summary, indent=2)


if __name__ == "__main__":
    # Quick test
    print(get_memory_summary())
    print("\n" + recall_memory("Node 10"))
