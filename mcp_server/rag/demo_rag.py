import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mcp_server.rag.rag_tool import search_knowledge_base_handler, index_documents

if __name__ == "__main__":
    index_documents([
        {
            "text": "Visit 2026-03-01: Fiber break near Cairo Metro Line 3, prescribed drops.",
            "entity_id": "node_10",
            "role_required": "any",
        },
        {
            "text": "Visit 2026-05-14: High optical attenuation, NOC sign-off required.",
            "entity_id": "node_10",
            "role_required": "NOC_Engineer",
        },
        {
            "text": "Visit 2026-01-10: Routine checkup, no notes.",
            "entity_id": "node_99",
            "role_required": "any",
        },
    ])

    guest_result = search_knowledge_base_handler(
        {"query": "Fiber break Cairo Metro", "entity_id": "node_10", "top_k": 5},
        session_role="Guest",
    )
    print("--- Guest sees ---")
    print(guest_result)

    engineer_result = search_knowledge_base_handler(
        {"query": "optical attenuation", "entity_id": "node_10", "top_k": 5},
        session_role="NOC_Engineer",
    )
    print("\n--- NOC_Engineer sees ---")
    print(engineer_result)

    empty_result = search_knowledge_base_handler(
        {"query": "anything", "entity_id": "node_does_not_exist", "top_k": 5},
        session_role="NOC_Engineer",
    )
    print("\n--- unknown entity ---")
    print(empty_result)

    no_match_result = search_knowledge_base_handler(
        {"query": "broken leg surgery", "entity_id": "node_10", "top_k": 5},
        session_role="NOC_Engineer",
    )
    print("\n--- no keyword overlap ---")
    print(no_match_result)
