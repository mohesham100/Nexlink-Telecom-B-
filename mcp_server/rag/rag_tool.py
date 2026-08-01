from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from mcp_server.rag.keyword_search import KeywordStore

knowledge_store = KeywordStore()

def index_documents(docs: list[dict]):
    """Index list of document dicts into knowledge_store."""
    for doc in docs:
        knowledge_store.upsert(
            payload=doc["text"],
            metadata={
                "entity_id": doc["entity_id"],
                "role_required": doc.get("role_required", "any"),
            },
        )

class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(..., description="Keywords to search for")
    entity_id: str = Field(..., description="Scope search to this entity only")
    top_k: int = Field(default=3, ge=1, le=10)

    model_config = ConfigDict(extra="forbid")  # additionalProperties: false

def search_knowledge_base_handler(args: dict, session_role: str) -> str:
    parsed = SearchKnowledgeBaseInput.model_validate(args)

    matches = knowledge_store.query(
        query_text=parsed.query,
        top_k=parsed.top_k,
        filter={"entity_id": parsed.entity_id},
    )

    visible = [
        m for m in matches
        if m["metadata"]["role_required"] in ("any", session_role)
    ]

    if not visible:
        return "No relevant records found for this query."

    return "\n\n".join(m["payload"] for m in visible)

# Default NOC domain documents
index_documents([
    {
        "text": "Visit 2026-03-01 Node node_10: Fiber break near Cairo Metro Line 3 excavation. Splicing repaired.",
        "entity_id": "node_10",
        "role_required": "any",
    },
    {
        "text": "Visit 2026-05-14 Node node_10: High optical attenuation fixed by cleaning LC connectors at patch panel 4B.",
        "entity_id": "node_10",
        "role_required": "NOC_Engineer",
    },
    {
        "text": "Visit 2026-01-10 Node node_99: Routine checkup, no issues.",
        "entity_id": "node_99",
        "role_required": "any",
    },
])
