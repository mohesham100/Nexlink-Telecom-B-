import json
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class FactExtraction(BaseModel):
    fact: str = Field(description="The extracted factual claim")
    category: str = Field(description="Category of the fact: 'node_status', 'customer_config', 'incident_cause', 'procedure', or 'equipment_info'")

class FactsList(BaseModel):
    facts: List[FactExtraction]

class ConflictCheck(BaseModel):
    is_conflict: bool = Field(description="Whether the new fact contradicts the existing fact")
    conflict_resolution_note: str = Field(description="Explanation of contradiction and resolution (why new wins)")

class ConsolidationLayer:
    """
    Semantic Consolidation Layer.
    Periodically processes unconsolidated episodic memories to extract and update semantic facts.
    Uses ChatOllama with qwen-accurate:9b to extract facts and check conflicts.
    """
    
    def __init__(self, episodic_store, semantic_store, llm_model: str = "qwen-accurate:9b"):
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.llm = ChatOllama(model=llm_model, format="json")

    def _extract_facts(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Uses LLM to extract structured facts from episodes."""
        content_block = "\n".join([f"[{ep['timestamp']}] {ep['content']}" for ep in episodes])
        prompt = f"""
        Extract factual claims from the following episodic memory logs.
        For each fact, categorize it as: 'node_status', 'customer_config', 'incident_cause', 'procedure', or 'equipment_info'.
        Output ONLY a JSON object matching this schema:
        {{
            "facts": [
                {{"fact": "Extracted fact string", "category": "category_name"}}
            ]
        }}
        
        Logs:
        {content_block}
        """
        
        try:
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)
            # Find facts array in case the LLM wrapped it differently
            return data.get("facts", [])
        except Exception as e:
            print(f"Failed to extract facts: {e}")
            return []

    def _check_conflict(self, new_fact: str, existing_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Uses LLM to detect if new fact contradicts existing facts."""
        for ex_fact in existing_facts:
            prompt = f"""
            Compare these two facts for contradiction:
            Existing Fact: {ex_fact['fact']}
            New Fact: {new_fact}
            
            Are they in direct conflict or does the new fact supersede the old one?
            Output ONLY a JSON object matching this schema:
            {{
                "is_conflict": true/false,
                "conflict_resolution_note": "Explanation if conflict"
            }}
            """
            try:
                response = self.llm.invoke(prompt)
                result = json.loads(response.content)
                if result.get("is_conflict"):
                    return {
                        "conflict": True,
                        "old_fact_id": ex_fact["id"],
                        "note": result.get("conflict_resolution_note", "Newer fact supersedes older fact.")
                    }
            except Exception as e:
                print(f"Failed to check conflict: {e}")
                
        return {"conflict": False}

    def _resolve_conflict(self, new_fact_data: Dict[str, Any], conflict_data: Dict[str, Any], episode_ids: List[int]) -> None:
        """Resolves conflict by superseding the old fact with the new one."""
        self.semantic_store.update_fact(
            old_fact_id=conflict_data["old_fact_id"],
            new_fact=new_fact_data["fact"],
            source_episode_ids=episode_ids,
            conflict_note=conflict_data["note"]
        )

    def run_consolidation(self) -> Dict[str, Any]:
        """
        Main periodic pass for consolidation.
        """
        summary = {
            "episodes_processed": 0,
            "new_facts_added": 0,
            "conflicts_resolved": 0,
            "errors": 0
        }
        
        episodes = self.episodic_store.query_unconsolidated(limit=200)
        if not episodes:
            return summary
            
        summary["episodes_processed"] = len(episodes)
        
        # Group by entity_id
        grouped = {}
        for ep in episodes:
            # We group by node_id or customer_id, or "general" if neither
            entity = ep.get("node_id") or ep.get("customer_id") or "general"
            if entity not in grouped:
                grouped[entity] = []
            grouped[entity].append(ep)
            
        for entity_id, eps in grouped.items():
            episode_ids = [e["id"] for e in eps]
            facts_extracted = self._extract_facts(eps)
            
            # Use None for "general" entity_id
            db_entity_id = None if entity_id == "general" else entity_id
            
            for fact_data in facts_extracted:
                category = fact_data.get("category", "procedure")
                fact_text = fact_data.get("fact", "")
                
                existing = self.semantic_store.query_facts(entity_id=db_entity_id, category=category)
                
                # Check for exact matches to skip
                if any(ex["fact"] == fact_text for ex in existing):
                    continue
                    
                conflict_check = self._check_conflict(fact_text, existing)
                
                if conflict_check.get("conflict"):
                    self._resolve_conflict(fact_data, conflict_check, episode_ids)
                    summary["conflicts_resolved"] += 1
                else:
                    self.semantic_store.add_fact(
                        fact=fact_text,
                        category=category,
                        entity_id=db_entity_id,
                        source_episode_ids=episode_ids
                    )
                    summary["new_facts_added"] += 1
                    
            # Mark processed episodes as consolidated
            self.episodic_store.mark_consolidated(episode_ids)
            
        # Expire stale facts
        self.semantic_store.expire_stale_facts()
        
        return summary
