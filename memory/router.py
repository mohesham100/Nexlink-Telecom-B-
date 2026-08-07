import json
from datetime import datetime
from typing import List, Dict, Any

class MemoryRouter:
    """
    Promote-or-Drop Router for processing short-term memory overflows.
    KEY CONSTRAINT: This router does NOT write to semantic memory. Only forget or episodic.
    """
    
    def __init__(self, episodic_store, routing_log_path: str):
        self.episodic_store = episodic_store
        self.routing_log_path = routing_log_path
        
        # Keywords that boost importance
        self.critical_keywords = ["error", "down", "upgraded", "incident", "vip", "critical"]

    def _evaluate_importance(self, item: Dict[str, Any]) -> float:
        """Returns 0.0-1.0 importance score."""
        score = 0.2  # baseline
        
        msg_type = item.get("type", "")
        content = item.get("content", "").lower()
        
        # Boost based on message type
        if msg_type in ["tool_result", "user_decision"]:
            score += 0.3
        elif msg_type == "tool_call":
            score += 0.1
            
        # Boost based on keywords
        if any(keyword in content for keyword in self.critical_keywords):
            score += 0.3
            
        # Boost based on entity mentions (simple heuristic: specific node or customer)
        if "node " in content or "customer " in content or "id=" in content:
            score += 0.2
            
        return min(1.0, score)

    def _decide(self, item: Dict[str, Any], importance: float) -> str:
        """Returns 'forget' or 'episodic' based on importance threshold."""
        # THRESHOLD: items with importance >= 0.4 get promoted, below get forgotten
        return "episodic" if importance >= 0.4 else "forget"

    def _log_decision(self, item: Dict[str, Any], decision: str, importance: float, reasoning: str) -> None:
        """Appends to routing_log.jsonl."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "content_snippet": item.get("content", "")[:100],
            "decision": decision,
            "importance": importance,
            "reasoning": reasoning
        }
        with open(self.routing_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def route(self, items: List[Dict[str, Any]], session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        The main routing function.
        Processes each aging item from short-term overflow, decides to FORGET or PROMOTE.
        Returns counts of forgotten and promoted items.
        """
        result = {"forgotten": [], "promoted": []}
        
        for item in items:
            importance = self._evaluate_importance(item)
            decision = self._decide(item, importance)
            
            if decision == "episodic":
                # Determine event type simply based on content
                content = item.get("content", "")
                event_type = "diagnostic_result"
                if "change" in content.lower() or "upgraded" in content.lower():
                    event_type = "config_change"
                elif "incident" in content.lower() or "error" in content.lower() or "down" in content.lower():
                    event_type = "incident_finding"
                elif item.get("type") == "user_decision":
                    event_type = "user_decision"
                    
                # Note: node_id and customer_id extraction could be more robust, defaulting to None
                self.episodic_store.store(
                    session_id=session_id,
                    event_type=event_type,
                    content=item.get("content", ""),
                    node_id=None,
                    customer_id=None,
                    importance=importance,
                    metadata={"source_role": item.get("role"), "original_type": item.get("type")}
                )
                result["promoted"].append(item)
                reasoning = "Importance threshold met, storing to episodic memory."
            else:
                result["forgotten"].append(item)
                reasoning = "Low importance, routing to forget."
                
            self._log_decision(item, decision, importance, reasoning)
            
        return result
