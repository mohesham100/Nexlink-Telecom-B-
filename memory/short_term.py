import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class ShortTermMemory:
    """
    Rolling message buffer for short-term memory.
    Holds a configurable maximum number of messages and prunes the oldest when overflowing.
    """
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []

    def add(self, role: str, content: str, msg_type: str) -> None:
        """Adds to buffer, triggers overflow check."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "type": msg_type,
            "token_estimate": self._estimate_tokens(content)
        }
        self.buffer.append(entry)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation based on word count."""
        return len(text.split()) * 4 // 3  # rough estimate

    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """Returns last n messages."""
        return self.buffer[-n:] if n > 0 else []

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns full buffer."""
        return list(self.buffer)

    def get_by_type(self, msg_type: str) -> List[Dict[str, Any]]:
        """Filter by type."""
        return [msg for msg in self.buffer if msg.get("type") == msg_type]

    def is_overflowing(self) -> bool:
        """True if buffer size >= max_size."""
        return len(self.buffer) >= self.max_size

    def get_overflow_items(self, n: int) -> List[Dict[str, Any]]:
        """Returns n oldest items to be routed."""
        return self.buffer[:n]

    def remove_oldest(self, n: int) -> None:
        """Removes n oldest from buffer."""
        if n > 0:
            self.buffer = self.buffer[n:]

    def estimate_tokens(self) -> int:
        """Rough token count of the entire buffer."""
        return sum(msg.get("token_estimate", 0) for msg in self.buffer)


class Scratchpad:
    """
    Scratchpad for maintaining the current state, plan, and active incident.
    KEY DESIGN: Scratchpad is NEVER pruned by buffer overflow. When the buffer prunes 
    old messages, the scratchpad remains intact. This is independent of ShortTermMemory.
    """
    def __init__(self):
        self.current_plan: str = ""
        self.sub_goals: List[str] = []
        self.working_state: Dict[str, Any] = {}
        self.active_incident: Optional[Dict[str, Any]] = None

    def update_plan(self, plan: str) -> None:
        self.current_plan = plan

    def add_sub_goal(self, goal: str) -> None:
        self.sub_goals.append(goal)

    def complete_sub_goal(self, index: int) -> None:
        if 0 <= index < len(self.sub_goals):
            self.sub_goals.pop(index)

    def set_working_state(self, key: str, value: Any) -> None:
        self.working_state[key] = value

    def get_working_state(self, key: str) -> Any:
        return self.working_state.get(key)

    def set_active_incident(self, incident_dict: Dict[str, Any]) -> None:
        self.active_incident = incident_dict

    def clear_active_incident(self) -> None:
        self.active_incident = None

    def to_context_string(self) -> str:
        """Serializes scratchpad to a string for LLM context injection."""
        context_parts = []
        if self.current_plan:
            context_parts.append(f"Current Plan: {self.current_plan}")
        if self.sub_goals:
            goals_str = "\n".join([f"- {g}" for g in self.sub_goals])
            context_parts.append(f"Sub-Goals:\n{goals_str}")
        if self.working_state:
            context_parts.append(f"Working State: {json.dumps(self.working_state)}")
        if self.active_incident:
            context_parts.append(f"Active Incident: {json.dumps(self.active_incident)}")
            
        return "\n\n".join(context_parts) if context_parts else "Scratchpad is empty."

    def clear(self) -> None:
        """Resets everything."""
        self.current_plan = ""
        self.sub_goals = []
        self.working_state = {}
        self.active_incident = None
