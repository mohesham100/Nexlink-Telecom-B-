from typing import List, Dict, Any
from context_eval.token_utils import estimate_tokens


def apply(turns: List[Dict[str, Any]], keep_last_tool_outputs: int = 3) -> Dict[str, Any]:
    """
    Keep ALL dialogue turns (user/assistant) verbatim -- they're cheap and the bloat
    isn't there. Mask all tool_result turns except the most recent
    `keep_last_tool_outputs`, replacing older ones with a short placeholder.
    This targets NOC's actual failure mode: tool JSON bloat, not conversation length.
    """
    tool_indices = [i for i, t in enumerate(turns) if t.get("type") == "tool_result"]
    keep_indices = set(tool_indices[-keep_last_tool_outputs:]) if keep_last_tool_outputs > 0 else set()

    kept_turns = []
    for i, t in enumerate(turns):
        if t.get("type") == "tool_result" and i not in keep_indices:
            masked = dict(t)
            masked["content"] = f"[tool output masked -- {len(t.get('content', ''))} chars omitted]"
            kept_turns.append(masked)
        else:
            kept_turns.append(t)

    input_tokens = sum(estimate_tokens(t.get("content", "")) for t in kept_turns)
    return {
        "kept_turns": kept_turns,
        "input_tokens": input_tokens,
        "output_tokens": 0,  # masking is pure string substitution, no LLM call
    }
