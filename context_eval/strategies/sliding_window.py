from typing import List, Dict, Any
from context_eval.token_utils import estimate_tokens


def apply(turns: List[Dict[str, Any]], window_size: int = 10) -> Dict[str, Any]:
    """
    Keep only the last `window_size` turns verbatim. Everything older is dropped
    entirely -- cheapest strategy, but anything buried before the window is gone.
    """
    kept = turns[-window_size:] if window_size > 0 else []
    input_tokens = sum(estimate_tokens(t.get("content", "")) for t in kept)
    return {
        "kept_turns": kept,
        "input_tokens": input_tokens,
        "output_tokens": 0,  # no LLM call needed for this strategy
    }
