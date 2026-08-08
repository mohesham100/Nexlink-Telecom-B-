from typing import List, Dict, Any
from context_eval.token_utils import estimate_tokens, render
from context_eval.summarizer import Summarizer, MockSummarizer


def apply(turns: List[Dict[str, Any]], compact_every: int = 15,
          summarizer: Summarizer = None) -> Dict[str, Any]:
    """
    Every `compact_every` turns, compact that chunk into a single summary turn via
    an LLM call. Keeps all summaries plus any turns not yet old enough to compact.
    Costs extra output tokens (the summary generations) and latency (LLM round trips),
    the tradeoff this strategy is scored against.
    """
    summarizer = summarizer or MockSummarizer()
    kept_turns: List[Dict[str, Any]] = []
    output_tokens = 0

    chunks = [turns[i:i + compact_every] for i in range(0, len(turns), compact_every)]
    # Keep the LAST chunk verbatim (still "fresh"); summarize all earlier chunks.
    for idx, chunk in enumerate(chunks):
        if idx == len(chunks) - 1:
            kept_turns.extend(chunk)
        else:
            chunk_text = render(chunk)
            summary = summarizer.summarize(chunk_text)
            output_tokens += estimate_tokens(summary)
            kept_turns.append({
                "role": "system",
                "type": "summary",
                "content": summary,
            })

    input_tokens = sum(estimate_tokens(t.get("content", "")) for t in kept_turns)
    return {
        "kept_turns": kept_turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
