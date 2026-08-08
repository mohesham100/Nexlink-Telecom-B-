from typing import List, Dict, Any
from context_eval.token_utils import estimate_tokens, render, is_high_signal
from context_eval.summarizer import Summarizer, MockSummarizer


def apply(turns: List[Dict[str, Any]], recent_window: int = 8,
          middle_window: int = 20, summarizer: Summarizer = None) -> Dict[str, Any]:
    """
    Four zones, each pruned differently:
      1. CRITICAL zone: any turn flagged high-signal (SLA/VIP/outage/etc keywords),
         regardless of age -- kept verbatim always.
      2. RECENT zone: last `recent_window` turns -- kept verbatim (agent's active context).
      3. MIDDLE zone: next `middle_window` turns back -- summarized via one LLM call.
      4. OLD zone: everything older than that -- dropped to a one-line marker.
    """
    summarizer = summarizer or MockSummarizer()
    n = len(turns)
    recent_start = max(0, n - recent_window)
    middle_start = max(0, recent_start - middle_window)

    recent_zone = turns[recent_start:]
    middle_zone = turns[middle_start:recent_start]
    old_zone = turns[:middle_start]

    # Critical turns from old/middle zones get pulled forward verbatim regardless of age.
    critical_from_old = [t for t in old_zone if is_high_signal(t)]
    critical_from_middle = [t for t in middle_zone if is_high_signal(t)]
    middle_zone_non_critical = [t for t in middle_zone if not is_high_signal(t)]

    kept_turns: List[Dict[str, Any]] = []
    output_tokens = 0

    if old_zone:
        kept_turns.append({
            "role": "system", "type": "zone_marker",
            "content": f"[{len(old_zone)} older turns pruned from OLD zone]",
        })
    kept_turns.extend(critical_from_old)

    if middle_zone_non_critical:
        summary = summarizer.summarize(render(middle_zone_non_critical))
        output_tokens += estimate_tokens(summary)
        kept_turns.append({"role": "system", "type": "summary", "content": summary})
    kept_turns.extend(critical_from_middle)

    kept_turns.extend(recent_zone)

    input_tokens = sum(estimate_tokens(t.get("content", "")) for t in kept_turns)
    return {
        "kept_turns": kept_turns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
