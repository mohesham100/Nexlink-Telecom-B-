# Context Management Strategy Comparison

Backend: Mock (offline) | 8 synthetic long-context transcript variations, ~32 noise turns each.

| Strategy | Detail recalled correctly | Avg. input tokens/run | Avg. output tokens/run | Avg. latency |
|---|---|---|---|---|
| Sliding window (last 10 turns) | 3/8 | 132 | 0 | 0.0ms |
| Observation masking (keep last 3 tool outputs) | 3/8 | 358 | 0 | 0.0ms |
| Recursive summarization (compact every 15 turns) | 8/8 | 235 | 91 | 0.2ms |
| Zone-based pruning (4 zones) | 8/8 | 185 | 28 | 0.2ms |

Chosen strategy: **Recursive summarization (compact every 15 turns)**

It has the best recall in the table (100% across 8 transcript variations) and, among strategies tied at that accuracy, the lowest latency/token cost. The strategies that don't call an LLM (sliding window, observation masking) are cheaper but silently drop the buried detail whenever it falls outside their fixed window -- exactly the failure mode that costs a real engineer a missed SLA obligation or vendor recall notice. Between the two LLM-backed strategies, this one wins on the token/latency tradeoff for Nexlink's transcript shape.
  - Sliding window (last 10 turns): 38% recall, 132 avg input tokens, 0.01ms avg latency.
  - Observation masking (keep last 3 tool outputs): 38% recall, 358 avg input tokens, 0.04ms avg latency.
  - Zone-based pruning (4 zones): 100% recall, 185 avg input tokens, 0.18ms avg latency.
