# Context Management Evaluation Results

| Strategy | Accuracy (N/5) | Avg Input Tokens | Avg Output Tokens | Avg Latency (s) |
|----------|----------------|------------------|-------------------|-----------------|
| Sliding Window | 0/5 | 193 | 94 | 45.44 |
| Observation Masking | 4/5 | 193 | 161 | 28.70 |
| Recursive Summarization | 5/5 | 193 | 186 | 41.52 |
| Zone-Based Pruning | 4/5 | 193 | 125 | 29.22 |

## Justification
Based on the evaluation, **Recursive Summarization** is recommended. It effectively preserves critical historical context (evidenced by higher accuracy) while significantly reducing token bloat from tool outputs, providing a balance of performance and context retention suitable for NOC operations.