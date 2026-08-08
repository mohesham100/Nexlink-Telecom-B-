# Retrieval Architecture Comparison

Backend: Mock (offline, hashed bag-of-words embeddings)

> **Caveat:** offline mode uses a hashed bag-of-words mock embedding, which behaves more like keyword matching than a real dense embedding model. This narrows the naive-vs-hybrid gap you'd see in production -- with real Gemini embeddings, naive RAG is expected to miss exact-identifier questions (Clause/SOP numbers) more often, since those don't embed distinctively in dense vector space. Re-run with --gemini once a key is configured to get the production-representative numbers.

| Architecture | Accuracy (12 test questions) | Avg. tokens/query | Avg. latency/query |
|---|---|---|---|
| Naive RAG | 10/12 | 205 | 1.5ms |
| Hybrid search (vector + BM25) | 10/12 | 210 | 1.7ms |
| Agentic RAG (multi-hop) | 12/12 | 236 | 1.9ms |

By category:
  - Naive RAG: general: 4/4, citation: 4/4, multi_part: 2/4
  - Hybrid search (vector + BM25): general: 4/4, citation: 4/4, multi_part: 2/4
  - Agentic RAG (multi-hop): general: 4/4, citation: 4/4, multi_part: 4/4

## Shipping decision
Naive RAG: 83% overall. Hybrid: 83% overall (1.7ms avg). Agentic: 100% overall (1.9ms avg, 1.1x hybrid's latency).
Nexlink's real call volume is dominated by general and citation-heavy questions during live triage, where an engineer is waiting on a diagnostic result -- not multi-part decomposition questions. **Ship hybrid search as the default retrieval path**, and route only questions that clearly need multiple facets (detected the same way the agentic planner detects them) to the agentic path. This mirrors the lab's own worked example: agentic RAG's accuracy gain on multi-part questions doesn't justify its latency/token cost as the default for every query.
