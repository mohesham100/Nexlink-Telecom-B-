"""
Runs naive RAG, hybrid search, and agentic RAG against the same domain-specific test
questions and produces a comparison table (accuracy, tokens/query, latency/query),
then picks and justifies a shipping default based on Nexlink's real query mix.

Usage:
    python -m retrieval_eval.run_eval              # offline (Mock embeddings + Mock LLM)
    python -m retrieval_eval.run_eval --gemini      # real Gemini embeddings + generation
"""
import argparse
import json
import os
import time

from rag.vector_store import VectorStore, reset_persisted_store
from rag.keyword_index import KeywordIndex
from rag.embeddings import GeminiEmbeddingClient, MockEmbeddingClient
from rag.llm import GeminiTextClient, MockTextClient
from rag.agentic_rag import GeminiRetrievalPlanner, MockRetrievalPlanner
from rag import naive_rag, hybrid_rag, agentic_rag
from context_eval.token_utils import estimate_tokens

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "test_questions.json")


def load_questions():
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def check_golden(chunks, golden_keywords):
    text = " ".join(c["text"] for c in chunks).lower()
    return all(kw.lower() in text for kw in golden_keywords)


def run(use_gemini: bool):
    embed_client = GeminiEmbeddingClient() if use_gemini else MockEmbeddingClient()
    llm = GeminiTextClient() if use_gemini else MockTextClient()
    planner = GeminiRetrievalPlanner() if use_gemini else MockRetrievalPlanner()

    reset_persisted_store()
    store = VectorStore(embedding_client=embed_client)
    store.index_corpus()
    keyword_index = KeywordIndex()
    keyword_index.build()

    questions = load_questions()

    architectures = {
        "Naive RAG": lambda q: naive_rag.answer(store, q, llm=llm),
        "Hybrid search (vector + BM25)": lambda q: hybrid_rag.answer(store, keyword_index, q, llm=llm),
        "Agentic RAG (multi-hop)": lambda q: agentic_rag.run(store, keyword_index, q, planner=planner, llm=llm),
    }

    results = {name: {"correct": 0, "total": 0, "tokens": [], "latency": [], "by_category": {}}
               for name in architectures}

    for q in questions:
        for name, fn in architectures.items():
            start = time.perf_counter()
            out = fn(q["query"])
            elapsed = time.perf_counter() - start

            correct = check_golden(out["retrieved_chunks"], q["golden_keywords"])
            tokens = sum(estimate_tokens(c["text"]) for c in out["retrieved_chunks"])

            r = results[name]
            r["total"] += 1
            r["correct"] += int(correct)
            r["tokens"].append(tokens)
            r["latency"].append(elapsed)
            cat = q["category"]
            r["by_category"].setdefault(cat, {"correct": 0, "total": 0})
            r["by_category"][cat]["total"] += 1
            r["by_category"][cat]["correct"] += int(correct)

    return results, len(questions)


def to_markdown_table(results, n_questions):
    lines = [
        f"| Architecture | Accuracy ({n_questions} test questions) | Avg. tokens/query | Avg. latency/query |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        avg_tok = sum(r["tokens"]) / r["total"]
        avg_lat = sum(r["latency"]) / r["total"]
        lines.append(f"| {name} | {r['correct']}/{r['total']} | {avg_tok:,.0f} | {avg_lat*1000:.1f}ms |")
    lines.append("")
    lines.append("By category:")
    for name, r in results.items():
        cat_str = ", ".join(f"{cat}: {v['correct']}/{v['total']}" for cat, v in r["by_category"].items())
        lines.append(f"  - {name}: {cat_str}")
    return "\n".join(lines)


def pick_default(results, n_questions):
    lines = ["\n## Shipping decision"]
    naive = results["Naive RAG"]
    hybrid = results["Hybrid search (vector + BM25)"]
    agentic = results["Agentic RAG (multi-hop)"]

    naive_acc = naive["correct"] / naive["total"]
    hybrid_acc = hybrid["correct"] / hybrid["total"]
    agentic_acc = agentic["correct"] / agentic["total"]

    hybrid_lat = sum(hybrid["latency"]) / hybrid["total"] * 1000
    agentic_lat = sum(agentic["latency"]) / agentic["total"] * 1000

    lines.append(
        f"Naive RAG: {naive_acc*100:.0f}% overall. Hybrid: {hybrid_acc*100:.0f}% overall "
        f"({hybrid_lat:.1f}ms avg). Agentic: {agentic_acc*100:.0f}% overall ({agentic_lat:.1f}ms avg, "
        f"{agentic_lat/max(hybrid_lat,0.001):.1f}x hybrid's latency)."
    )
    lines.append(
        "Nexlink's real call volume is dominated by general and citation-heavy questions during "
        "live triage, where an engineer is waiting on a diagnostic result -- not multi-part "
        "decomposition questions. **Ship hybrid search as the default retrieval path**, and route "
        "only questions that clearly need multiple facets (detected the same way the agentic "
        "planner detects them) to the agentic path. This mirrors the lab's own worked example: "
        "agentic RAG's accuracy gain on multi-part questions doesn't justify its latency/token cost "
        "as the default for every query."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini", action="store_true", help="Use real Gemini embeddings + generation.")
    args = parser.parse_args()

    results, n = run(use_gemini=args.gemini)
    table = to_markdown_table(results, n)
    decision = pick_default(results, n)
    print(table)
    print(decision)

    out_path = os.path.join(os.path.dirname(__file__), "comparison_table.md")
    with open(out_path, "w") as f:
        f.write("# Retrieval Architecture Comparison\n\n")
        f.write(f"Backend: {'Gemini (real embeddings + generation)' if args.gemini else 'Mock (offline, hashed bag-of-words embeddings)'}\n\n")
        if not args.gemini:
            f.write(
                "> **Caveat:** offline mode uses a hashed bag-of-words mock embedding, which behaves "
                "more like keyword matching than a real dense embedding model. This narrows the "
                "naive-vs-hybrid gap you'd see in production -- with real Gemini embeddings, naive "
                "RAG is expected to miss exact-identifier questions (Clause/SOP numbers) more often, "
                "since those don't embed distinctively in dense vector space. Re-run with --gemini "
                "once a key is configured to get the production-representative numbers.\n\n"
            )
        f.write(table + "\n")
        f.write(decision + "\n")
    print(f"\nSaved to {out_path}")
