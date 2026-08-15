"""
Nexlink Telecom NOC RAG Package

This package provides various Retrieval-Augmented Generation (RAG) implementations
for querying NOC documentation, post-mortems, maintenance visits, and vendor bulletins.

Modules:
- vector_store: ChromaDB wrapper for vector storage and retrieval.
- ingest: Pipeline for chunking and embedding corpus documents.
- naive_rag: Standard top-k vector search and generation.
- hybrid_rag: Combines vector similarity with BM25 keyword search.
- agentic_rag: Multi-hop retrieval with LLM reasoning.
- self_rag: Self-reflection RAG for answer verification.
"""
