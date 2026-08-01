# Issue: Add search_knowledge_base tool (Option A - RAG)

## Why this is needed
NOC engineers need to search unstructured incident visit notes and runbook reports by entity (node ID) and keywords instead of reading large policy files into the LLM context window.

## Changes
- Created `mcp_server/rag/keyword_search.py` for keyword BM25 retrieval.
- Created `mcp_server/rag/rag_tool.py` with typed Pydantic input schema `SearchKnowledgeBaseInput`.
- Added role checking in handler (`role_required` in `("any", session_role)`).
- Registered `search_knowledge_base` tool on MCP server.
