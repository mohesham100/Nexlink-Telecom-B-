import time
import os
from rank_bm25 import BM25Okapi
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from rag.vector_store import VectorStore

class HybridRetriever:
    def __init__(self):
        self.store = VectorStore()
        # Initialize BM25 on the fly for simplicity (in a real prod app, you'd cache/persist this)
        self.corpus_chunks = []
        self._build_bm25()

    def _build_bm25(self):
        """Fetches all documents from ChromaDB to build the BM25 index."""
        # A bit hacky: get all docs to build BM25. 
        # In a real scenario, maintain parallel BM25 index.
        results = self.store.collection.get()
        self.corpus_chunks = results["documents"]
        
        tokenized_corpus = [doc.lower().split() for doc in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5, where_filter: dict = None) -> list[str]:
        # 1. Vector Search
        vector_results = self.store.query(query, n_results=top_k, where_filter=where_filter)
        vector_docs = vector_results["documents"][0] if vector_results and vector_results["documents"] else []
        
        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_docs = [self.corpus_chunks[i] for i in bm25_top_indices]
        
        # 3. Reciprocal Rank Fusion (RRF)
        k = 60
        rrf_scores = {}
        
        for rank, doc in enumerate(vector_docs):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += 1.0 / (k + rank + 1)
            
        for rank, doc in enumerate(bm25_docs):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += 1.0 / (k + rank + 1)
            
        # Sort by RRF score and take top_k
        sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_docs[:top_k]


def hybrid_rag_query(query: str, top_k: int = 5, where_filter: dict = None) -> dict:
    """
    Hybrid RAG pipeline: Combines vector similarity (ChromaDB) + keyword BM25.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """
    start_time = time.time()
    
    retriever = HybridRetriever()
    retrieved_chunks = retriever.retrieve(query, top_k=top_k, where_filter=where_filter)
    
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    llm = ChatOllama(model="qwen-accurate:9b", temperature=0)
    
    prompt = f"""You are a NOC assistant for Nexlink Telecom. 
Answer the following query using ONLY the provided context. If the context does not contain the answer, say "I don't know based on the provided context."

Context:
{context}

Query: {query}
"""

    messages = [
        SystemMessage(content="You are a helpful NOC AI assistant."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    latency = time.time() - start_time
    
    tokens_used = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        tokens_used = response.usage_metadata.get('total_tokens', 0)
    else:
        tokens_used = len(prompt.split()) + len(response.content.split())
        
    return {
        "answer": response.content.strip(),
        "retrieved_chunks": retrieved_chunks,
        "tokens_used": tokens_used,
        "latency": latency
    }
