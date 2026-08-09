import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from rag.vector_store import VectorStore

def naive_rag_query(query: str, top_k: int = 5, where_filter: dict = None) -> dict:
    """
    Standard Naive RAG pipeline: retrieve top-k vector matches and generate answer.
    """
    start_time = time.time()
    
    # 1. Retrieve
    store = VectorStore()
    results = store.query(query, n_results=top_k, where_filter=where_filter)
    
    retrieved_chunks = []
    if results and results["documents"] and len(results["documents"]) > 0:
        retrieved_chunks = results["documents"][0]
        
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # 2. Generate
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
    
    # Extract usage if available, else approximate
    tokens_used = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        tokens_used = response.usage_metadata.get('total_tokens', 0)
    else:
        # Rough estimation if token usage is missing
        tokens_used = len(prompt.split()) + len(response.content.split())
        
    return {
        "answer": response.content.strip(),
        "retrieved_chunks": retrieved_chunks,
        "tokens_used": tokens_used,
        "latency": latency
    }
