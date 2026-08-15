import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from rag.vector_store import VectorStore

def agentic_rag_query(query: str, top_k: int = 5, where_filter: dict = None) -> dict:
    """
    Multi-hop retrieval loop:
    1. LLM analyzes query and formulates search.
    2. Retrieve from vector store.
    3. LLM decides if sufficient or needs re-retrieval.
    4. May rewrite query and retrieve again (max 3 hops).
    5. Final generation.
    """
    start_time = time.time()
    store = VectorStore()
    llm = ChatOllama(model="qwen-accurate:9b", temperature=0)
    
    max_hops = 3
    current_hop = 1
    gathered_context = set()
    current_query = query
    tokens_used = 0
    
    while current_hop <= max_hops:
        # Retrieve
        results = store.query(current_query, n_results=top_k, where_filter=where_filter)
        chunks = results["documents"][0] if results and results["documents"] else []
        for chunk in chunks:
            gathered_context.add(chunk)
            
        context_str = "\n\n---\n\n".join(gathered_context)
        
        # Decide if sufficient
        eval_prompt = f"""You are evaluating if the current context is sufficient to answer the user's original query.
Original Query: {query}
Current Context:
{context_str}

If the context contains enough information to fully answer the query, reply strictly with "SUFFICIENT".
If not, reply strictly with a new, different search query that might help find the missing information.
"""
        messages = [
            SystemMessage(content="You are a retrieval evaluator. Be concise."),
            HumanMessage(content=eval_prompt)
        ]
        
        response = llm.invoke(messages)
        decision = response.content.strip()
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_used += response.usage_metadata.get('total_tokens', 0)
        else:
            tokens_used += len(eval_prompt.split()) + len(decision.split())
            
        if decision == "SUFFICIENT" or current_hop == max_hops:
            break
            
        current_query = decision
        current_hop += 1
        
    # Final Generation
    final_context = "\n\n---\n\n".join(gathered_context)
    gen_prompt = f"""You are a NOC assistant for Nexlink Telecom. 
Answer the original query using ONLY the gathered context.

Context:
{final_context}

Original Query: {query}
"""
    messages = [
        SystemMessage(content="You are a helpful NOC AI assistant."),
        HumanMessage(content=gen_prompt)
    ]
    final_response = llm.invoke(messages)
    
    if hasattr(final_response, 'usage_metadata') and final_response.usage_metadata:
        tokens_used += final_response.usage_metadata.get('total_tokens', 0)
    else:
        tokens_used += len(gen_prompt.split()) + len(final_response.content.split())
        
    latency = time.time() - start_time
    
    return {
        "answer": final_response.content.strip(),
        "retrieved_chunks": list(gathered_context),
        "tokens_used": tokens_used,
        "latency": latency,
        "hops": current_hop
    }
