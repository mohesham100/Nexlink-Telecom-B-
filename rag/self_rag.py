import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

def check_relevance(query: str, chunk: str) -> dict:
    """
    Is the retrieved chunk actually relevant to the query?
    """
    llm = ChatOllama(model="qwen-accurate:9b", temperature=0)
    prompt = f"""Evaluate if the following text chunk is relevant to the query.
Query: {query}
Chunk: {chunk}

Return your evaluation as a JSON object with two keys:
"relevant": boolean (true if relevant, false otherwise)
"reasoning": string (brief explanation)
"""
    messages = [
        SystemMessage(content="You are a JSON-outputting relevance evaluator."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Try to parse JSON
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in content:
            json_str = content.split("```")[1].strip()
            return json.loads(json_str)
        else:
            return json.loads(content)
    except json.JSONDecodeError:
        # Fallback
        is_relevant = "true" in content.lower() and "false" not in content.lower()
        return {"relevant": is_relevant, "reasoning": "Failed to parse JSON, guessed from text."}

def check_support(query: str, answer: str, chunks: list[str]) -> dict:
    """
    Is the generated answer actually supported by the retrieved chunks?
    """
    llm = ChatOllama(model="qwen-accurate:9b", temperature=0)
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Evaluate if the generated answer is supported by the provided context.
Query: {query}
Answer: {answer}
Context:
{context}

Return your evaluation as a JSON object with two keys:
"supported": string (must be one of: "SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")
"reasoning": string (brief explanation)
"""
    messages = [
        SystemMessage(content="You are a JSON-outputting support evaluator."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in content:
            json_str = content.split("```")[1].strip()
            return json.loads(json_str)
        else:
            return json.loads(content)
    except json.JSONDecodeError:
        return {"supported": "NOT_SUPPORTED", "reasoning": "Failed to parse JSON."}

def verified_rag_query(query: str, rag_fn, **kwargs) -> dict:
    """
    Wraps any RAG function with Self-RAG verification.
    """
    # 1. Run standard RAG
    result = rag_fn(query, **kwargs)
    
    # 2. Check Relevance (Optional, we could filter chunks before generation, but here we just score them)
    relevant_chunks = []
    for chunk in result["retrieved_chunks"]:
        rel = check_relevance(query, chunk)
        if rel.get("relevant"):
            relevant_chunks.append(chunk)
            
    # 3. Check Support
    support = check_support(query, result["answer"], result["retrieved_chunks"])
    
    result["verification"] = {
        "support_status": support.get("supported", "UNKNOWN"),
        "support_reasoning": support.get("reasoning", ""),
        "relevant_chunks_ratio": f"{len(relevant_chunks)}/{len(result['retrieved_chunks'])}" if result['retrieved_chunks'] else "0/0"
    }
    
    if support.get("supported") != "SUPPORTED":
        result["answer"] = f"[WARNING: Answer may not be fully supported by context]\n{result['answer']}"
        
    return result
