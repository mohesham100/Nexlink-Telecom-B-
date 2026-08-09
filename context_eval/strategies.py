import json
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def estimate_tokens(text: str) -> int:
    """Estimates the token count based on word count."""
    return int(len(text.split()) * 1.3)

def _estimate_transcript_tokens(transcript: List[Dict[str, Any]]) -> int:
    """Estimates the token count of a full transcript."""
    total = 0
    for msg in transcript:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        else:
            total += estimate_tokens(str(content))
    return total

def sliding_window(transcript: List[Dict[str, Any]], window_size: int = 10) -> Dict[str, Any]:
    """
    Strategy 1: Sliding Window
    Keeps only the last `window_size` turns.
    """
    tokens_before = _estimate_transcript_tokens(transcript)
    pruned_transcript = transcript[-window_size:] if len(transcript) > window_size else transcript.copy()
    tokens_after = _estimate_transcript_tokens(pruned_transcript)
    
    return {
        "pruned_transcript": pruned_transcript,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after
    }

def observation_masking(transcript: List[Dict[str, Any]], keep_last_n_tool_outputs: int = 3) -> Dict[str, Any]:
    """
    Strategy 2: Observation/Tool-Output Masking
    Keeps ALL user messages and AI responses.
    For tool_result messages: keeps the last `keep_last_n_tool_outputs` full tool outputs,
    replaces older ones with a summarized line.
    """
    tokens_before = _estimate_transcript_tokens(transcript)
    
    # First, identify all tool results
    tool_result_indices = [i for i, msg in enumerate(transcript) if msg.get("role") in ("tool", "tool_result")]
    keep_indices = set(tool_result_indices[-keep_last_n_tool_outputs:]) if tool_result_indices else set()
    
    pruned_transcript = []
    for i, msg in enumerate(transcript):
        if msg.get("role") in ("tool", "tool_result"):
            if i in keep_indices:
                pruned_transcript.append(msg.copy())
            else:
                tool_name = msg.get("name", msg.get("tool_name", "unknown_tool"))
                summarized_msg = msg.copy()
                summarized_msg["content"] = f"[Tool output summarized: {tool_name} returned data]"
                pruned_transcript.append(summarized_msg)
        else:
            pruned_transcript.append(msg.copy())
            
    tokens_after = _estimate_transcript_tokens(pruned_transcript)
    
    return {
        "pruned_transcript": pruned_transcript,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after
    }

def recursive_summarization(transcript: List[Dict[str, Any]], chunk_size: int = 15, llm_model: str = "qwen-accurate:9b") -> Dict[str, Any]:
    """
    Strategy 3: Recursive Summarization
    Every `chunk_size` turns, summarize the oldest block into a compact summary message.
    """
    tokens_before = _estimate_transcript_tokens(transcript)
    
    if len(transcript) <= chunk_size:
        return {
            "pruned_transcript": transcript.copy(),
            "tokens_before": tokens_before,
            "tokens_after": tokens_before
        }
    
    llm = ChatOllama(model=llm_model, temperature=0)
    
    # We will summarize everything except the last `chunk_size` turns
    to_summarize = transcript[:-chunk_size]
    recent_turns = transcript[-chunk_size:]
    
    summary_prompt = "Summarize the following conversation history, highlighting any critical details about node status, VIP customers, SLAs, or ongoing issues:\n\n"
    for msg in to_summarize:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        summary_prompt += f"{role}: {content}\n\n"
        
    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content
    except Exception as e:
        summary = f"Summary failed: {str(e)}"
        
    pruned_transcript = [{"role": "system", "content": f"Previous conversation summary: {summary}"}] + recent_turns
    
    tokens_after = _estimate_transcript_tokens(pruned_transcript)
    
    return {
        "pruned_transcript": pruned_transcript,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after
    }

def zone_based_pruning(transcript: List[Dict[str, Any]], zones: dict = None) -> Dict[str, Any]:
    """
    Strategy 4: Zone-Based Pruning
    Divides context into 4 zones:
    Zone 1 - System: always kept
    Zone 2 - Recent dialogue: last 8 turns (non-tool)
    Zone 3 - Tool outputs: last 3 full
    Zone 4 - Historical context: dropped/summarized (for simplicity here, we mask old tools and drop old dialogue except system)
    """
    tokens_before = _estimate_transcript_tokens(transcript)
    
    # Identify system messages
    system_msgs = [m for m in transcript if m.get("role") == "system"]
    
    # Separate recent dialogue (last 8 non-tool messages)
    non_tool_msgs = [m for m in transcript if m.get("role") not in ("tool", "tool_result", "system")]
    recent_dialogue = non_tool_msgs[-8:] if len(non_tool_msgs) > 8 else non_tool_msgs
    
    # Separate tool outputs (last 3 kept full, others summarized)
    tool_msgs = [m for i, m in enumerate(transcript) if m.get("role") in ("tool", "tool_result")]
    recent_tools = tool_msgs[-3:] if len(tool_msgs) > 3 else tool_msgs
    old_tools = tool_msgs[:-3] if len(tool_msgs) > 3 else []
    
    summarized_old_tools = []
    for msg in old_tools:
        tool_name = msg.get("name", msg.get("tool_name", "unknown_tool"))
        m = msg.copy()
        m["content"] = f"[Tool output summarized: {tool_name} returned data]"
        summarized_old_tools.append(m)
        
    # Reconstruct transcript preserving order roughly
    # A robust implementation would map indices, but here we rebuild functionally:
    pruned_transcript = []
    pruned_transcript.extend(system_msgs)
    
    # To preserve order, let's just do a single pass using the sets we defined
    recent_dialogue_ids = {id(m) for m in recent_dialogue}
    recent_tool_ids = {id(m) for m in recent_tools}
    old_tool_ids = {id(m) for m in old_tools}
    
    for msg in transcript:
        m_id = id(msg)
        if msg.get("role") == "system":
            continue # already added
        elif m_id in recent_dialogue_ids:
            pruned_transcript.append(msg.copy())
        elif m_id in recent_tool_ids:
            pruned_transcript.append(msg.copy())
        elif m_id in old_tool_ids:
            tool_name = msg.get("name", msg.get("tool_name", "unknown_tool"))
            m = msg.copy()
            m["content"] = f"[Tool output summarized: {tool_name} returned data]"
            pruned_transcript.append(m)
        # Old dialogue is dropped (Zone 4 optimization)
            
    tokens_after = _estimate_transcript_tokens(pruned_transcript)
    
    return {
        "pruned_transcript": pruned_transcript,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after
    }
