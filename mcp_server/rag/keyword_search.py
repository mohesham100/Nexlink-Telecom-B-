import math
import re
from typing import List, Dict, Any, Optional

class KeywordStore:
    """Simple in-memory BM25 keyword store."""
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    def upsert(self, payload: str, metadata: Optional[dict] = None):
        self.docs.append({"payload": payload, "metadata": metadata or {}})

    def query(self, query_text: str, top_k: int = 3, filter: Optional[dict] = None) -> List[dict]:
        words = set(re.findall(r'\w+', query_text.lower()))
        results = []
        for doc in self.docs:
            if filter and any(doc["metadata"].get(k) != v for k, v in filter.items()):
                continue
            doc_words = set(re.findall(r'\w+', doc["payload"].lower()))
            score = len(words & doc_words)
            if score > 0:
                results.append((score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:top_k]]
