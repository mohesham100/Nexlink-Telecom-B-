"""
Summarizer backend shared by recursive_summarization.py and zone_based_pruning.py.

Same lazy-Gemini / offline-mock pattern as memory/gemini_client.py, but for plain
text generation rather than structured JSON.
"""
import os
import re
from typing import Optional, Protocol
from context_eval.token_utils import CRITICAL_KEYWORDS, estimate_tokens


class Summarizer(Protocol):
    def summarize(self, text: str) -> str: ...


class GeminiSummarizer:
    def __init__(self, model: Optional[str] = None):
        self.model_name = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Use MockSummarizer for offline runs, "
                    "or set the key in .env to use the real Gemini backend."
                )
            from google import genai
            self._client = genai.Client(api_key=api_key)
        return self._client

    def summarize(self, text: str) -> str:
        client = self._get_client()
        prompt = (
            "Compact the following NOC agent conversation/tool-output log into a short "
            "summary that preserves every specific fact, number, ID, SLA/compliance "
            "detail, and decision. Drop only small talk and redundant chatter.\n\n"
            f"{text}"
        )
        response = client.models.generate_content(model=self.model_name, contents=prompt)
        return (response.text or "").strip()


class MockSummarizer:
    """
    Offline extractive summarizer: keeps the first sentence, the last sentence, and any
    sentence containing a high-signal NOC keyword (SLA, VIP, outage, etc). This is a
    realistic stand-in for what a real LLM summarizer usually keeps -- important detail
    survives IF it's flagged as high-signal, but generic tool-output noise doesn't.
    """

    def summarize(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+|\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return ""
        keep = set()
        keep.add(0)
        keep.add(len(sentences) - 1)
        for i, s in enumerate(sentences):
            if any(kw in s.lower() for kw in CRITICAL_KEYWORDS):
                keep.add(i)
        summary_sentences = [sentences[i] for i in sorted(keep)]
        return "[SUMMARY] " + " ".join(summary_sentences)
