"""Shared local model used by the hands-on scripts.

The guide calls ChatAnthropic. This machine has no Anthropic key, so the same
agent loop runs on the local Ollama model qwen3:4b.
"""

from langchain_ollama import ChatOllama

MODEL_NAME = "qwen3:4b"


def make_llm(num_predict: int = 512) -> ChatOllama:
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        reasoning=False,
        num_predict=num_predict,
    )


def visible_text(content: object) -> str:
    """Drop a leftover qwen think-tag so the printed answer is the final text."""
    text = content if isinstance(content, str) else str(content or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
