# modules/llm_response_generator.py
"""
LLM response generator that accepts conversation history (list of role/text dicts)
and RAG hits, composes a prompt/messages, calls external LLM API (configurable),
and returns a single text response (Arabic), or None on failure.
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any

logger = logging.getLogger("llm_response")

# Environment controls:
# GPT_OSS_API_URL - full URL to call
# GPT_OSS_API_KEY - optional bearer token
# GPT_OSS_API_SCHEMA - "openai" or "generic" (default generic)
# GPT_OSS_MODEL - optional model name if using openai schema

def _build_system_message():
    return "أنت وكيل خدمة عملاء مصري باللهجة المصرية. كن ودودًا، مختصرًا، ومهذبًا. أجب باللغة العربية المصرية."

def _format_history_as_messages(history: List[Dict[str, Any]]):
    """
    Convert stored history entries to LLM messages list (role/message)
    history entries expected: {'role': 'user'|'assistant', 'text': '...', 'meta': {...}}
    Returns list suitable for OpenAI-like chat APIs.
    """
    msgs = []
    for item in history:
        role = item.get("role", "user")
        text = item.get("text", "")
        if role not in ("system", "user", "assistant"):
            role = "user"
        msgs.append({"role": role, "content": text})
    return msgs

def call_llm_for_response(history: List[Dict[str, Any]], rag_hits: List[Any], call_sid: str = "web", intent: str = None, max_tokens: int = 150) -> str | None:
    """
    history: list of {'role','text',...}
    rag_hits: list of (doc, score) or similar from Retriever
    Returns: string response in Arabic, or None if no LLM output
    """
    api_url = os.getenv("GPT_OSS_API_URL")
    api_key = os.getenv("GPT_OSS_API_KEY")
    if not api_url:
        # No remote LLM configured — return None so pipeline falls back to RAG or DM reply
        logger.debug("No GPT_OSS_API_URL configured; skipping LLM call.")
        return None

    # Build RAG context text
    rag_context = ""
    if rag_hits:
        try:
            lines = []
            for hit in rag_hits:
                # hit can be (doc, score) or (doc) depending on retriever
                if isinstance(hit, tuple) and len(hit) >= 1:
                    doc = hit[0]
                else:
                    doc = hit
                text = doc.get("text") if isinstance(doc, dict) else str(doc)
                lines.append(text)
            rag_context = "\n".join(lines)
        except Exception as e:
            logger.exception("Failed building rag context: %s", e)
            rag_context = ""

    schema = os.getenv("GPT_OSS_API_SCHEMA", "generic").lower()
    model_name = os.getenv("GPT_OSS_MODEL")

    # Build messages
    messages = []
    system_msg = _build_system_message()
    messages.append({"role": "system", "content": system_msg})

    # Append history
    messages.extend(_format_history_as_messages(history))

    # Add RAG context as user assistant instruction
    if rag_context:
        messages.append({"role": "system", "content": f"المعرفة المرجعية:\n{rag_context}"})

    # Prompt to produce concise, helpful Arabic response based on last user utterance
    # Optionally include intent as hint
    intent_hint = f" (نوع الطلب: {intent})" if intent else ""
    messages.append({"role": "user", "content": f"بناءً على المحادثة أعلاه{intent_hint}، أجب بإيجاز باللهجة المصرية:"})

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        if schema == "openai":
            payload = {
                "model": model_name or "gpt-oss-120b",
                "messages": messages,
                "max_tokens": max_tokens,
            }
        else:
            # generic servers often accept `input` as a single string; join messages
            joined = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            payload = {"input": joined, "max_tokens": max_tokens}

        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        # flexible parsing of response JSON
        try:
            j = resp.json()
        except Exception:
            logger.warning("LLM returned non-json response; using text")
            return resp.text.strip() if resp.text else None

        # openai-like parsing
        if schema == "openai":
            # try common locations
            if isinstance(j.get("choices"), list) and j["choices"]:
                c = j["choices"][0]
                # chat completions
                if isinstance(c, dict) and c.get("message") and isinstance(c["message"], dict):
                    msg = c["message"].get("content")
                    if msg:
                        return msg.strip()
                # text completions
                if isinstance(c, dict) and c.get("text"):
                    return c["text"].strip()
        else:
            # generic schema: try common fields
            for k in ("output", "text", "result", "answer"):
                if k in j and isinstance(j[k], str):
                    return j[k].strip()
            # choices fallback
            if isinstance(j.get("choices"), list) and j["choices"]:
                c = j["choices"][0]
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    return c["text"].strip()

        # If nothing extracted, fallback to whole json string (but we prefer None so upstream can fallback)
        logger.warning("Unexpected LLM response format: %s", j)
        return None

    except Exception as e:
        logger.exception("LLM API call failed: %s", e)
        return None