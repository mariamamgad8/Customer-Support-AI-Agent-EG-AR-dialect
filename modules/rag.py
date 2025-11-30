# modules/rag.py
# Simple RAG: SBERT retriever over the small KB and a local LLM fallback.
import os
import json
import logging
import requests
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("rag")

KB_SMALL = [
    {
        "title": "أسعار المينيو",
        "content": [
            "باستا ألفريدو: 95 جنيه",
            "شاورما فراخ: 75 جنيه",
            "شاورما لحمة: 85 جنيه",
        ],
    },
    {
        "title": "معلومات التوصيل",
        "content": [
            "سعر التوصيل: 20 جنيه داخل طنطا فقط",
            "التوصيل خلال 30 لـ 45 دقيقة",
        ],
    },
    {
        "title": "سياسة الاسترجاع",
        "content": [
            "تقدر تسترجع الفلوس خلال 24 ساعة لو الطلب فيه مشكلة",
            "لو الأكل مش مطابق للوصف، تواصل مع خدمة العملاء فوراً",
        ],
    },
]

EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_DEFAULT_STORE = os.path.join(os.getcwd(), "models", "embeddings")


class Retriever:
    def __init__(self, embed_model=EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model)
        self.documents = []
        self.doc_embeddings = None

    def fit(self, kb_list):
        docs = []
        for idx, item in enumerate(kb_list):
            title = item.get("title", "")
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    docs.append(
                        {
                            "id": idx,
                            "title": title,
                            "text": f"{title} — {c}" if title else c,
                        }
                    )
            else:
                docs.append(
                    {
                        "id": idx,
                        "title": title,
                        "text": f"{title} — {content}" if title else content,
                    }
                )
        self.documents = docs
        texts = [d["text"] for d in docs]
        self.doc_embeddings = self.embedder.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )

    def retrieve(self, query, top_k=2):
        if self.doc_embeddings is None:
            raise RuntimeError("Retriever not fit yet")
        q = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q, self.doc_embeddings, top_k=top_k)[0]
        results = [(self.documents[h["corpus_id"]], float(h["score"])) for h in hits]
        return results


def load_or_build_retriever(kb=None):
    # If a models/rag/docs.json exists in the workspace, prefer it
    default_docs = os.path.join(os.getcwd(), "models", "rag", "docs.json")
    if os.path.exists(default_docs):
        try:
            with open(default_docs, "r", encoding="utf-8") as fh:
                docs = json.load(fh)
            # docs.json expected to be a list of {id,title,text}
            kb_list = []
            for item in docs:
                title = item.get("title") or ""
                text = item.get("text") or item.get("content") or ""
                # normalize into expected format used by Retriever.fit
                kb_list.append({"title": title, "content": [text]})
            r = Retriever()
            r.fit(kb_list)
            return r
        except Exception as e:
            logger.exception(f"Failed loading docs.json for RAG: {e}")

    kb = kb or KB_SMALL
    r = Retriever()
    r.fit(kb)
    return r


# Simple LLM fallback: for dev we return extracted snippet, or a template
def simple_llm_answer(context, question):
    # If context contains numeric currency, return that; else short template
    import re

    m = re.search(r"(\d{1,6}\s*(?:جنيه|جنيهات|EGP|ج\.م))", context)
    if m:
        return m.group(1)
    # fallback: pick the most relevant line
    lines = [line for line in context.split("\n") if line.strip()]
    return lines[0] if lines else "محتاج أعرف أكتر يا فندم"


def call_llm_api(context: str, question: str, max_tokens: int = 150):
    """Call an external LLM API (configurable via env vars).

    Expected envs:
      - GPT_OSS_API_URL: full URL of the model inference endpoint
      - GPT_OSS_API_KEY: optional API key to send as Authorization Bearer

    The function is intentionally flexible about the response JSON shape.
    """
    api_url = os.getenv("GPT_OSS_API_URL")
    api_key = os.getenv("GPT_OSS_API_KEY")
    if not api_url:
        return None

    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely in Arabic:"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Support multiple common API schemas via GPT_OSS_API_SCHEMA env var
    schema = os.getenv("GPT_OSS_API_SCHEMA", "generic").lower()
    model_name = os.getenv("GPT_OSS_MODEL")

    if schema == "openai":
        # OpenAI-compatible chat format
        payload = {
            "model": model_name or "gpt-oss-120b",
            "messages": [
                {"role": "system", "content": "You are a helpful Arabic assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }
    else:
        # Generic schema (many OSS servers accept `input`)
        payload = {"input": prompt, "max_tokens": max_tokens}

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        # flexible parsing
        try:
            j = resp.json()
        except Exception:
            return resp.text.strip()

        # common fields to check
        for k in ("output", "text", "result", "answer"):
            if k in j and isinstance(j[k], str):
                return j[k].strip()
        # openai-like choices
        if isinstance(j.get("choices"), list) and j["choices"]:
            c = j["choices"][0]
            if isinstance(c, dict) and c.get("message") and isinstance(c["message"], dict):
                # Chat completions
                msg = c["message"].get("content")
                if msg:
                    return msg.strip()
            if isinstance(c, dict) and c.get("text"):
                return c["text"].strip()
        # fallback to whole json
        return json.dumps(j, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"LLM API call failed: {e}")
        return None


def rag_answer_or_fallback(query, retriever: Retriever, top_k=2):
    hits = retriever.retrieve(query, top_k=top_k)
    if not hits:
        return "محتاج أعرف أكتر يا فندم"

    # Build context from hits
    context_text = "\n".join([doc["text"] for doc, _ in hits])

    # If an external LLM API is configured, call it with the context + question
    llm_resp = call_llm_api(context_text, query)
    if llm_resp:
        return llm_resp

    # If LLM is not available, pick the best matching hit using token overlap + embedding score
    import re

    # tokenize query (Arabic-aware regexp and latin words)
    q_tokens = [tok for tok in re.findall(r"[\w\u0600-\u06FF]+", query) if len(tok) > 1]

    best = None
    best_score = -1.0
    for (doc, emb_score) in hits:
        text = doc.get("text", "")
        # count token overlaps
        overlap = sum(1 for tok in q_tokens if tok in text)
        # combine embedding score and overlap (weights tuned empirically)
        combined = (overlap * 2.0) + float(emb_score)
        if combined > best_score:
            best_score = combined
            best = text

    # Try to extract numeric currency from the best doc first
    m = re.search(r"(\d{1,6}\s*(?:جنيه|جنيهات|EGP|ج\.م))", best or context_text)
    if m:
        return m.group(1)

    # If best exists, return it; otherwise try other fallbacks
    if best:
        return best

    # Fallback: pick the most relevant line
    lines = [line for line in context_text.split("\n") if line.strip()]
    return lines[0] if lines else "محتاج أعرف أكتر يا فندم"
