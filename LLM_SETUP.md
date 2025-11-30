# Voice AI Agent - LLM-Enhanced Pipeline Setup

## What's New

The pipeline now uses an LLM to generate human-like customer support responses. After the Dialogue Manager produces a response, the LLM is called with:
- Customer transcript
- Detected intent
- Dialogue Manager's initial reply
- Top-K relevant documents from the knowledge base

The LLM then generates a professional, friendly Arabic response suitable for TTS.

## Architecture

```
Audio → ASR → NLU (Model) → Dialogue Manager → RAG (Retriever) + LLM → TTS → Output
                                              ↓
                                         LLM enriches response
                                         using context + KB
```

## Environment Configuration

### NLU Model (Auto-loaded)
- Automatically loads from `models/nlu/` if present
- Env var: `NLU_MODEL_DIR` (optional override)

### RAG Retriever (Auto-loaded)
- Automatically loads KB from `models/rag/docs.json` if present
- Falls back to in-memory `KB_SMALL`

### LLM API Configuration

To enable LLM response generation, set these environment variables:

```bash
# Required:
GPT_OSS_API_URL=http://localhost:8000/v1/completions
# or for OpenAI-compatible:
GPT_OSS_API_URL=http://localhost:8000/v1/chat/completions

# Optional:
GPT_OSS_API_KEY=your-api-key
GPT_OSS_API_SCHEMA=generic        # or 'openai'
GPT_OSS_MODEL=gpt-oss-120b
RAG_TOP_K=3                        # number of KB docs to pass to LLM
```

### API Schema Support

**Generic Schema (default)**
- Endpoint receives: `{"input": "prompt...", "max_tokens": 300, "temperature": 0.7}`
- Response fields checked: `output`, `text`, `result`, `answer`, `generated_text`

**OpenAI-Compatible Schema**
- Set `GPT_OSS_API_SCHEMA=openai`
- Endpoint receives: `{"model": "...", "messages": [...], "max_tokens": 300}`
- Parses: `choices[0].message.content` or `choices[0].text`

## Example LLM Responses

The LLM generates responses following these patterns:

### Greeting + Order Request
**Input:** "اهلا و سهلا عايزه اعمل اوردر"
```
أهلاً وسهلاً بيكي في مطعم Test & Go يا فندم!
منوّرانا دايماً!
ممكن اتشرف بإسم حضرتك؟
```

### Order Confirmation with Prices
**Input:** "باستا الفريدو و فرينش فرايز و كولا"
```
تمام يا فندم، تم تسجيل طلب حضرتِك:

الأوردر بتاعك:
• باستا ألفريدو — 95 جنيه
• فرينش فرايز — 25 جنيه
• كولا — 20 جنيه

الإجمالي: 140 جنيه

تحبّي الأوردر للدليفري ولا تيك أواي؟
```

### Refund Policy
**Input:** "سياسة الاسترجاع"
```
معلش يا فندم على اللي حصل!
تقدري تسترجعي الفلوس خلال 24 ساعة لو الطلب فيه مشكلة.
ممكن تقوليلي الحاصل بالظبط علشان نساعدك؟
```

## Testing

### Test Prompt Generation
```bash
python test_llm_prompt.py
```
This validates that prompts are correctly formatted for the LLM without requiring an API endpoint.

### Test RAG Behavior
```bash
pytest tests/test_rag_behavior.py -v
```

### Live Server Test
1. Ensure server is running: `python app2.py`
2. Open http://localhost:5050 in browser
3. Record audio or use `/process-audio-direct` endpoint with a WAV file

## Key Files

- `modules/llm_response_generator.py` — LLM prompt building and API calls
- `modules/nlu.py` — Auto-loads trained NLU model from `models/nlu/`
- `modules/rag.py` — Auto-loads KB from `models/rag/docs.json`, improved retrieval
- `app2.py` — Main pipeline with LLM integration
- `tests/test_rag_behavior.py` — Unit tests for RAG
- `test_rag.py` — Quick RAG validation
- `test_llm_prompt.py` — Prompt format validation

## How It Works

### 1. Request Flow
```
Request → ASR (audio→text) → NLU (intent) → DM (template) 
       → RAG (retrieve KB) + LLM (enrich response) → TTS → Response
```

### 2. LLM Enrichment
- RAG retrieves top-K documents matching the user query
- LLM receives: system prompt (role definition), user prompt (context + KB + intent)
- LLM generates professional, friendly Arabic response
- If LLM fails, falls back to RAG extraction or DM reply

### 3. Response Quality
- Tone: Formal, friendly, professional customer support
- Format: Plain Arabic text (no JSON, no special chars)
- Length: Typically 50-200 tokens (fits TTS well)
- Language: Egyptian Arabic (casual, natural)

## Troubleshooting

### LLM Not Being Called
- Check `GPT_OSS_API_URL` is set and reachable
- Check logs for: `LLM API call failed: ...`
- If not set, falls back to DM reply + RAG extraction

### Wrong Prices/Items
- LLM prefers documents with token overlap from the query
- If retrieval is poor, increase `RAG_TOP_K` or improve KB entries
- Test with: `python test_rag.py`

### Encoding Issues
- All code uses UTF-8 for Arabic
- Windows console may need UTF-8 mode: `chcp 65001`

## Environment File Example (.env)

```
# NLU
NLU_MODEL_DIR=./models/nlu

# RAG
RAG_TOP_K=3

# LLM (GPT-OSS 120B)
GPT_OSS_API_URL=http://localhost:8000/v1/completions
GPT_OSS_API_KEY=your-key-here
GPT_OSS_API_SCHEMA=generic
GPT_OSS_MODEL=gpt-oss-120b

# TTS
ELEVEN_API_KEY=your-eleven-labs-key

# Twilio (optional)
TWILIO_ACCOUNT_SID=your-sid
TWILIO_AUTH_TOKEN=your-token
TWILIO_FROM_NUMBER=+1234567890
BASE_URL=https://your-ngrok-url.ngrok.io
```

## Next Steps

1. Set up your GPT-OSS 120B server endpoint
2. Configure `GPT_OSS_API_URL` and schema in `.env`
3. Restart `app2.py`
4. Test with `python test_llm_prompt.py` to verify prompt formatting
5. Monitor logs for `[call_sid] Using LLM-enhanced response` to confirm LLM is active
