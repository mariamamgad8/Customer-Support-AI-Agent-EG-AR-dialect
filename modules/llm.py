from dotenv import load_dotenv
load_dotenv()

import os
import requests

class AzureLLM:
    def __init__(self):
        self.endpoint = os.getenv("HF_INFERENCE_ENDPOINT")
        self.api_key = os.getenv("HF_INFERENCE_API_KEY")
        if not self.endpoint or not self.api_key:
            raise RuntimeError("Azure endpoint or API key missing")
        self.headers = {
            "api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def generate(self, prompt, max_new_tokens=128, temperature=0.0):
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens)
        }
        resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict) and "choices" in j:
            choices = j.get("choices") or []
            if len(choices) > 0:
                first = choices[0]
                msg = first.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"].strip()
                if "content" in first:
                    return str(first["content"]).strip()
        return str(j)

def get_llm():
    return AzureLLM()
