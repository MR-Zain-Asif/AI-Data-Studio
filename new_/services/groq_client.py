"""DataStudio AI — Groq API client with fallback handling."""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from config.settings import GROQ_MODEL_HEAVY, GROQ_MODEL_FAST

load_dotenv()

def _get_groq_client():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None


def _get_candidate_models(fast: bool = False) -> list[str]:
    primary = GROQ_MODEL_FAST if fast else GROQ_MODEL_HEAVY
    defaults = [primary, "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
    seen = set()
    result = []
    for m in defaults:
        if m and m not in seen:
            seen.add(m)
            result.append(m)

    client = _get_groq_client()
    if client:
        try:
            models = client.models.list()
            for m in models.data:
                mid = m.id.lower()
                if any(k in mid for k in ["llama", "mixtral", "gemma", "qwen"]):
                    if not any(k in mid for k in ["whisper", "guard", "canopylabs", "orpheus", "tts", "audio", "vision"]):
                        if m.id not in seen:
                            seen.add(m.id)
                            result.append(m.id)
        except Exception:
            pass

    return result


def call_groq(system: str, user: str, fast: bool = False, max_tokens: int = 1500) -> str:
    """Call Groq LLM with given system and user prompts. Returns text string."""
    client = _get_groq_client()
    if not client:
        return "Groq API key not configured in .env. Please configure GROQ_API_KEY to enable AI chat."

    candidate_models = _get_candidate_models(fast=fast)
    last_error = None

    for model in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            continue

    return f"AI temporarily unavailable: {str(last_error)}"


def call_groq_json(system: str, user: str, max_tokens: int = 2000) -> dict:
    """Call Groq with JSON response format. Returns parsed dict."""
    client = _get_groq_client()
    if not client:
        return {}

    candidate_models = _get_candidate_models(fast=False)

    for model in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system + "\nReturn ONLY valid JSON. No markdown wrappers, no extra text before or after."},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                cleaned = raw.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except Exception:
                continue
        except Exception:
            continue

    return {"error": "All available Groq models failed"}
