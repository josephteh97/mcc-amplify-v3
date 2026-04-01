"""
Thin wrapper around the local Ollama HTTP API for SEA-LION vision queries.
"""

import base64
import time
import json
import requests

# ── CONFIGURATION ────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:11434"
MODEL = "aisingapore/Gemma-SEA-LION-v4-4B-VL:latest"
MAX_RETRIES = 3
BACKOFF_BASE = 2   # seconds
TIMEOUT      = 300 # seconds — vision inference on a floor plan can take a while


# ── STARTUP CHECK ────────────────────────────────────────────────────────────

_model_verified = False


def _verify_model():
    """Raise RuntimeError if the SEA-LION model tag is not present in ollama list.
    Only checks once per process lifetime — subsequent calls are no-ops.
    Called lazily on first query so import failures surface with context.
    """
    global _model_verified
    if _model_verified:
        return
    try:
        resp = requests.get(f"{BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
        if not any(MODEL in t or t in MODEL for t in tags):
            raise RuntimeError(
                f"SEA-LION model not found. Run: ollama pull {MODEL}\n"
                f"Available models: {tags}"
            )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {BASE_URL}. "
            "Make sure Ollama is running: ollama serve"
        )
    _model_verified = True


# ── INTERNAL HELPERS ─────────────────────────────────────────────────────────

def _chat(payload: dict) -> str:
    """Send a POST /api/chat request with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{BASE_URL}/api/chat",
                json=payload,
                timeout=TIMEOUT,
            )
            if not resp.ok:
                detail = resp.text[:300] if resp.text else "(no body)"
                raise requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code} from Ollama: {detail}", response=resp
                )

            # stream=False → single JSON object
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Ollama request failed after {MAX_RETRIES} attempts: {exc}")
            time.sleep(BACKOFF_BASE ** attempt)


def _query(prompt: str, image_path: str = None) -> str:
    _verify_model()
    message = {"role": "user", "content": prompt}
    if image_path is not None:
        with open(image_path, "rb") as f:
            message["images"] = [base64.b64encode(f.read()).decode("utf-8")]
    return _chat({"model": MODEL, "messages": [message], "stream": False})


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def query_vision(image_path: str, prompt: str) -> str:
    """Send an image + text prompt to SEA-LION and return the text response."""
    return _query(prompt, image_path=image_path)


def query_text(prompt: str) -> str:
    """Send a text-only prompt to SEA-LION and return the text response."""
    return _query(prompt)
