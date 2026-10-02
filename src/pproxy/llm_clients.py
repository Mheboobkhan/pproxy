"""Thin wrappers around the two model hops used by the experiment.

    ollama_generate  -> local model, does the rephrasing (never leaves the box)
    openai_complete  -> remote model, produces the answer under test

Both return a (text, meta) pair. `meta` is recorded verbatim in
response.jsonl so that latency and token counts are available at analysis
time without a second run.
"""

import time

import requests

from . import config


class LLMError(RuntimeError):
    """Raised when a hop fails after all retries are exhausted."""


def _with_retries(fn, *, attempts=None, base_delay=None):
    """Call `fn`, retrying transient failures with exponential backoff."""
    attempts = attempts or config.MAX_RETRIES
    base_delay = base_delay or config.RETRY_BASE_DELAY
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - surfaced as LLMError below
            last_exc = exc
            if attempt == attempts - 1:
                break
            time.sleep(base_delay * (2 ** attempt))
    raise LLMError(f"{fn.__name__ if hasattr(fn, '__name__') else 'call'} "
                   f"failed after {attempts} attempts: {last_exc}") from last_exc


# --------------------------------------------------------------------------
# local hop
# --------------------------------------------------------------------------
def ollama_generate(prompt, *, model=None, system=None):
    """Run a prompt against the local Ollama server."""
    model = model or config.OLLAMA_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE},
    }
    if system:
        payload["system"] = system

    def _call():
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    started = time.perf_counter()
    data = _with_retries(_call)
    elapsed = time.perf_counter() - started

    text = (data.get("response") or "").strip()
    if not text:
        raise LLMError(f"ollama returned an empty response for model {model!r}")

    meta = {
        "provider": "ollama",
        "model": model,
        "latency_s": round(elapsed, 3),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
    }
    return text, meta


# --------------------------------------------------------------------------
# remote hop
# --------------------------------------------------------------------------
_openai_client = None


def _client(remote=None):
    if remote is not None:
        remote.validate()
        from openai import OpenAI
        return OpenAI(api_key=remote.api_key, base_url=remote.base_url,
                      timeout=config.REQUEST_TIMEOUT, max_retries=0)
    global _openai_client
    if _openai_client is None:
        if not config.OPENAI_API_KEY:
            raise LLMError("Set OPENAI_API_KEY or HF_TOKEN")
        from openai import OpenAI  # imported lazily so baseline-only runs
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY,  # need no SDK
                                base_url=config.OPENAI_BASE_URL)
    return _openai_client


def openai_complete(prompt, *, model=None, system=None, remote=None):
    """Send the (possibly transformed) query to the remote model."""
    model = model or config.OPENAI_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    client = _client(remote)

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )

    started = time.perf_counter()
    resp = _with_retries(_call)
    elapsed = time.perf_counter() - started

    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    meta = {
        "provider": remote.provider if remote else config.REMOTE_PROVIDER,
        "base_url": remote.base_url if remote else config.OPENAI_BASE_URL or "https://api.openai.com/v1",
        "model": model,
        "latency_s": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "finish_reason": resp.choices[0].finish_reason,
    }
    return text, meta

# SUGGESTION: _with_retries currently retries every exception, including a
# 401 from a bad key -- three attempts and a 6s wait to learn the key is
# wrong. Worth splitting retryable (timeout, 429, 5xx) from fatal (401, 400)
# once the happy path is confirmed working.
