"""Remote hop on Claude, via the official Anthropic SDK.

Kept separate from llm_clients.py so the OpenAI arm is untouched. Same
contract as the other hops: returns (text, meta), with meta recorded
verbatim in the output file.
"""

import time

import anthropic

import config
from llm_clients import LLMError

_client = None


def _get_client():
    global _client
    if _client is None:
        # SDK retries 408/409/429/5xx and connection errors itself; auth and
        # bad-request errors fail immediately.
        _client = anthropic.Anthropic(max_retries=config.MAX_RETRIES)
    return _client


def claude_complete(prompt, *, model=None, system=None):
    """Send the (possibly transformed) query to Claude."""
    model = model or config.CLAUDE_MODEL
    kwargs = {}
    if system:
        kwargs["system"] = system

    started = time.perf_counter()
    try:
        # Use the standard Messages API across supported Claude models.
        resp = _get_client().messages.create(
            model=model,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
    except anthropic.APIError as exc:
        raise LLMError(f"claude request failed: {exc}") from exc
    elapsed = time.perf_counter() - started

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    meta = {
        "provider": "anthropic",
        "model": model,
        "served_model": resp.model,
        "request_id": getattr(resp, "_request_id", None),
        "latency_s": round(elapsed, 3),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "stop_reason": resp.stop_reason,
    }
    if resp.stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        meta["refusal_category"] = getattr(details, "category", None)
        raise LLMError(f"claude refused (category={meta['refusal_category']})")
    if not text:
        raise LLMError(f"claude returned no text (stop_reason={resp.stop_reason})")
    return text, meta
