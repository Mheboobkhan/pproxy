"""Resolve cloud settings without mixing credentials between providers."""

import os
from dataclasses import dataclass

from . import config
from .llm_clients import LLMError


@dataclass(frozen=True)
class Remote:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None

    def validate(self):
        if self.provider == "claude":
            if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
                raise LLMError("Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN for Claude")
        elif not self.api_key:
            variable = "HF_TOKEN" if self.provider == "huggingface" else "OPENAI_API_KEY"
            raise LLMError(f"Set {variable} for {self.provider}")
        elif self.provider == "openai" and self.api_key.startswith("hf_"):
            raise LLMError("A Hugging Face token cannot authenticate to OpenAI; use --provider huggingface")

    def complete(self, prompt):
        if self.provider == "claude":
            from .claude_client import claude_complete
            return claude_complete(prompt, model=self.model)
        from .llm_clients import openai_complete
        return openai_complete(prompt, model=self.model, remote=self)


def resolve(provider=None, model=None):
    provider = provider or os.getenv("PPROXY_PROVIDER") or config.REMOTE_PROVIDER
    if provider == "claude":
        return Remote(provider, model or config.CLAUDE_MODEL)
    if provider == "huggingface":
        key = os.getenv("HF_TOKEN")
        legacy_key = os.getenv("OPENAI_API_KEY", "")
        if not key and legacy_key.startswith("hf_"):
            key = legacy_key
        return Remote(provider, model or os.getenv("HF_MODEL") or "openai/gpt-oss-120b",
                      key, "https://router.huggingface.co/v1")
    if provider == "openai":
        return Remote(provider, model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
                      os.getenv("OPENAI_API_KEY"), "https://api.openai.com/v1")
    if provider == "openai-compatible":
        return Remote(provider, model or config.OPENAI_MODEL,
                      config.OPENAI_API_KEY, config.OPENAI_BASE_URL)
    raise ValueError(f"Unknown provider: {provider}")
