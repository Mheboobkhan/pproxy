"""Central configuration for pproxy experiment runs.

Everything tunable lives here so that a run is reproducible from env vars
alone -- no edits to the experiment code between conditions.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent

# --- data files -----------------------------------------------------------
QUERIES_PATH = Path(os.getenv("PPROXY_QUERIES", ROOT / "queries.jsonl"))
RESPONSES_PATH = Path(os.getenv("PPROXY_RESPONSES", ROOT / "response.jsonl"))

# --- local rephraser (Ollama) --------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# --- remote answerer (OpenAI-compatible, including Hugging Face) ----------
# Accept a native HF token, or an HF token in the legacy OPENAI_API_KEY slot.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
_hf_token = bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("hf_"))
# Any OpenAI-compatible endpoint (Groq, OpenRouter, Hugging Face router,
# Gemini). None means api.openai.com.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or (
    "https://router.huggingface.co/v1" if _hf_token else None
)
REMOTE_PROVIDER = (
    "huggingface" if urlparse(OPENAI_BASE_URL or "").hostname == "router.huggingface.co"
    else "openai" if not OPENAI_BASE_URL
    else "openai-compatible"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or (
    os.getenv("HF_MODEL") or "openai/gpt-oss-120b"
    if REMOTE_PROVIDER == "huggingface" else "gpt-4o-mini"
)

# --- remote answerer (Claude) --------------------------------------------
# Credentials come from ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), read by
# the SDK itself -- nothing to configure here.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
# Claude Opus 5 thinks by default and thinking tokens count against
# max_tokens, so the 512 used for the OpenAI arm would truncate answers.
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "16000"))
PROD_PATH = Path(os.getenv("PPROXY_PROD", ROOT / "prod.jsonl"))

# --- sampling -------------------------------------------------------------
# Temperature 0 on BOTH hops. The pre-registered comparison in PREREG.md is
# between transformation modes; sampling noise would otherwise be confounded
# with the effect being measured.
TEMPERATURE = 0.0
MAX_TOKENS = 512

# --- transport ------------------------------------------------------------
REQUEST_TIMEOUT = int(os.getenv("PPROXY_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("PPROXY_RETRIES", "3"))
RETRY_BASE_DELAY = 2.0  # seconds; doubled per attempt

# SUGGESTION: record the exact model build in response.jsonl, not just the
# model name. Ollama tags like "llama3.2" are mutable -- if the tag is
# repulled mid-study the runs are no longer comparable. `ollama show
# --modelfile <tag>` gives a digest worth pinning in the paper.
