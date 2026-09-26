"""Central configuration for pproxy experiment runs.

Everything tunable lives here so that a run is reproducible from env vars
alone -- no edits to the experiment code between conditions.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- data files -----------------------------------------------------------
QUERIES_PATH = Path(os.getenv("PPROXY_QUERIES", ROOT / "queries.jsonl"))
RESPONSES_PATH = Path(os.getenv("PPROXY_RESPONSES", ROOT / "response.jsonl"))

# --- local rephraser (Ollama) --------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# --- remote answerer (OpenAI) --------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
