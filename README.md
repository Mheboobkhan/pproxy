# pproxy

Compare original prompts (`baseline`) with prompts rewritten by local Ollama
(`rephrase`) before sending them to a cloud model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
```

Keep Ollama running (`ollama serve`) for rephrase mode.

Export your provider's credential in the terminal before running:
`OPENAI_API_KEY`, `HF_TOKEN`, or `ANTHROPIC_API_KEY`. Environment variables
are read directly; `.env` files are not loaded automatically.

Choose a cloud provider in `pproxy`:

```bash
# Requires OPENAI_API_KEY
python pproxy.py --provider openai --limit 1 --out openai-smoke.jsonl

# Requires HF_TOKEN (legacy OPENAI_API_KEY=hf_... also works)
python pproxy.py --provider huggingface --limit 1 --out hf-smoke.jsonl

# Requires ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
python pproxy.py --provider claude --limit 1 --out claude-smoke.jsonl
```

Claude uses the Anthropic Messages API; this option does not invoke the Claude
Code CLI or use a Claude Code subscription.

Use `--model MODEL_ID` to override the cloud model. Defaults are `gpt-4o-mini`
for OpenAI, `openai/gpt-oss-120b` for Hugging Face, and `CLAUDE_MODEL`
(default `claude-opus-5`) for Claude. Provider environment model overrides are
`OPENAI_MODEL`, `HF_MODEL`, and `CLAUDE_MODEL`, respectively.

`--provider` overrides `PPROXY_PROVIDER`. Without either, the existing endpoint
and credential detection chooses the OpenAI-compatible provider. Explicit
OpenAI and Hugging Face options select their respective endpoints independently
of `OPENAI_BASE_URL`; automatic selection still supports custom compatible URLs.

By default both modes run. Use `--modes rephrase` to send only rewritten prompts,
or `--modes baseline` to avoid needing Ollama. `--dry-run` makes no cloud calls
and needs no cloud credentials. Remove `--limit 1` for the full dataset.

Output is append-only JSONL. Successful cloud records are resumed by query ID,
mode, provider, and model. Dry-run records do not skip later cloud calls.
`--overwrite` reruns records and appends duplicates; use a new output file for
each fresh experiment. Originals and sensitive-detail annotations remain in
the local output alongside rewrites, answers, and request metadata.

## Optional local final answer

```bash
python pproxy.py --provider huggingface --modes rephrase --limit 5 \
  --local-synthesis --out hf-final-answers.jsonl
```

`--local-synthesis` adds an Ollama call after each successful rephrase cloud
response. The local model receives the original query, rewritten query, and
cloud response, and adapts the answer to the original user's context. Baseline
answers stay as returned by the cloud. Without the flag, no synthesis call runs.
Dry runs only transform prompts and do not synthesize answers.

Each rewrite has a `query_id`: Unix timestamp in nanoseconds plus the SHA-256
hash of the original query's first three whitespace-separated words. Repeated
rewrites get new IDs. The identifier stays in local records and synthesis input;
only the rewritten text is sent to the cloud. The dataset's `id` is retained.
Rewrite-only and `prod.py` records also carry the identifier.

`response` preserves the cloud answer; `final_response` holds the local final
answer when synthesis is enabled, otherwise the cloud answer. `synthesis_meta`
records local request metadata. A synthesis failure retains the cloud response,
leaves the final response empty, and records `synthesis_error` and `error`.
Runs with and without synthesis are resumed separately. Retrying a failed
record reruns the entire pipeline, including the cloud request.

## Repository layout

- `pproxy.py`: main CLI for provider selection and experiment modes.
- `transform.py`: local rewriting, tracking identifiers, and final synthesis.
- `providers.py`, `llm_clients.py`, `claude_client.py`: cloud and local clients.
- `runner.py`, `config.py`: batch execution, resume handling, and settings.
- `rewrite.py`, `prod.py`: standalone rewrite-only and Claude pipelines.
- `queries.jsonl`: annotated source dataset.
- `PREREG.md`, `TODO.md`: research hypotheses and outstanding work.

## TODO

Hypotheses are recorded in [PREREG.md](PREREG.md). The current baseline and
local abstraction/rephrase modes provide the starting implementation for H1;
the hypotheses still need evaluation.

- [ ] **H1 — Prompt abstraction:** evaluate sensitive-detail removal and answer
  utility for the implemented local rephrase mode against baseline.
- [ ] **H2 — Approach based on the Shokari paper:** identify the exact paper,
  define the transformation and comparison criteria, and implement the mode.
- [ ] **H3 — Prompt chunking and response synthesis:** implement chunking,
  record chunk boundaries, and synthesize the cloud responses. The existing
  optional local synthesis step currently handles a single cloud response.
- [ ] **H4 — K-1 pseudo conversation:** define the conversation construction
  and evaluation procedure, then implement it as a selectable mode.

## Local files

Generated JSONL outputs and random samples, development test scripts, virtual
environments, Python caches, `.env` files, and editor files are ignored by Git. `queries.jsonl`
remains tracked as the source dataset.
