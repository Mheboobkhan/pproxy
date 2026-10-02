# pproxy

pproxy is a privacy-aware proxy between a user's request and a cloud LLM. In
rephrase mode, a local LLM rewrites the request to remove or generalize sensitive
details while preserving what the user needs help with. Only the rewritten
request is sent to the cloud model. An optional local synthesis step then adapts
the cloud's answer using the original request, keeping that original context
local throughout the rephrase workflow. The project helps evaluate how well this
approach balances privacy and useful answers; sensitive-detail removal depends
on the local model's rewrite and is not guaranteed.

Compare original prompts (`baseline`) with prompts rewritten by local Ollama
(`rephrase`) before sending them to a cloud model. Baseline mode sends the
original prompt to the cloud for comparison.

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
python -m src.pproxy --provider openai --limit 1 --out openai-smoke.jsonl

# Requires HF_TOKEN (legacy OPENAI_API_KEY=hf_... also works)
python -m src.pproxy --provider huggingface --limit 1 --out hf-smoke.jsonl

# Requires ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
python -m src.pproxy --provider claude --limit 1 --out claude-smoke.jsonl
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
python -m src.pproxy --provider huggingface --modes rephrase --limit 5 \
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

```text
src/pproxy/       Runtime package: CLI, configuration, transforms, runner, clients
data/            Source dataset (queries.jsonl)
docs/            Research notes and TODO checklist
tests/           Local development tests (ignored by Git)
outputs/         Generated responses and samples (ignored by Git)
pproxy.py         Compatibility launcher for existing commands
requirements.txt  Runtime dependencies
```

Run commands from the repository root. `python -m src.pproxy` is the main
entry point; `python pproxy.py` also works. Standalone pipelines run with
`python -m src.pproxy.rewrite` and `python -m src.pproxy.prod`.
The default dataset is `data/queries.jsonl`, and default results go into
`outputs/`. Custom `--queries` and `--out` paths are relative to your working
directory. Existing local results and random samples have moved into `outputs/`.

## TODO

- [ ] **H1 — Prompt abstraction:** evaluate sensitive-detail removal and answer
  utility for the implemented local rephrase mode against baseline.
- [ ] **H2 — Approach based on the Shokari paper:** identify the exact paper,
  define the transformation and comparison criteria, and implement the mode.
- [ ] **H3 — Prompt chunking and response synthesis:** implement chunking,
  record chunk boundaries, and synthesize the cloud responses. The existing
  optional local synthesis step currently handles a single cloud response.
- [ ] **H4 — K-1 pseudo conversation:** define the conversation construction
  and evaluation procedure, then implement it as a selectable mode.
