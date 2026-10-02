#!/usr/bin/env python3
"""Full pipeline on Claude: local rewrite -> Claude answer -> prod.jsonl.

For each query in queries.jsonl the LOCAL Ollama model rephrases it, only the
rephrasing is sent to Claude, and one record is written holding the original
query, the local rewrite, and Claude's response.

Usage
-----
    python -m src.pproxy.prod                        # all queries -> prod.jsonl
    python -m src.pproxy.prod --limit 3              # smoke test on the first 3
    python -m src.pproxy.prod --model llama3.1:8b    # override the local model
    python -m src.pproxy.prod --claude-model claude-sonnet-5

Requires ANTHROPIC_API_KEY and `ollama serve` running locally.
"""

import argparse
import datetime as _dt
import json
import os
import sys

from . import config
from . import transform
from .claude_client import claude_complete
from .llm_clients import LLMError
from .rewrite import load_queries

MODE = "rephrase"


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def load_done(path):
    """Ids already answered successfully, so a re-run can resume."""
    done = set()
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return done
    with fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # blank or half-written line from an interrupted run
            if not row.get("error"):
                done.add(row.get("id"))
    return done


def run_one(row):
    record = {
        "id": row.get("id"),
        "mode": MODE,
        "query_id": None,
        "intent_label": row.get("intent_label"),
        "query": row.get("query"),       # original, never sent to Claude
        "rewritten_query": None,         # local model output, sent to Claude
        "response": None,                # Claude's answer
        "pii_spans": row.get("pii_spans", []),
        "rewriter_meta": None,
        "response_meta": None,
        "error": None,
        "timestamp": _now(),
    }
    try:
        sent, tmeta = transform.rephrase(row["query"])
        record["rewritten_query"] = sent
        record["rewriter_meta"] = tmeta.get("rephraser")
        record["query_id"] = tmeta.get("query_id")
        answer, rmeta = claude_complete(sent)
        record["response"] = answer
        record["response_meta"] = rmeta
    except (LLMError, KeyError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main(argv=None):
    p = argparse.ArgumentParser(prog="prod", description=__doc__.splitlines()[0])
    p.add_argument("--queries", default=None, help="path to queries.jsonl")
    p.add_argument("--out", default=None, help="output jsonl (default prod.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="only the first N queries")
    p.add_argument("--model", default=None, help="override OLLAMA_MODEL (local)")
    p.add_argument("--claude-model", default=None, help="override CLAUDE_MODEL")
    p.add_argument("--overwrite", action="store_true",
                   help="redo ids already answered in the output file")
    args = p.parse_args(argv)

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        print("ANTHROPIC_API_KEY is not set; export it first", file=sys.stderr)
        return 2

    if args.model:
        config.OLLAMA_MODEL = args.model
    if args.claude_model:
        config.CLAUDE_MODEL = args.claude_model

    out_path = args.out or config.PROD_PATH
    rows = load_queries(args.queries or config.QUERIES_PATH)
    if args.limit:
        rows = rows[:args.limit]
    done = set() if args.overwrite else load_done(out_path)
    todo = [r for r in rows if r.get("id") not in done]

    print(f"{len(rows)} queries, {len(done)} already done, {len(todo)} to do")
    print(f"local: {config.OLLAMA_MODEL} at {config.OLLAMA_HOST} -> "
          f"remote: {config.CLAUDE_MODEL}")

    failures = 0
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {row.get('id')} ... ", end="", flush=True)
            record = run_one(row)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()  # an interrupted run loses at most the row in flight
            if record["error"]:
                failures += 1
                print(f"ERROR {record['error']}")
            else:
                print(f"ok ({record['response_meta']['output_tokens']} tokens)")

    print(f"done: {len(todo) - failures} ok, {failures} failed -> {out_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
