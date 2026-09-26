#!/usr/bin/env python3
"""Rewrite stage only -- local model, nothing leaves the machine.

Reads queries.jsonl, asks the local Ollama model to rephrase each query, and
writes one record per query holding BOTH the original and the rewrite so the
two can be compared directly (span survival, length drift, leakage).

No remote call is made and the openai package is never imported, so this
runs with only `requests` installed and no OPENAI_API_KEY set.

Usage
-----
    python rewrite.py                      # all queries -> response.jsonl
    python rewrite.py --limit 3            # smoke test on the first 3
    python rewrite.py --out rewrites.jsonl # keep separate from run output
    python rewrite.py --model mistral      # override the local model
"""

import argparse
import datetime as _dt
import json
import sys

import config
import transform
from llm_clients import LLMError

# Marks these records as rewrite-only, since by default they land in the same
# response.jsonl that pproxy.py appends full pipeline runs to. Filter on this
# field (or use --out) to keep the two apart at analysis time.
STAGE = "rewrite_only"


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def load_queries(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from None
    return rows


def load_done(path):
    """Ids already rewritten successfully, so a re-run can resume."""
    done = set()
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return done
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") == STAGE and not row.get("error"):
                done.add(row.get("id"))
    return done


def rewrite_one(row):
    """Rephrase a single query locally and build its output record."""
    record = {
        "id": row.get("id"),
        "stage": STAGE,
        "intent_label": row.get("intent_label"),
        "query": row.get("query"),            # original, unchanged
        "rewritten_query": None,              # what the local model produced
        # carried through so span survival can be scored without a join back
        # to queries.jsonl
        "pii_spans": row.get("pii_spans", []),
        "rewriter_meta": None,
        "error": None,
        "timestamp": _now(),
    }
    try:
        text, meta = transform.rephrase(row["query"])
        record["rewritten_query"] = text
        record["rewriter_meta"] = meta.get("rephraser")
    except (LLMError, KeyError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="rewrite",
        description="Rewrite every query with the local model only.",
    )
    p.add_argument("--queries", default=None, help="path to queries.jsonl")
    p.add_argument("--out", default=None, help="output jsonl (default response.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="only the first N queries")
    p.add_argument("--model", default=None, help="override OLLAMA_MODEL")
    p.add_argument("--overwrite", action="store_true",
                   help="redo ids already rewritten in the output file")
    args = p.parse_args(argv)

    if args.model:
        config.OLLAMA_MODEL = args.model

    queries_path = args.queries or config.QUERIES_PATH
    out_path = args.out or config.RESPONSES_PATH

    rows = load_queries(queries_path)
    if args.limit:
        rows = rows[:args.limit]
    done = set() if args.overwrite else load_done(out_path)
    todo = [r for r in rows if r.get("id") not in done]

    print(f"{len(rows)} queries, {len(done)} already rewritten, {len(todo)} to do")
    print(f"local model: {config.OLLAMA_MODEL} at {config.OLLAMA_HOST}")

    failures = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {row.get('id')} ... ", end="", flush=True)
            record = rewrite_one(row)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()  # an interrupted run loses at most the row in flight
            if record["error"]:
                failures += 1
                print(f"ERROR {record['error']}")
            else:
                print(f"ok ({len(record['rewritten_query'])} chars)")

    print(f"done: {len(todo) - failures} ok, {failures} failed -> {out_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

# SUGGESTION: this file is the cheap inner loop for H1 -- it costs nothing
# per run and touches no remote API, so it is the right place to iterate on
# REPHRASE_PROMPT in transform.py before spending OpenAI calls.
#
# SUGGESTION: once a batch exists, the first thing worth checking is how
# often the local model returned the query unchanged. A rewrite identical to
# the original means zero privacy gain but full utility, and it should be
# reported as its own category rather than averaged into the results.
#
# SUGGESTION: the obvious follow-up script is one that reads these records
# and checks, per row, which pii_spans values still appear verbatim in
# rewritten_query. That single number is most of what H1 claims.
