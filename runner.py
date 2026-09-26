"""Drive every query in queries.jsonl through every mode, append results.

Results are appended to response.jsonl one line at a time, flushed after
each write: a crash or Ctrl-C partway through a run loses at most the row in
flight, and --resume picks up from there.
"""

import datetime as _dt
import json
import sys

import config
import transform
from llm_clients import LLMError


def load_queries(path=None):
    """Read queries.jsonl, returning a list of dicts."""
    path = path or config.QUERIES_PATH
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


def load_done(path=None):
    """Return the set of (id, mode) pairs already present in response.jsonl."""
    path = path or config.RESPONSES_PATH
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
                continue  # a half-written line from an interrupted run
            if row.get("error") is None:
                done.add((row.get("id"), row.get("mode")))
    return done


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def run_one(row, mode, *, send=True):
    """Transform one query, send it onward, and build the output record."""
    from llm_clients import openai_complete  # local import keeps dry-runs
                                             # free of the openai dependency
    record = {
        "id": row.get("id"),
        "mode": mode,
        "intent_label": row.get("intent_label"),
        "query": row.get("query"),
        # carried through so scoring can join spans to responses without
        # re-reading queries.jsonl
        "pii_spans": row.get("pii_spans", []),
        "sent_query": None,
        "response": None,
        "transform_meta": None,
        "response_meta": None,
        "error": None,
        "timestamp": _now(),
    }

    try:
        sent, tmeta = transform.transformer(row["query"], mode)
        record["sent_query"] = sent
        record["transform_meta"] = tmeta
        if send:
            answer, rmeta = openai_complete(sent)
            record["response"] = answer
            record["response_meta"] = rmeta
    except (LLMError, ValueError, KeyError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def run_all(modes=None, *, limit=None, resume=True, send=True,
            queries_path=None, out_path=None, log=print):
    """Run every (query, mode) pair and append each result as it completes."""
    modes = tuple(modes or transform.DEFAULT_MODES)
    out_path = out_path or config.RESPONSES_PATH
    rows = load_queries(queries_path)
    if limit:
        rows = rows[:limit]
    done = load_done(out_path) if resume else set()

    planned = [(r, m) for r in rows for m in modes
               if (r.get("id"), m) not in done]
    log(f"{len(rows)} queries x {len(modes)} modes; "
        f"{len(done)} already done, {len(planned)} to run")

    failures = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, (row, mode) in enumerate(planned, 1):
            log(f"[{i}/{len(planned)}] {row.get('id')} {mode} ... ", end="")
            record = run_one(row, mode, send=send)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()  # survive an interrupted run
            if record["error"]:
                failures += 1
                log(f"ERROR {record['error']}")
            else:
                log("ok")

    log(f"done: {len(planned) - failures} ok, {failures} failed -> {out_path}")
    return failures

# SUGGESTION: run_all is sequential, so 39 queries x 2 modes is ~78 remote
# calls back to back. That is fine for correctness and keeps rate limits out
# of the picture, but if it gets slow, parallelise across QUERIES (not
# modes) -- the modes for one query should stay adjacent in time so that any
# drift in the remote model affects both arms equally.
#
# SUGGESTION: queries.jsonl currently holds 39 rows, not 40 -- q033 is
# missing from the id sequence. Worth filling before the real run so the
# legal class is balanced at 8 like the others.
