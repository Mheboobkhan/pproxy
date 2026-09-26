#!/usr/bin/env python3
"""pproxy -- privacy proxy experiment harness.

Takes every query in queries.jsonl, applies a transformation mode, sends the
result to the remote model, and writes one record per (query, mode) to
response.jsonl.

    baseline   query is sent to the remote model unchanged
    rephrase   a LOCAL Ollama model rewrites the query first; only the
               rewrite reaches the remote model

Usage
-----
    python pproxy.py                          # both modes, all queries
    python pproxy.py --modes baseline         # one arm only
    python pproxy.py --limit 3 --dry-run      # transform only, no API calls
    python pproxy.py --overwrite              # ignore existing response.jsonl

Requires OPENAI_API_KEY in the environment and `ollama serve` running
locally for the rephrase mode.
"""

import argparse
import sys

import config
import transform
from runner import run_all
from transform import transformer  # re-exported: original entry point

__all__ = ["transformer", "main"]


def build_parser():
    p = argparse.ArgumentParser(
        prog="pproxy",
        description="Run queries.jsonl through transformation modes.",
    )
    p.add_argument(
        "--modes", nargs="+", default=list(transform.DEFAULT_MODES),
        choices=sorted(transform.MODES),
        help="modes to run (default: baseline rephrase)",
    )
    p.add_argument("--queries", default=None, help="path to queries.jsonl")
    p.add_argument("--out", default=None, help="path to response.jsonl")
    p.add_argument("--limit", type=int, default=None,
                   help="only run the first N queries (smoke test)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-run pairs already present in response.jsonl")
    p.add_argument("--dry-run", action="store_true",
                   help="apply transforms but make no remote calls")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.dry_run and not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set; export it or pass --dry-run",
              file=sys.stderr)
        return 2

    failures = run_all(
        modes=args.modes,
        limit=args.limit,
        resume=not args.overwrite,
        send=not args.dry_run,
        queries_path=args.queries,
        out_path=args.out,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

# SUGGESTION: --overwrite appends a second copy of each pair rather than
# truncating response.jsonl, so the file can end up holding two records for
# the same (id, mode). Analysis should take the LAST record per pair, or the
# flag should truncate -- worth deciding before the real run.
#
# SUGGESTION: the study measures what the remote model sees, so it is worth
# asserting that the raw query never reaches it in rephrase mode. A cheap
# check: assert sent_query != query for every rephrase record, and inspect
# any row where the local model echoed the input back verbatim.
