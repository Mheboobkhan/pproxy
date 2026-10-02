"""Compatibility launcher; implementation lives in src/pproxy."""

from src.pproxy.cli import main, transformer

if __name__ == "__main__":
    raise SystemExit(main())
