"""Backwards-compatible shim for the bidmate-compare CLI.

The real implementation lives in :mod:`bidmate_rag.cli.compare`. New usage
should prefer the ``bidmate-compare`` console script.
"""

from __future__ import annotations

from bidmate_rag.cli.compare import main

if __name__ == "__main__":
    main()
