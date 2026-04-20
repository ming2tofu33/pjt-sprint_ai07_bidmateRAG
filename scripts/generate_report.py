"""Backwards-compatible shim for the report CLI.

The real implementation lives in :mod:`bidmate_rag.cli.report`. New usage
should prefer the ``bidmate-report`` console script.
"""

from __future__ import annotations

from bidmate_rag.cli.report import main

if __name__ == "__main__":
    main()
