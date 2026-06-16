#!/usr/bin/env python3
"""CLI wrapper for the text extraction stage."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from local_rag.extraction.extract_text import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
