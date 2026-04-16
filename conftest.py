"""
Root conftest — ensures the project root is on sys.path so tests can use
absolute imports like `from shared.core.hmm ...` regardless of whether
pytest is invoked via `uv run pytest` or `python -m pytest`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
