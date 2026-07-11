"""Pytest bootstrap.

Make the src/ package importable so `pytest` works straight from a checkout, even
without an editable install.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(autouse=True)
def _fresh_file_seen():
    """Isolate the per-session freshness bookkeeping (tools._FILE_SEEN, plan 073)
    between tests — a leftover seen-hash from one test must not make another test's
    line edit look stale."""
    from chad import tools
    tools._FILE_SEEN.clear()
    yield
    tools._FILE_SEEN.clear()
