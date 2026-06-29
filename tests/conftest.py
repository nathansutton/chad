"""Pytest bootstrap.

Make the src/ package importable so `pytest` works straight from a checkout, even
without an editable install.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
