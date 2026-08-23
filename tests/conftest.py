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
def _checkpoint_tmpdir(tmp_path, monkeypatch):
    """Point edit-checkpoint shadow repos (checkpoint._history_root) at a per-test
    tmp dir. Any test that drives run_turn through a file-mutating tool takes a
    real snapshot — of the pytest CWD, into the developer's ~/.chad, at ~300ms
    per shot — unless redirected here."""
    monkeypatch.setenv("CHAD_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))


@pytest.fixture(autouse=True)
def _spill_tmpdir(tmp_path, monkeypatch):
    """Point bash-output spills (tools._spill_bash) at a per-test tmp dir —
    any test that runs an oversized bash command would otherwise leave files in the
    user's real ~/.cache/chad/spill."""
    monkeypatch.setenv("CHAD_SPILL_DIR", str(tmp_path / "spill"))


@pytest.fixture(autouse=True)
def _fresh_ambient(monkeypatch):
    """Isolate ambient session state between tests, and stub the
    env-manifest builder: every Agent/prompt construction would otherwise spawn
    a dozen real `--version` subprocesses per test. test_ambient.py un-stubs it
    explicitly to test the real builder."""
    from chad import ambient
    ambient.reset()
    monkeypatch.setattr(ambient, "_build_manifest",
                        lambda: "- present: python3 3.11 · git 2.43")
    yield
    ambient.reset()



