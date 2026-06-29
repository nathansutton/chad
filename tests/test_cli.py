"""Characterization tests for cli.py's model-resolution + env parsing — the bootstrap
path that currently has ZERO direct coverage. All model-free: we never call main()
(which loads MLX and reads argv); we drive the pure helpers `_env_int` and `_pick_model`,
monkeypatching `_detect_ram_gb` (it shells out to sysctl) and `os.path.isdir`.

A bug in `_pick_model` silently picks the wrong model size, or downloads a 12 GB repo
on a box that can't run it; a bug in `_env_int` mis-parses an advanced knob. These pin
the current contract so a refactor can't drift it.

Run: `uv run python test_cli.py`
"""

import os

import pytest

from chad import cli

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def test_env_int(monkeypatch):
    # unset -> None
    monkeypatch.delenv("CHAD_X_INT", raising=False)
    check("unset -> None", cli._env_int("CHAD_X_INT") is None)
    # a valid integer string parses
    monkeypatch.setenv("CHAD_X_INT", "131072")
    check("valid int parses", cli._env_int("CHAD_X_INT") == 131072)
    # "0" is a non-empty string -> parses to 0 (NOT treated as falsy/None)
    monkeypatch.setenv("CHAD_X_INT", "0")
    check("zero parses to 0", cli._env_int("CHAD_X_INT") == 0)
    # empty string is falsy -> None (the `if val` short-circuit)
    monkeypatch.setenv("CHAD_X_INT", "")
    check("empty string -> None", cli._env_int("CHAD_X_INT") is None)
    # CONTRACT: a non-numeric value raises ValueError (int() does); the code does NOT
    # guard it — documenting current behavior, not endorsing it.
    monkeypatch.setenv("CHAD_X_INT", "not-a-number")
    raised = False
    try:
        cli._env_int("CHAD_X_INT")
    except ValueError:
        raised = True
    check("non-numeric raises ValueError", raised)


def test_pick_model_override(monkeypatch):
    # An explicit CHAD_MODEL wins outright, regardless of RAM or local dirs, and the
    # reason mentions the override.
    monkeypatch.setenv("CHAD_MODEL", "/some/local/model")
    # even with isdir/ram set to surprising values the override must short-circuit first
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 8.0)
    model, why = cli._pick_model()
    check("override returns CHAD_MODEL value", model == "/some/local/model", model)
    check("override reason mentions override", "override" in why.lower(), why)


def test_pick_model_ram_thresholds(monkeypatch):
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    # No local dirs anywhere -> HF repo ids.
    monkeypatch.setattr(os.path, "isdir", lambda p: False)

    # RAM below the big-box threshold -> 9B HF repo (the safe fallback).
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 16.0)
    model, why = cli._pick_model()
    check("low RAM -> 9B HF repo", model == cli._HF_9B, model)
    check("low RAM reason mentions 9B", "9B" in why, why)

    # RAM at/above the threshold -> 35B HF repo (the default).
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    model, why = cli._pick_model()
    check("big RAM -> 35B HF repo", model == cli._HF_35B, model)
    check("big RAM reason mentions 35B", "35B" in why, why)

    # RAM unreadable (None) -> falls through to the big/35B branch (does NOT crash).
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: None)
    model, _ = cli._pick_model()
    check("unknown RAM -> 35B HF repo", model == cli._HF_35B, model)


def test_pick_model_prefers_local_dir(monkeypatch):
    monkeypatch.delenv("CHAD_MODEL", raising=False)

    # Small box AND the locally-built 9B dir exists -> use the local path, not the repo.
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 16.0)
    monkeypatch.setattr(os.path, "isdir", lambda p: p == cli._LOCAL_9B)
    model, _ = cli._pick_model()
    check("local 9B preferred over HF repo", model == cli._LOCAL_9B, model)

    # Big box AND the locally-built 35B dir exists -> use the local path.
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    monkeypatch.setattr(os.path, "isdir", lambda p: p == cli._LOCAL_35B)
    model, _ = cli._pick_model()
    check("local 35B preferred over HF repo", model == cli._LOCAL_35B, model)


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as mp:
        test_env_int(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_override(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_ram_thresholds(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_prefers_local_dir(mp)
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
