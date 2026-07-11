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

    # RAM unreadable (None) -> the SAFE smaller model, never a surprise 12 GB
    # download on unknown hardware (devex review T3: the old fall-through to the
    # 35B branch contradicted _detect_ram_gb's "safe (smaller) model" contract).
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: None)
    model, why = cli._pick_model()
    check("unknown RAM -> safe 9B HF repo", model == cli._HF_9B, model)
    check("unknown RAM reason says undetectable", "undetectable" in why, why)


def test_model_download_gb():
    check("35B repo -> ~12 GB", cli._model_download_gb(cli._HF_35B) == 12.0)
    check("9B repo -> ~5 GB", cli._model_download_gb(cli._HF_9B) == 5.0)


def test_free_disk_gb():
    # A real, existing path reports a plausible number.
    free = cli._free_disk_gb("~")
    check("home free-disk readable", free is not None and free > 0, free)
    # A nonexistent deep path climbs to an existing parent instead of crashing.
    free = cli._free_disk_gb("~/definitely/not/a/real/dir/blobs")
    check("missing path climbs to parent", free is not None and free > 0, free)


def test_ensure_model_disk_preflight(monkeypatch, capsys):
    """Devex review T2: a machine without room for the download must be refused
    BEFORE the download starts, with the shortfall and the cache-GC command named."""
    import huggingface_hub
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: None)
    monkeypatch.setattr(cli, "_free_disk_gb", lambda path: 1.0)
    with pytest.raises(SystemExit) as e:
        cli._ensure_model(cli._HF_9B)
    check("preflight exits 1", e.value.code == 1, e.value.code)
    err = capsys.readouterr().err
    check("names the shortfall", "not enough free disk" in err, err)
    check("names required space", "~5 GB" in err, err)
    check("points at cache GC", "hf cache" in err, err)


def test_ensure_model_disk_preflight_unreadable(monkeypatch):
    """If free disk can't be read the preflight must NOT block (it guards, never
    gates): the flow proceeds to the confirm prompt / download attempt."""
    import huggingface_hub
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    # Cache hit short-circuits before any prompt — proves we got PAST the preflight.
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: None)
    monkeypatch.setattr(cli, "_free_disk_gb", lambda path: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    with pytest.raises(SystemExit) as e:
        cli._ensure_model(cli._HF_9B)
    # Exit came from the user's "n" at the prompt, not the disk preflight.
    check("unreadable disk does not block", e.value.code == 1, e.value.code)


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


def test_ram_aware_ctx_limit():
    GB = 1e9
    # Measured 24 GB M4 Pro numbers (scripts/kv_ram_probe.py): 19.07 GB working set,
    # 12.06 GB resident after load, 20,578 B/token, 262 k window. Expect a trigger well
    # above the old 120 k cap but below the window.
    n = cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578,
                                reserve_gb=1.5, safety=0.90)
    check("24GB box: above old 120k cap", n > 120_000, n)
    check("24GB box: below the window", n < 262144 - 2048, n)
    check("24GB box: in the measured ~175k range", 150_000 < n < 200_000, n)

    # Tight box (less working set) compacts sooner — strictly smaller window.
    tight = cli.ram_aware_ctx_limit(262144, 10.0 * GB, 8.0 * GB, 20578)
    check("tight box compacts sooner", tight < n, (tight, n))

    # Huge box is capped at the model window minus the gen margin, never above it.
    huge = cli.ram_aware_ctx_limit(262144, 400.0 * GB, 12.0 * GB, 20578)
    check("huge box capped at window-margin", huge == 262144 - 2048, huge)

    # Degenerate inputs -> None so the caller keeps the old fixed cap.
    check("no KV cost -> None", cli.ram_aware_ctx_limit(262144, 19 * GB, 12 * GB, 0) is None)
    check("no budget -> None", cli.ram_aware_ctx_limit(262144, 0, 12 * GB, 20578) is None)

    # Reserve eats into the budget: a bigger scratch reserve -> a smaller window.
    big_reserve = cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578,
                                          reserve_gb=4.0)
    check("bigger reserve shrinks window", big_reserve < n, (big_reserve, n))

    # Over-subscribed (model already past the safe budget) -> floor, never negative.
    floored = cli.ram_aware_ctx_limit(262144, 14 * GB, 18 * GB, 20578)
    check("over-subscribed -> floor", floored == 8192, floored)


def test_env_float(monkeypatch):
    monkeypatch.delenv("CHAD_X_F", raising=False)
    check("unset float -> None", cli._env_float("CHAD_X_F") is None)
    monkeypatch.setenv("CHAD_X_F", "2.5")
    check("float parses", cli._env_float("CHAD_X_F") == 2.5)
    monkeypatch.setenv("CHAD_X_F", "")
    check("empty float -> None", cli._env_float("CHAD_X_F") is None)


def test_version_flag(monkeypatch, capsys):
    # argparse's `version` action prints to stdout and exits 0 during parse_args(),
    # BEFORE _preflight() ever runs — so --version works even off Apple Silicon.
    monkeypatch.setattr("sys.argv", ["chad", "--version"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    check("--version exits 0", exc.value.code == 0, repr(exc.value.code))
    out = capsys.readouterr().out
    check("--version prints chad 0.1.0", out.startswith("chad 0.1.0"), out)


def test_preflight_skips_apple_gate_for_remote_backends(monkeypatch):
    # The remote backends load no MLX, so _preflight must NOT hard-stop on a non-Apple
    # host — that's what lets chad run inside a Linux benchmark container against a remote
    # server. Simulate a Linux/x86 box and assert llama/openai pass while mlx would exit.
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
    cli._preflight("llama")   # must return, not exit
    cli._preflight("openai")  # must return, not exit
    with pytest.raises(SystemExit) as exc:
        cli._preflight("mlx")
    check("mlx backend still gated off Apple Silicon", exc.value.code == 1)


def test_version_string_never_raises(monkeypatch):
    # The commit detail is best-effort: if distribution metadata is unreadable the
    # helper must still return a plain "chad <version>" string, never propagate.
    def boom(*_a, **_k):
        raise RuntimeError("no metadata")
    monkeypatch.setattr("importlib.metadata.distribution", boom)
    s = cli._version_string()
    check("still a string", isinstance(s, str), repr(s))
    check("starts with chad ", s.startswith("chad "), s)


def test_home_dir_note_written_in_home(monkeypatch, capsys):
    # Plan 060: launching in ~ prints a one-line nudge to cd into a project — no exit,
    # no behavior change. chad snapshots the cwd, so home is rarely the intended dir.
    home = os.path.expanduser("~")
    monkeypatch.setattr(os, "getcwd", lambda: home)
    cli._maybe_home_dir_note()
    err = capsys.readouterr().err
    check("home-dir note written", "home directory" in err, err)


def test_home_dir_note_absent_in_project(monkeypatch, capsys):
    # A real project dir (not ~) gets no note — home-dir only, no marker-file guessing.
    monkeypatch.setattr(os, "getcwd", lambda: "/Users/x/some/project")
    cli._maybe_home_dir_note()
    err = capsys.readouterr().err
    check("no note outside home", err == "", repr(err))


if __name__ == "__main__":
    test_ram_aware_ctx_limit()
    with pytest.MonkeyPatch.context() as mp:
        test_env_float(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_env_int(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_override(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_ram_thresholds(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_prefers_local_dir(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_version_string_never_raises(mp)
    print(f"\n{PASS} passed, {FAIL} failed")
    # Note: the home-dir note tests need pytest's capsys fixture; run them via `pytest`.
    raise SystemExit(1 if FAIL else 0)
