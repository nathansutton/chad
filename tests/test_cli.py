"""Characterization tests for cli.py's model-resolution + env parsing — the bootstrap
path that currently has ZERO direct coverage. All model-free: we never call main()
(which loads MLX and reads argv); we drive the pure helpers `_env_int` and `_pick_model`,
monkeypatching `_detect_ram_gb` (it shells out to sysctl) and `os.path.isdir`.

A bug in `_pick_model` silently picks the wrong model size, or downloads a 12 GB repo
on a box that can't run it; a bug in `_env_int` mis-parses an advanced knob. These pin
the current contract so a refactor can't drift it.

Run: `uv run python tests/test_cli.py`
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
    # reason says the choice was requested rather than defaulted.
    monkeypatch.setenv("CHAD_MODEL", "/some/local/model")
    # even with isdir/ram set to surprising values the override must short-circuit first
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 8.0)
    model, why = cli._pick_model()
    check("override returns CHAD_MODEL value", model == "/some/local/model", model)
    check("override reason says requested", "requested" in why.lower(), why)
    # `--model` (the `spec` argument) outranks CHAD_MODEL: the flag is the more specific
    # signal, and a shell that exports CHAD_MODEL globally must not pin every run.
    model, _ = cli._pick_model("/flag/model")
    check("--model beats CHAD_MODEL", model == "/flag/model", model)
    # `--model auto` is the explicit spelling of "ignore the override, use RAM".
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    model, why = cli._pick_model("auto")
    check("auto falls through to the RAM default", model == cli._HF_9B, model)


def test_pick_model_aliases(monkeypatch):
    """`--model 9b` / `--model 35b` are the discoverable spelling of the shipped pair —
    the point of the flag is that choosing small-or-large needs no repo id."""
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)   # no local build -> HF repos
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 8.0)  # tiny box: alias must win
    for spec, expect in (("9b", cli._HF_9B), ("35b", cli._HF_35B),
                         ("9B", cli._HF_9B), ("35B", cli._HF_35B)):
        model, why = cli._pick_model(spec)
        check(f"--model {spec} -> shipped repo", model == expect, model)
        check(f"--model {spec} reason says requested", "requested" in why, why)
    # Anything not an alias is passed through untouched (HF repo id or local dir).
    model, _ = cli._pick_model("mlx-community/Whatever-4bit")
    check("unknown spec passes through", model == "mlx-community/Whatever-4bit", model)


def test_pick_model_ram_thresholds(monkeypatch):
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    # No local dirs anywhere -> HF repo ids.
    monkeypatch.setattr(os.path, "isdir", lambda p: False)

    # RAM below the big-box threshold -> 9B HF repo (the safe fallback).
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 16.0)
    model, why = cli._pick_model()
    check("low RAM -> 9B HF repo", model == cli._HF_9B, model)
    check("low RAM reason mentions 9B", "9B" in why, why)

    # A 24 GB Mac gets the 35B. It did NOT before the fused attention kernel and the
    # 8-bit-from-the-start KV cache cut the per-token cost — the floor was 32 GB and a
    # 24 GB box silently got the small model. This is the case the threshold exists for,
    # so it is pinned rather than left to the 16/64 pair either side of it.
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 24.0)
    model, why = cli._pick_model()
    check("24 GB -> 35B HF repo", model == cli._HF_35B, model)

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


def test_pick_model_flag_auto_ignores_env(monkeypatch):
    # '--model auto' forces the RAM pick even when CHAD_MODEL is set: the env must NOT
    # win. Big box + no local dirs -> the 35B default, not the env value.
    monkeypatch.setenv("CHAD_MODEL", "/env/repo")
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    model, why = cli._pick_model("auto")
    check("--model auto ignores env, RAM-picks 35B", model == cli._HF_35B, model)
    check("--model auto reason is a default", "default" in why, why)


def test_pick_model_flag_9b_both_ram(monkeypatch, capsys):
    # Forcing 9B yields 9B on a big box AND a small box, and NEVER warns (9B is always
    # safe). No local dirs -> HF repo.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    for ram in (16.0, 64.0):
        monkeypatch.setattr(cli, "_detect_ram_gb", lambda ram=ram: ram)
        model, why = cli._pick_model("9b")
        check(f"--model 9b -> 9B at {ram} GB", model == cli._HF_9B, model)
        check("--model 9b reason names source", "--model" in why, why)
    check("forcing 9B never warns", capsys.readouterr().err == "")


def test_pick_model_flag_35b_big_no_warn(monkeypatch, capsys):
    # Forcing 35B on a big box is exactly what the RAM pick would do -> no warning.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    model, why = cli._pick_model("35b")
    check("--model 35b -> 35B HF repo", model == cli._HF_35B, model)
    check("35B on big box does not warn", capsys.readouterr().err == "")


def test_pick_model_flag_35b_small_warns(monkeypatch, capsys):
    # Forcing 35B under the RAM threshold still honors the override BUT warns (2A): the
    # harness advises, the caller decides.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 16.0)
    model, _ = cli._pick_model("35b")
    check("--model 35b honored on small box", model == cli._HF_35B, model)
    err = capsys.readouterr().err
    check("small-box 35B warns", "35b forced" in err, err)
    check("warning names the OOM risk", "OOM" in err, err)


def test_pick_model_flag_35b_ram_none_warns(monkeypatch, capsys):
    # RAM undetectable + forced 35B: warn (we can't vouch for the memory) and proceed.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: None)
    model, _ = cli._pick_model("35b")
    check("--model 35b honored on unknown RAM", model == cli._HF_35B, model)
    check("unknown-RAM 35B warns 'undetectable'",
          "undetectable" in capsys.readouterr().err)


def test_pick_model_flag_repo_passthrough(monkeypatch):
    # A non-alias value is a literal repo id / local dir, passed through unchanged (the
    # CLI twin of CHAD_MODEL). RAM is irrelevant.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 8.0)
    model, why = cli._pick_model("/some/local/model")
    check("--model repo passthrough", model == "/some/local/model", model)
    check("passthrough reason names source + override",
          "--model" in why and "override" in why.lower(), why)


def test_pick_model_flag_beats_env(monkeypatch):
    # Both set -> the CLI flag wins over CHAD_MODEL.
    monkeypatch.setenv("CHAD_MODEL", "/env/repo")
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    model, why = cli._pick_model("9b")
    check("--model beats CHAD_MODEL", model == cli._HF_9B, model)
    check("winner reason names --model", "--model" in why, why)


def test_pick_model_flag_prefers_local_dir(monkeypatch):
    # A forced size prefers the locally-built dir, same as the RAM path.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    monkeypatch.setattr(os.path, "isdir", lambda p: p == cli._LOCAL_35B)
    model, _ = cli._pick_model("35b")
    check("forced 35B prefers local dir", model == cli._LOCAL_35B, model)


def test_pick_model_default_equals_auto(monkeypatch):
    # Regression lock for bench.py's no-arg callers: _pick_model() must behave exactly
    # like _pick_model("auto") for a fixed environment.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    check("no-arg == auto", cli._pick_model() == cli._pick_model("auto"))


def test_ram_aware_ctx_limit():
    GB = 1e9
    # Measured 24 GB M4 Pro numbers: 19.07 GB working set, 12.06 GB resident after
    # load, 20,578 B/token KV, 262 k window. The divisor is kv × slope_factor (1.75,
    # the 2026-07-12 ram_safety_check all-in fit: peak grows 35.7 KB/token, not the
    # KV-only 20.5) — the raw-KV pick of ~175k extrapolated to 102.9% of budget.
    n = cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578,
                                reserve_gb=1.5, safety=0.90)
    check("24GB box: below the window", n < 262144 - 2048, n)
    check("24GB box: in the safe ~100k range", 90_000 < n < 110_000, n)
    # The measured worst case (peak ≈ active + 1.0 GB + 35.7 KB/tok × (ctx + 8k gen))
    # must stay under the working-set budget at the picked trigger.
    worst = 12.06 * GB + 1.0 * GB + 35_700 * (n + 8192)
    check("24GB box: worst-case peak under budget", worst < 19.07 * GB, worst / GB)
    # slope_factor=1.0 recovers the old raw-KV behavior (env override escape hatch).
    raw = cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578,
                                  slope_factor=1.0)
    check("slope_factor=1.0 recovers raw-KV pick", 150_000 < raw < 200_000, raw)

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

    # Host physical pressure: when the host's reclaimable band is
    # tighter than the Metal band, IT binds — Docker/harbor pressure the Metal
    # budget cannot see must shrink the window.
    baseline = cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578)
    pressured = cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578,
                                        host_avail_bytes=3.0 * GB)
    check("tight host band binds below the Metal band", pressured < baseline,
          (pressured, baseline))
    check("tight host band still floored, never negative",
          cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578,
                                  host_avail_bytes=0.5 * GB) == 8192)
    # A roomy host band changes nothing — the Metal band stays the binding one.
    roomy = cli.ram_aware_ctx_limit(262144, 19.07 * GB, 12.06 * GB, 20578,
                                    host_avail_bytes=200 * GB)
    check("roomy host band leaves the Metal result", roomy == baseline,
          (roomy, baseline))


def test_host_avail_bytes():
    # Live probe on macOS: returns a plausible positive number (or None on a
    # platform without vm_stat — not this CI). Bounds, not exact values.
    got = cli._host_avail_bytes()
    check("host avail probe returns bytes", got is None or got > 100e6, got)


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
    from chad import __version__
    check(f"--version prints chad {__version__}",
          out.startswith(f"chad {__version__}"), out)


def test_preflight_skips_apple_gate_for_remote_backend(monkeypatch):
    # The remote backend loads no MLX, so _preflight must NOT hard-stop on a non-Apple
    # host — that's what lets chad run inside a Linux benchmark container against a remote
    # server. Simulate a Linux/x86 box and assert llama passes while mlx would exit.
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
    cli._preflight("llama")   # must return, not exit
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
    # Launching in ~ prints a one-line nudge to cd into a project — no exit,
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


# --- sampler env: one call, so the knobs cannot drift apart ---------------

def test_apply_sampler_env_sets_all_three(monkeypatch):
    """They used to be three inlined sibling blocks in main(), which is why `chad serve`
    — building its own engine — silently ran without ANY of them. One function now, so a
    caller cannot honor temp and forget min_p."""
    class E:
        temp = 0.0
        min_p = 0.0
        top_p = 0.0
    monkeypatch.setenv("CHAD_TEMP", "0.7")
    monkeypatch.setenv("CHAD_MIN_P", "0.05")
    monkeypatch.setenv("CHAD_TOP_P", "0.95")
    e = E()
    cli.apply_sampler_env(e)
    assert (e.temp, e.min_p, e.top_p) == (0.7, 0.05, 0.95)


def test_apply_sampler_env_ignores_junk_and_leaves_unset_knobs_alone(monkeypatch):
    class E:
        temp = 0.3
        min_p = 0.02
        top_p = 0.9
    monkeypatch.setenv("CHAD_TEMP", "hot")
    monkeypatch.delenv("CHAD_MIN_P", raising=False)
    monkeypatch.delenv("CHAD_TOP_P", raising=False)
    e = E()
    cli.apply_sampler_env(e)
    assert (e.temp, e.min_p, e.top_p) == (0.3, 0.02, 0.9)


def test_serve_applies_the_same_sampler_env_as_the_local_cli():
    """The point of `chad serve` is that a container measures the model people actually
    run. A server that ignored CHAD_MIN_P would sample differently from a local chad
    started with it — the exact drift this server exists to eliminate."""
    import inspect

    from chad import serve
    assert "apply_sampler_env" in inspect.getsource(serve.run)


# --- subcommand dispatch ------------------------------------------------------
# `serve` / `prove` / `levers` are matched against argv[0] rather than routed through
# argparse subparsers: the default invocation's positional is a free-form task string,
# and a subparser layout would either shadow it or force `chad -- "some task"`. These
# pin both halves of that rule — the bare word dispatches, the same word inside a
# sentence stays a task — because getting it wrong silently swallows real work.

def test_subcommand_dispatch_matches_only_the_bare_word():
    parse = cli._agent_parser().parse_args
    for word in cli._SUBCOMMANDS:
        check(f"bare '{word}' dispatches", [word][0] in cli._SUBCOMMANDS)
        sentence = f"{word} the results into a summary"
        check(f"'{word} ...' stays a task", sentence not in cli._SUBCOMMANDS)
        check(f"'{word} ...' parses as the task positional",
              parse([sentence]).task == sentence)


def test_serve_parser_owns_host_and_port():
    """--host/--port used to sit on the top-level parser, so `chad --port 9999 "fix it"`
    parsed fine and then ignored the port. They belong to the one command that reads
    them; the agent parser must now reject them outright."""
    serve_args = cli._serve_parser().parse_args(["--host", "0.0.0.0", "--port", "9999"])
    check("serve parses host", serve_args.host == "0.0.0.0", serve_args.host)
    check("serve parses port", serve_args.port == 9999, serve_args.port)
    with pytest.raises(SystemExit):
        cli._agent_parser().parse_args(["--port", "9999"])
    with pytest.raises(SystemExit):
        cli._agent_parser().parse_args(["--host", "0.0.0.0"])


def test_unattended_governor_knobs_are_env_only():
    """The governor cluster is set by unattended runners, never by hand, and every knob
    already had a CHAD_* twin — so the CLI spellings are gone rather than merely hidden.
    Rejecting them beats accepting-and-ignoring: a runner still passing `--turn-budget-s`
    must fail loudly, not silently run without a wall budget."""
    for flag, value in (("--turn-budget-s", "1770"), ("--turn-budget-tokens", "300000"),
                        ("--auto-continue", "3"), ("--think-ceiling", "6000")):
        with pytest.raises(SystemExit):
            cli._agent_parser().parse_args(["task", flag, value])
    with pytest.raises(SystemExit):
        cli._agent_parser().parse_args(["task", "--review-pass"])
    help_text = cli._agent_parser().format_help()
    for flag in ("--turn-budget-s", "--turn-budget-tokens", "--auto-continue",
                 "--review-pass", "--think-ceiling", "--levers"):
        check(f"{flag} absent from --help", flag not in help_text)
    # --think-budget is the one knob of this family kept on the CLI: unlike the rest it
    # is a capability/latency trade a person might reach for interactively.
    check("--think-budget stays visible", "--think-budget" in help_text)


def test_governor_env_knobs_are_read_by_the_cli():
    """cli.main does its own arithmetic against the wall budget (how much of the TASK
    deadline a relaunch inherits), so it must read CHAD_TURN_BUDGET_S itself rather than
    leaving it to Agent.__init__ — the knob being env-only is what makes that load-bearing."""
    import inspect
    src = inspect.getsource(cli._main)
    for var in ("CHAD_TURN_BUDGET_S", "CHAD_AUTO_CONTINUE", "CHAD_REVIEW_PASS"):
        check(f"{var} read in _main", var in src)


def test_backend_failure_reports_problem_cause_fix(capsys):
    """A dead remote backend used to exit through a raw BackendError traceback, which
    reads as a chad crash rather than 'the server you pointed me at is down'."""
    from chad.base_engine import BackendError
    err = BackendError("llama-server connection failed: URLError: Connection refused")
    with pytest.raises(SystemExit):
        cli._fail_backend(err, "http://10.0.0.5:8081")
    out = capsys.readouterr().err
    check("names the problem", "remote backend stopped answering" in out, out)
    check("names the cause", "cause:" in out and "connection failed" in out, out)
    check("names the unreachable url", "http://10.0.0.5:8081" in out, out)
    check("offers a fix", "fix:" in out, out)
    check("no traceback", "Traceback" not in out, out)
