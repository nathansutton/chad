"""Characterization tests for cli.py's model-resolution + env parsing — the bootstrap
path that currently has ZERO direct coverage. All model-free: we never call main()
(which loads MLX and reads argv); we drive the pure helpers `_env_int` and `_pick_model`,
monkeypatching `_detect_ram_gb` (it shells out to sysctl) and `os.path.isdir`.

A bug in `_pick_model` silently picks the wrong model size, or downloads a 12 GB repo
on a box that can't run it; a bug in `_env_int` mis-parses an advanced knob. These pin
the current contract so a refactor can't drift it.

Run: `uv run python tests/test_cli.py`
"""

import json
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
    # `--model auto` is the explicit spelling of "ignore the override, use the default".
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    model, why = cli._pick_model("auto")
    check("auto falls through to the shipped default", model == cli._HF_MODEL, model)


def test_pick_model_no_size_shorthands(monkeypatch):
    """2.0.0 retired the Ornith pair and with it `--model 35b` / `--model 9b`.

    The shorthands must not silently resolve to anything: they are now ordinary specs,
    passed through as literal HF repo ids. Pinned because the failure mode of a
    half-removed alias table is that `--model 9b` quietly loads the *default* model and
    the user never learns their flag stopped meaning anything.
    """
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    for spec in ("9b", "35b", "27b", "mlx-community/Whatever-4bit"):
        model, why = cli._pick_model(spec)
        check(f"--model {spec} passes through literally", model == spec, model)
        check(f"--model {spec} reason says requested", "requested" in why.lower(), why)


def test_pick_model_one_model_every_box(monkeypatch):
    """One model, whatever the RAM: there is no smaller tier to fall back to."""
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)   # no local build -> HF repo
    for ram in (16.0, 24.0, 64.0, None):
        monkeypatch.setattr(cli, "_detect_ram_gb", lambda ram=ram: ram)
        model, why = cli._pick_model()
        check(f"RAM {ram} -> the shipped repo", model == cli._HF_MODEL, model)
        check(f"RAM {ram} reason is a default", "default" in why, why)


def test_pick_model_small_box_warns(monkeypatch, capsys):
    """Below the 24 GB target chad warns and proceeds — it advises, it does not gate.

    Retiring the 9B removed the safe fallback, so this warning is the only thing
    standing between a 16 GB Mac and a silently unusable context window.
    """
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)

    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 16.0)
    model, _ = cli._pick_model()
    err = capsys.readouterr().err
    check("small box still served", model == cli._HF_MODEL, model)
    check("small box warns", "below the" in err, err)
    check("warning names the RAM read", "16 GB" in err, err)

    # RAM unreadable: same warning path, named as undetectable rather than a number.
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: None)
    cli._pick_model()
    check("unknown RAM warns 'undetectable'", "undetectable" in capsys.readouterr().err)

    # At/above the target: silent.
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 24.0)
    cli._pick_model()
    check("24 GB box does not warn", capsys.readouterr().err == "")


def test_model_download_gb():
    check("shipped repo -> ~13 GB", cli._model_download_gb(cli._HF_MODEL) == 13.2)


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
        cli._ensure_model(cli._HF_MODEL)
    check("preflight exits 1", e.value.code == 1, e.value.code)
    err = capsys.readouterr().err
    check("names the shortfall", "not enough free disk" in err, err)
    check("names required space", "~13 GB" in err, err)
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
        cli._ensure_model(cli._HF_MODEL)
    # Exit came from the user's "n" at the prompt, not the disk preflight.
    check("unreadable disk does not block", e.value.code == 1, e.value.code)


def test_cached_weights_complete(monkeypatch, tmp_path):
    """The guard must read WEIGHTS, not metadata. An interrupted first download leaves
    config.json + tokenizer in the snapshot and no tensors; treating that as a cache hit
    is what sent the load into mlx_lm's `No safetensors found` with no way back."""
    import huggingface_hub
    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    index = snap / "model.safetensors.index.json"

    def fake_cache(repo, filename, **kw):
        f = snap / filename
        return str(f) if f.exists() else None
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_cache)

    # metadata only — the exact state a ctrl-c'd first run leaves behind
    check("metadata alone is not complete",
          cli._cached_weights_complete("repo/x") is False)

    # sharded, index present but a shard still missing
    index.write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    (snap / "model-00001-of-00002.safetensors").write_text("x")
    check("missing shard is not complete",
          cli._cached_weights_complete("repo/x") is False)

    (snap / "model-00002-of-00002.safetensors").write_text("x")
    check("all shards present is complete",
          cli._cached_weights_complete("repo/x") is True)

    # a corrupt index must re-fetch rather than crash
    index.write_text("{not json")
    check("unreadable index is not complete",
          cli._cached_weights_complete("repo/x") is False)

    # single-file layout
    index.unlink()
    (snap / "model.safetensors").write_text("x")
    check("single-file layout is complete",
          cli._cached_weights_complete("repo/x") is True)


def test_ensure_model_resumes_partial_cache(monkeypatch, capsys):
    """A partial cache must NOT be reported as a fresh download: the message names the
    interrupted download, so a re-fetch on a machine that 'already has' the model does
    not read as a bug."""
    import huggingface_hub
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    # config.json cached, no weights anywhere — the interrupted-download state.
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, filename, **kw: "/c/config.json"
                        if filename == "config.json" else None)
    monkeypatch.setattr(cli, "_free_disk_gb", lambda path: 500.0)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    with pytest.raises(SystemExit):
        cli._ensure_model("repo/x")
    err = capsys.readouterr().err
    check("names the incomplete cache", "incomplete" in err, err)
    check("says it resumes", "Resuming" in err, err)


def test_pick_model_prefers_local_dir(monkeypatch):
    # A dev clone that already built the weights uses them instead of re-downloading.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    monkeypatch.setattr(os.path, "isdir", lambda p: p == cli._LOCAL_MODEL)
    model, _ = cli._pick_model()
    check("local build preferred over HF repo", model == cli._LOCAL_MODEL, model)
    # `--model auto` takes the same path (it means "the default", not "ignore local").
    model, _ = cli._pick_model("auto")
    check("auto also prefers the local build", model == cli._LOCAL_MODEL, model)


def test_pick_model_flag_auto_ignores_env(monkeypatch):
    # '--model auto' forces the default even when CHAD_MODEL is set: the env must NOT win.
    monkeypatch.setenv("CHAD_MODEL", "/env/repo")
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    model, why = cli._pick_model("auto")
    check("--model auto ignores env", model == cli._HF_MODEL, model)
    check("--model auto reason is a default", "default" in why, why)


def test_pick_model_flag_repo_passthrough(monkeypatch):
    # A spec is a literal repo id / local dir, passed through unchanged (the CLI twin of
    # CHAD_MODEL). RAM is irrelevant.
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
    model, why = cli._pick_model("/flag/repo")
    check("--model beats CHAD_MODEL", model == "/flag/repo", model)
    check("winner reason names --model", "--model" in why, why)


def test_pick_model_default_equals_auto(monkeypatch):
    # Regression lock for bench.py's no-arg callers: _pick_model() must behave exactly
    # like _pick_model("auto") for a fixed environment.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(cli, "_detect_ram_gb", lambda: 64.0)
    check("no-arg == auto", cli._pick_model() == cli._pick_model("auto"))


def test_ram_aware_ctx_limit():
    GB = 1e9
    # Measured on a 24 GB M4 Pro running the 27B (q3, 8-bit KV, memory clamp on, ONE
    # load — load/teardown cycling panics the GPU): 19.07 GB Metal working set, 12.329 GB
    # resident after load, 34,816 B/token KV, 262 k window. Peak memory over ctx:
    #   8k 14.59 GB | 16k 15.60 | 32k 16.82 | 49k 18.35 | 65k 18.91
    # The gap between peak and active — the prefill transient — climbs to 4.15 GB by 49k
    # and is FLAT from there (the adaptive chunker shrinks the chunk as the free band
    # closes), and past that point peak grows 33,936 B/token against a 34,816 B/token
    # cache. So the cost model is a fixed transient plus KV at its raw rate, which is
    # what the governor subtracts and divides by.
    BUDGET, ACTIVE, KV, TRANSIENT = 19.07 * GB, 12.329 * GB, 34_816, 4.3 * GB

    def peak_at(ctx):
        """The measured cost model, extrapolated to a candidate trigger."""
        return ACTIVE + TRANSIENT + KV * ctx

    n = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV)
    check("27B on 24GB: below the window", n < 262144 - 2048, n)
    # This is the assertion that keeps the defaults honest: whatever trigger the governor
    # picks, the measured peak at that trigger must stay inside the Metal budget. It is
    # the reason `safety` alone cannot be the whole story — the 4.3 GB transient and the
    # 12.3 GB of weights spend 87% of the budget before the first cached token.
    check("27B: measured peak at the trigger is under budget", peak_at(n) < BUDGET,
          peak_at(n) / GB)
    check("27B: and is not leaving the box idle", peak_at(n) > 0.95 * BUDGET,
          peak_at(n) / GB)

    # `safety` is the single headroom lever: tightening it strictly shrinks the window,
    # and it is the ONLY knob that does (the flat reserve_gb it replaced is gone).
    tighter = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV, safety=0.80)
    check("tighter safety shrinks window", tighter < n, (tighter, n))
    check("default safety holds back 2.5%",
          cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV, safety=0.975) == n)

    # The transient is a FIXED subtraction, not a per-token slope. A model that ignores
    # it (transient_bytes=0) over-picks by exactly its worth in tokens, which on this box
    # is the difference between 97% of budget and walking over the wall.
    blind = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV, transient_bytes=0)
    check("ignoring the transient over-picks", blind > n, (blind, n))
    check("...and that over-pick busts the budget", peak_at(blind) > BUDGET,
          peak_at(blind) / GB)
    check("transient costs exactly its worth in tokens",
          abs((blind - n) - TRANSIENT / KV) <= 1, (blind - n, TRANSIENT / KV))

    # Tight box (less working set) compacts sooner — strictly smaller window.
    tight = cli.ram_aware_ctx_limit(262144, 10.0 * GB, 8.0 * GB, KV)
    check("tight box compacts sooner", tight < n, (tight, n))

    # Huge box is capped at the model window minus the gen margin, never above it.
    huge = cli.ram_aware_ctx_limit(262144, 400.0 * GB, ACTIVE, KV)
    check("huge box capped at window-margin", huge == 262144 - 2048, huge)

    # Degenerate inputs -> None so the caller keeps the old fixed cap.
    check("no KV cost -> None", cli.ram_aware_ctx_limit(262144, 19 * GB, 12 * GB, 0) is None)
    check("no budget -> None", cli.ram_aware_ctx_limit(262144, 0, 12 * GB, KV) is None)

    # Over-subscribed (model already past the safe budget) -> floor, never negative.
    floored = cli.ram_aware_ctx_limit(262144, 14 * GB, 18 * GB, KV)
    check("over-subscribed -> floor", floored == 8192, floored)

    # Host physical pressure: when the host's reclaimable band is
    # tighter than the Metal band, IT binds — Docker/harbor pressure the Metal
    # budget cannot see must shrink the window.
    pressured = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV,
                                        host_avail_bytes=1.5 * GB)
    check("tight host band binds below the Metal band", pressured < n, (pressured, n))
    check("tight host band still floored, never negative",
          cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV,
                                  host_avail_bytes=0.02 * GB) == 8192)
    # A roomy host band changes nothing — the Metal band stays the binding one.
    roomy = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV, host_avail_bytes=200 * GB)
    check("roomy host band leaves the Metal result", roomy == n, (roomy, n))


def test_host_band_is_a_guard_not_the_primary_constraint():
    GB = 1e9
    BUDGET, ACTIVE, KV = 19.07 * GB, 12.329 * GB, 34_816
    metal = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV)
    # Measured right after a 12.3 GB load the reclaimable band reads 3.9 GB — the weights
    # took it and nothing has been reclaimed yet. The host branch must NOT bind there:
    # it is a soft pressure signal the OS compresses around, so it sizes the resident KV
    # cache only and never charges the short-lived prefill transient against it. Charging
    # it there put a box with room to spare on the 8192 floor.
    just_loaded = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV,
                                          host_avail_bytes=3.936 * GB)
    check("band right after load does not bind", just_loaded == metal,
          (just_loaded, metal))
    # It still bites when the box is genuinely oversubscribed by another process.
    squeezed = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV,
                                       host_avail_bytes=1.2 * GB)
    check("real pressure still binds", squeezed < metal, (squeezed, metal))


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
    test_host_band_is_a_guard_not_the_primary_constraint()
    with pytest.MonkeyPatch.context() as mp:
        test_env_float(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_env_int(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_override(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_one_model_every_box(mp)
    with pytest.MonkeyPatch.context() as mp:
        test_pick_model_no_size_shorthands(mp)
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


# --- DFlash2 drafter consent/download (rides _ensure_model) ---------------------

def test_drafter_bundle_resolution(monkeypatch, tmp_path):
    """The drafter ships INSIDE the weights dir: found when `dflash/` is there,
    None when it isn't (decode runs serial, no download), and CHAD_DFLASH_PATH
    overrides both."""
    from chad import mlx_dflash
    monkeypatch.delenv("CHAD_DFLASH_PATH", raising=False)
    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    check("no bundle -> None", mlx_dflash.bundle_dir(str(mdir)) is None)
    bundle = mdir / "dflash"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    check("bundle found", mlx_dflash.bundle_dir(str(mdir)) == str(bundle))
    monkeypatch.setenv("CHAD_DFLASH_PATH", str(tmp_path / "elsewhere"))
    check("env overrides the bundle",
          mlx_dflash.bundle_dir(str(mdir)) == str(tmp_path / "elsewhere"))
