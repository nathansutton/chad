"""Characterization tests for cli.py's model-resolution + env parsing — the bootstrap
path that currently has ZERO direct coverage. All model-free: we never load a model; we
drive the pure helpers `_env_int` and `_pick_model` directly, handing them a `cli.Host`
whose RAM probe is a stub (the real one shells out to sysctl) and a local-build path
under tmp_path that does or does not exist.

A bug in `_pick_model` silently picks the wrong model size, or downloads a 12 GB repo
on a box that can't run it; a bug in `_env_int` mis-parses an advanced knob. These pin
the current contract so a refactor can't drift it.

Run: `uv run python tests/test_cli.py`
"""

import inspect
import json
import os
import subprocess
import sys

import pytest

from chad import cli

PASS = 0
FAIL = 0

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _env_with_src():
    """This process's environment with the checkout's src/ first on PYTHONPATH, for a
    fresh interpreter that must import this tree's `chad`."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(
        p for p in (_SRC, os.environ.get("PYTHONPATH")) if p)}


def _host(ram):
    """A Host whose physical-RAM probe reads `ram` GiB (None = unreadable)."""
    return cli.Host(ram_gb=lambda: ram)


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


def test_pick_model_override(monkeypatch, tmp_path):
    # An explicit CHAD_MODEL wins outright, regardless of RAM or local dirs, and the
    # reason says the choice was requested rather than defaulted.
    monkeypatch.setenv("CHAD_MODEL", "/some/local/model")
    # even with a surprising RAM reading and a local build present, the override must
    # short-circuit first
    local = tmp_path / "local-build"
    local.mkdir()
    model, why = cli._pick_model(host=_host(8.0), local_model=str(local))
    check("override returns CHAD_MODEL value", model == "/some/local/model", model)
    check("override reason says requested", "requested" in why.lower(), why)
    # `--model` (the `spec` argument) outranks CHAD_MODEL: the flag is the more specific
    # signal, and a shell that exports CHAD_MODEL globally must not pin every run.
    model, _ = cli._pick_model("/flag/model", host=_host(8.0), local_model=str(local))
    check("--model beats CHAD_MODEL", model == "/flag/model", model)
    # `--model auto` is the explicit spelling of "ignore the override, use the default".
    model, why = cli._pick_model("auto", host=_host(64.0),
                                 local_model=str(tmp_path / "no-local-build"))
    check("auto falls through to the shipped default", model == cli._HF_MODEL, model)


def test_pick_model_no_size_shorthands(monkeypatch, tmp_path):
    """2.0.0 retired the Ornith pair and with it `--model 35b` / `--model 9b`.

    The shorthands must not silently resolve to anything: they are now ordinary specs,
    passed through as literal HF repo ids. Pinned because the failure mode of a
    half-removed alias table is that `--model 9b` quietly loads the *default* model and
    the user never learns their flag stopped meaning anything.
    """
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    no_local = str(tmp_path / "no-local-build")
    for spec in ("9b", "35b", "27b", "mlx-community/Whatever-4bit"):
        model, why = cli._pick_model(spec, host=_host(64.0), local_model=no_local)
        check(f"--model {spec} passes through literally", model == spec, model)
        check(f"--model {spec} reason says requested", "requested" in why.lower(), why)


def test_pick_model_one_model_every_box(monkeypatch, tmp_path):
    """One model, whatever the RAM: there is no smaller tier to fall back to."""
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    no_local = str(tmp_path / "no-local-build")   # no local build -> HF repo
    for ram in (16.0, 24.0, 64.0, None):
        model, why = cli._pick_model(host=_host(ram), local_model=no_local)
        check(f"RAM {ram} -> the shipped repo", model == cli._HF_MODEL, model)
        check(f"RAM {ram} reason is a default", "default" in why, why)


def test_pick_model_small_box_warns(monkeypatch, capsys, tmp_path):
    """Below the 24 GB target chad warns and proceeds — it advises, it does not gate.

    Retiring the 9B removed the safe fallback, so this warning is the only thing
    standing between a 16 GB Mac and a silently unusable context window.
    """
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    no_local = str(tmp_path / "no-local-build")

    model, _ = cli._pick_model(host=_host(16.0), local_model=no_local)
    err = capsys.readouterr().err
    check("small box still served", model == cli._HF_MODEL, model)
    check("small box warns", "below the" in err, err)
    check("warning names the RAM read", "16 GB" in err, err)

    # RAM unreadable: same warning path, named as undetectable rather than a number.
    cli._pick_model(host=_host(None), local_model=no_local)
    check("unknown RAM warns 'undetectable'", "undetectable" in capsys.readouterr().err)

    # At/above the target: silent.
    cli._pick_model(host=_host(24.0), local_model=no_local)
    check("24 GB box does not warn", capsys.readouterr().err == "")


def test_model_download_gb():
    from chad import gguf_pack
    shipped = cli._model_download_gb(cli._HF_MODEL)
    check("shipped file -> its size plus the sidecar", shipped == 13.2 + gguf_pack.SIDECAR_GB,
          shipped)
    check("an unknown hub file offline falls back to the heaviest known",
          cli._model_download_gb("unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_0.gguf")
          == 14.3 + gguf_pack.SIDECAR_GB)
    check("a packed repo keeps the packed figure", cli._model_download_gb("some/repo") == 8.3)


def test_default_model_is_a_hub_gguf():
    """The shipped model is one of Unsloth's files, named inside their repo, and the
    dev clone's local shortcut is the same file under models/."""
    from chad import gguf_pack
    check("default is owner/repo/file.gguf",
          gguf_pack.hub_spec(cli._HF_MODEL) == (gguf_pack.HUB_REPO, "Qwen3.8-27B-UD-Q3_K_XL.gguf"),
          cli._HF_MODEL)
    check("local shortcut is the same file", cli._LOCAL_MODEL.endswith("Qwen3.8-27B-UD-Q3_K_XL.gguf"))


def test_cached_weights_complete_for_a_hub_gguf():
    """A GGUF session is downloaded when the file AND its drafter/tokenizer sidecar are
    cached; the file alone would load, then fetch 1.2 GB inside the engine."""
    from chad import gguf_pack
    have = set()

    def fake_cache(repo, filename):
        return f"/c/{filename}" if (repo, filename) in have else None

    def complete():
        return cli._cached_weights_complete(cli._HF_MODEL, cached_file=fake_cache)

    check("nothing cached", complete() is False)
    have.add((gguf_pack.HUB_REPO, "Qwen3.8-27B-UD-Q3_K_XL.gguf"))
    check("file without sidecar is not complete", complete() is False)
    have.add((gguf_pack.TOKENIZER_DONOR, "dflash/model.safetensors"))
    have.add((gguf_pack.TOKENIZER_DONOR, "tokenizer.json"))
    check("drafter + tokenizer without the template is not complete", complete() is False)
    have.update((gguf_pack.TOKENIZER_DONOR, f) for f in gguf_pack._SIDECAR_FILES)
    check("file + whole sidecar is complete", complete() is True)


def test_ensure_model_hub_gguf(monkeypatch, capsys, tmp_path):
    """Cached: returns without touching the network. Uncached at a TTY: names the
    download and its size, and a `n` aborts before any fetch."""
    monkeypatch.chdir(tmp_path)
    cached = cli.Host(cached_file=lambda repo, filename: f"/c/{filename}",
                      free_disk_gb=lambda path: 500.0,
                      stdin_isatty=lambda: True, ask=_Terminal("n").ask)
    check("cached hub file returns", cli._ensure_model(cli._HF_MODEL, host=cached) is None)
    fresh = cli.Host(cached_file=lambda repo, filename: None,
                     free_disk_gb=lambda path: 500.0,
                     stdin_isatty=lambda: True, ask=_Terminal("n").ask)
    with pytest.raises(SystemExit):
        cli._ensure_model(cli._HF_MODEL, host=fresh)
    err = capsys.readouterr().err
    check("names the file", "Qwen3.8-27B-UD-Q3_K_XL.gguf" in err, err)
    check("states the size with the sidecar", "~14 GB" in err, err)
    check("a fresh fetch is not called a resume", "Resuming" not in err, err)
    check("names the converted copy", "converted copy (~13 GB)" in err, err)


def test_hub_gguf_preflight_counts_the_converted_copy(monkeypatch, capsys, tmp_path):
    """The first start saves the converted model beside the pack, the file's size
    again: room for the download alone is not room for the model."""
    monkeypatch.chdir(tmp_path)
    tight = cli.Host(cached_file=lambda repo, filename: None,
                     free_disk_gb=lambda path: 20.0,          # > 14.4 + 2, < 14.4 + 13.2 + 2
                     stdin_isatty=lambda: True, ask=_Terminal("n").ask)
    with pytest.raises(SystemExit):
        cli._ensure_model(cli._HF_MODEL, host=tight)
    err = capsys.readouterr().err
    check("refused before asking", "not enough free disk" in err, err)
    check("names both parts", "converted copy in ~/.cache/chad" in err, err)


def test_free_disk_gb():
    # A real, existing path reports a plausible number.
    free = cli._free_disk_gb("~")
    check("home free-disk readable", free is not None and free > 0, free)
    # A nonexistent deep path climbs to an existing parent instead of crashing.
    free = cli._free_disk_gb("~/definitely/not/a/real/dir/blobs")
    check("missing path climbs to parent", free is not None and free > 0, free)


class _Terminal:
    """A person at a TTY who answers every prompt with `answer`, and what they were asked."""

    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def ask(self, prompt):
        self.asked.append(prompt)
        return self.answer


def test_ensure_model_disk_preflight(monkeypatch, capsys, tmp_path):
    """Devex review T2: a machine without room for the download must be refused
    BEFORE the download starts, with the shortfall and the cache-GC command named."""
    monkeypatch.chdir(tmp_path)  # the repo id is not a local directory from here
    terminal = _Terminal("n")
    host = cli.Host(cached_file=lambda repo, filename: None,  # nothing cached
                    free_disk_gb=lambda path: 1.0,
                    stdin_isatty=lambda: True, ask=terminal.ask)
    with pytest.raises(SystemExit) as e:
        cli._ensure_model(cli._HF_MODEL, host=host)
    check("preflight exits 1", e.value.code == 1, e.value.code)
    check("refused before the consent prompt", terminal.asked == [], terminal.asked)
    err = capsys.readouterr().err
    check("names the shortfall", "not enough free disk" in err, err)
    check("names required space", "~14 GB" in err, err)
    check("points at cache GC", "hf cache" in err, err)


def test_ensure_model_disk_preflight_unreadable(monkeypatch, tmp_path):
    """If free disk can't be read the preflight must NOT block (it guards, never
    gates): the flow proceeds to the confirm prompt / download attempt."""
    monkeypatch.chdir(tmp_path)  # the repo id is not a local directory from here
    terminal = _Terminal("n")
    host = cli.Host(cached_file=lambda repo, filename: None,  # nothing cached
                    free_disk_gb=lambda path: None,           # free disk unreadable
                    stdin_isatty=lambda: True, ask=terminal.ask)
    with pytest.raises(SystemExit) as e:
        cli._ensure_model(cli._HF_MODEL, host=host)
    # Exit came from the user's "n" at the prompt, not the disk preflight.
    check("unreadable disk does not block", e.value.code == 1, e.value.code)
    check("reached the consent prompt", terminal.asked == ["Download now? [Y/n] "],
          terminal.asked)


def test_cached_weights_complete(tmp_path):
    """The guard must read WEIGHTS, not metadata. An interrupted first download leaves
    config.json + tokenizer in the snapshot and no tensors; treating that as a cache hit
    is what sent the load into mlx_lm's `No safetensors found` with no way back."""
    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    index = snap / "model.safetensors.index.json"

    def fake_cache(repo, filename):
        f = snap / filename
        return str(f) if f.exists() else None

    def complete():
        return cli._cached_weights_complete("repo/x", cached_file=fake_cache)

    # metadata only — the exact state a ctrl-c'd first run leaves behind
    check("metadata alone is not complete", complete() is False)

    # sharded, index present but a shard still missing
    index.write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    (snap / "model-00001-of-00002.safetensors").write_text("x")
    check("missing shard is not complete", complete() is False)

    (snap / "model-00002-of-00002.safetensors").write_text("x")
    check("all shards present is complete", complete() is True)

    # a corrupt index must re-fetch rather than crash
    index.write_text("{not json")
    check("unreadable index is not complete", complete() is False)

    # single-file layout
    index.unlink()
    (snap / "model.safetensors").write_text("x")
    check("single-file layout is complete", complete() is True)


def test_ensure_model_resumes_partial_cache(monkeypatch, capsys, tmp_path):
    """A partial cache must NOT be reported as a fresh download: the message names the
    interrupted download, so a re-fetch on a machine that 'already has' the model does
    not read as a bug."""
    monkeypatch.chdir(tmp_path)  # the repo id is not a local directory from here
    # config.json cached, no weights anywhere — the interrupted-download state.
    host = cli.Host(cached_file=lambda repo, filename: "/c/config.json"
                    if filename == "config.json" else None,
                    free_disk_gb=lambda path: 500.0,
                    stdin_isatty=lambda: True, ask=_Terminal("n").ask)
    with pytest.raises(SystemExit):
        cli._ensure_model("repo/x", host=host)
    err = capsys.readouterr().err
    check("names the incomplete cache", "incomplete" in err, err)
    check("says it resumes", "Resuming" in err, err)


def test_pick_model_prefers_local_dir(monkeypatch, tmp_path):
    # A dev clone that already built the weights uses them instead of re-downloading.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    local = tmp_path / "models" / "Qwen3.8-27B-q3_e3h5"
    local.mkdir(parents=True)
    model, _ = cli._pick_model(host=_host(64.0), local_model=str(local))
    check("local build preferred over HF repo", model == str(local), model)
    # `--model auto` takes the same path (it means "the default", not "ignore local").
    model, _ = cli._pick_model("auto", host=_host(64.0), local_model=str(local))
    check("auto also prefers the local build", model == str(local), model)
    # ...and a local copy of the GGUF file itself counts the same way.
    gguf = tmp_path / "models" / "Qwen3.8-27B-UD-Q3_K_XL.gguf"
    gguf.write_bytes(b"GGUF")
    model, _ = cli._pick_model(host=_host(64.0), local_model=str(gguf))
    check("local GGUF file preferred over the hub", model == str(gguf), model)
    # ...and the build looked for by default is the dev clone's own models/ dir.
    default = inspect.signature(cli._pick_model).parameters["local_model"].default
    check("default local build is the dev clone's", default == cli._LOCAL_MODEL, default)


def test_pick_model_flag_auto_ignores_env(monkeypatch, tmp_path):
    # '--model auto' forces the default even when CHAD_MODEL is set: the env must NOT win.
    monkeypatch.setenv("CHAD_MODEL", "/env/repo")
    model, why = cli._pick_model("auto", host=_host(64.0),
                                 local_model=str(tmp_path / "no-local-build"))
    check("--model auto ignores env", model == cli._HF_MODEL, model)
    check("--model auto reason is a default", "default" in why, why)


def test_pick_model_flag_repo_passthrough(monkeypatch):
    # A spec is a literal repo id / local dir, passed through unchanged (the CLI twin of
    # CHAD_MODEL). RAM is irrelevant.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    model, why = cli._pick_model("/some/local/model", host=_host(8.0))
    check("--model repo passthrough", model == "/some/local/model", model)
    check("passthrough reason names source + override",
          "--model" in why and "override" in why.lower(), why)


def test_pick_model_flag_beats_env(monkeypatch, tmp_path):
    # Both set -> the CLI flag wins over CHAD_MODEL.
    monkeypatch.setenv("CHAD_MODEL", "/env/repo")
    model, why = cli._pick_model("/flag/repo", host=_host(64.0),
                                 local_model=str(tmp_path / "no-local-build"))
    check("--model beats CHAD_MODEL", model == "/flag/repo", model)
    check("winner reason names --model", "--model" in why, why)


def test_pick_model_default_equals_auto(monkeypatch, tmp_path):
    # Regression lock for bench.py's no-arg callers: _pick_model() must behave exactly
    # like _pick_model("auto") for a fixed environment.
    monkeypatch.delenv("CHAD_MODEL", raising=False)
    fixed = {"host": _host(64.0), "local_model": str(tmp_path / "no-local-build")}
    check("no-arg == auto", cli._pick_model(**fixed) == cli._pick_model("auto", **fixed))


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
    BUDGET, ACTIVE, KV, TRANSIENT = 19.07 * GB, 12.329 * GB, 34_816, cli.PREFILL_TRANSIENT_BYTES

    def peak_at(ctx):
        """The measured cost model, extrapolated to a candidate trigger."""
        return ACTIVE + TRANSIENT + KV * ctx

    n = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV)
    check("27B on 24GB: below the window", n < 262144 - 2048, n)
    # This is the assertion that keeps the defaults honest: whatever trigger the governor
    # picks, the measured peak at that trigger must stay inside the Metal budget. It is
    # the reason `safety` alone cannot be the whole story — the prefill transient and the
    # 12.3 GB of weights spend most of the budget before the first cached token.
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
    # took it and nothing has been reclaimed yet. The band is a soft pressure signal the
    # OS compresses around, so it sizes the resident KV cache only and never charges the
    # short-lived prefill transient against it: charging it there put a box with room to
    # spare on the 8192 floor. The invariant is therefore "exactly the uncharged band, or
    # the Metal window if that is smaller" — with the flat 2 GB transient the Metal
    # window is the wider of the two here (122k against 110k), so the stale band trims
    # it by about a tenth until the next live recheck. It binds, but lightly.
    just_loaded = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV,
                                          host_avail_bytes=3.936 * GB)
    uncharged = int(3.936 * GB * 0.975 / KV)
    check("band right after load binds uncharged, if at all",
          just_loaded == min(metal, uncharged), (just_loaded, metal, uncharged))
    check("and binds lightly", 0.85 * metal <= just_loaded < metal, (just_loaded, metal))
    # Charged, the same reading would halve it.
    charged = int((3.936 * GB * 0.975 - cli.PREFILL_TRANSIENT_BYTES) / KV)
    check("charging the transient to the band would halve the window",
          charged < 0.5 * metal, (charged, metal))
    # It still bites when the box is genuinely oversubscribed by another process.
    squeezed = cli.ram_aware_ctx_limit(262144, BUDGET, ACTIVE, KV,
                                       host_avail_bytes=1.2 * GB)
    check("real pressure still binds", squeezed < metal, (squeezed, metal))


def test_prefill_transient_floor_tracks_the_cache_mode():
    """The flat floor was measured with BOTH mechanisms in — a quantized cache (no
    per-chunk snapshot pinning the buffers) and sliced prefill attention. Each alone
    leaves one growing term, so the governor must charge the tall floor for an fp16
    cache (CHAD_KV_BITS=0, an uncovered shape) or unsliced attention."""
    flat, tall = cli.PREFILL_TRANSIENT_BYTES, cli.PREFILL_TRANSIENT_BYTES_UNSLICED
    check("floors are ordered", flat < tall, (flat, tall))
    check("quantized + sliced -> flat", cli.prefill_transient_bytes(8, True) == flat)
    check("fp16 cache -> tall", cli.prefill_transient_bytes(None, True) == tall)
    check("explicit 0 -> tall", cli.prefill_transient_bytes(0, True) == tall)
    check("unsliced attention -> tall", cli.prefill_transient_bytes(8, False) == tall)


def test_pick_model_routes_a_gguf_file(monkeypatch, tmp_path):
    """A `.gguf` path goes through gguf_pack.materialize, however it is spelled: with
    a `~` and an upper-case suffix included. A path that is not a file falls through
    as an ordinary explicit request rather than being materialized."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "m.GGUF").write_bytes(b"GGUF")
    local = tmp_path / "local"
    local.mkdir()
    built = lambda path: "built:" + path  # noqa: E731 — the seam under test
    model, why = cli._pick_model("~/m.GGUF", host=_host(8.0), local_model=str(local),
                                 materialize=built)
    check("tilde + upper-case suffix route to materialize",
          model == "built:" + str(tmp_path / "m.GGUF"), model)
    check("reason names the source", why == "GGUF file (--model override)", why)
    missing = str(tmp_path / "missing.gguf")
    model, why = cli._pick_model(missing, host=_host(8.0), local_model=str(local),
                                 materialize=built)
    check("a missing .gguf is an ordinary explicit request", model == missing, model)


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


def test_version_flag():
    # argparse's `version` action prints to stdout and exits 0 during parse_args(),
    # BEFORE _preflight() ever runs — so --version works even off Apple Silicon. A real
    # process, so the flag arrives the way the console script receives it: on argv.
    from chad import __version__
    out = subprocess.run([sys.executable, "-m", "chad.cli", "--version"],
                         capture_output=True, text=True, env=_env_with_src(), timeout=120)
    check("--version exits 0", out.returncode == 0, repr(out.returncode) + out.stderr)
    check(f"--version prints chad {__version__}",
          out.stdout.startswith(f"chad {__version__}"), out.stdout)


def test_import_does_not_load_the_engine():
    # `chad --help`, `--version` and `chad levers` run with only this module imported, so
    # the ~0.75 s of mlx_lm + transformers must wait for a real run. A fresh interpreter,
    # because this one has already imported the engine for other tests.
    code = ("import sys, chad.cli; print([m for m in ('chad.engine', 'mlx_lm', "
            "'transformers') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=_env_with_src(), timeout=120, check=True)
    check("import chad.cli pulls in no engine", out.stdout.strip() == "[]",
          out.stdout + out.stderr)


def test_default_backend_is_the_real_engine_agent_and_repl():
    # `_main` builds its engine, agent loop and REPL from `_load_backend()`, which it
    # calls only once argparse and the subcommands are through (so the import cost above
    # is paid by real runs alone). What comes back must be the real classes.
    from chad.agent import Agent, repl
    from chad.engine import Engine
    backend = cli._load_backend()
    check("real engine class", backend.engine is Engine, backend.engine)
    check("real agent class", backend.agent is Agent, backend.agent)
    check("real repl", backend.repl is repl, backend.repl)


def test_preflight_skips_apple_gate_for_remote_backend(capsys):
    # The remote backend loads no MLX, so _preflight must NOT hard-stop on a non-Apple
    # host — that's what lets chad run inside a Linux benchmark container against a remote
    # server. Simulate a Linux/x86 box and assert llama passes while mlx would exit.
    linux = cli.Host(platform_id=lambda: ("Linux", "x86_64"))
    cli._preflight("llama", host=linux)   # must return, not exit
    with pytest.raises(SystemExit) as exc:
        cli._preflight("mlx", host=linux)
    check("mlx backend still gated off Apple Silicon", exc.value.code == 1)
    err = capsys.readouterr().err
    check("names the detected platform", "detected: Linux x86_64" in err, err)
    # An Apple Silicon Mac passes the gate.
    cli._preflight("mlx", host=cli.Host(platform_id=lambda: ("Darwin", "arm64")))


def test_version_string_never_raises():
    # The commit detail is best-effort: if distribution metadata is unreadable the
    # helper must still return a plain "chad <version>" string, never propagate.
    def unreadable_metadata():
        raise RuntimeError("no metadata")
    s = cli._version_string(direct_url_json=unreadable_metadata)
    check("still a string", isinstance(s, str), repr(s))
    check("starts with chad ", s.startswith("chad "), s)


def test_home_dir_note_written_in_home(monkeypatch, capsys, tmp_path):
    # Launching in ~ prints a one-line nudge to cd into a project — no exit,
    # no behavior change. chad snapshots the cwd, so home is rarely the intended dir.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", os.getcwd())  # launched from `~` itself
    cli._maybe_home_dir_note()
    err = capsys.readouterr().err
    check("home-dir note written", "home directory" in err, err)


def test_home_dir_note_absent_in_project(monkeypatch, capsys, tmp_path):
    # A real project dir (not ~) gets no note — home-dir only, no marker-file guessing.
    project = tmp_path / "some" / "project"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(project)
    cli._maybe_home_dir_note()
    err = capsys.readouterr().err
    check("no note outside home", err == "", repr(err))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --- sampler env: one call, so the knobs cannot drift apart ---------------

def test_apply_sampler_env_sets_all_three(monkeypatch):
    """They used to be three inlined sibling blocks in main(), so a second path that built
    its own engine silently ran without ANY of them. One function now, so a caller cannot
    honor temp and forget min_p."""
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
