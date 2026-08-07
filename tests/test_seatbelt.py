"""Seatbelt confinement for yolo-mode bash (seatbelt.py + the tool_bash seam).

The contract under test: wrapping happens ONLY when all four gates agree (lever on,
executing agent in yolo mode, platform capable, probe green); when any gate says no,
tool_bash's spawn is byte-identical to the pre-seatbelt behavior. The profile is a
deny-writes-outside-allowlist; the note appended on a detected denial is what keeps
the model from retrying into the wall. Real sandbox application is exercised in the
darwin-gated e2e at the bottom; everything else forces the gates so the suite passes
on any platform (and inside CI/test sandboxes, where Seatbelt cannot nest).
"""
import os
import subprocess

import pytest

from chad import levers, seatbelt, tools


@pytest.fixture(autouse=True)
def _reset_seatbelt(monkeypatch):
    monkeypatch.setattr(seatbelt, "_ctx", {"active": False, "workspace": None})
    monkeypatch.setattr(seatbelt, "_profiles", {})
    monkeypatch.setattr(seatbelt, "_probe_result", None)


def _force_capable(monkeypatch, ok=True):
    monkeypatch.setattr(seatbelt, "probe", lambda: ok)


# -- wrap gating --------------------------------------------------------------

def test_no_wrap_when_lever_off(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "yolo_seatbelt")
    _force_capable(monkeypatch)
    seatbelt.set_context(True, os.getcwd())
    assert seatbelt.wrap_argv("echo hi") is None


def test_no_wrap_outside_yolo_context(monkeypatch):
    _force_capable(monkeypatch)
    seatbelt.set_context(False, None)
    assert seatbelt.wrap_argv("echo hi") is None


def test_no_wrap_when_probe_fails(monkeypatch):
    _force_capable(monkeypatch, ok=False)
    seatbelt.set_context(True, os.getcwd())
    assert seatbelt.wrap_argv("echo hi") is None


def test_wrap_argv_shape(monkeypatch, tmp_path):
    _force_capable(monkeypatch)
    seatbelt.set_context(True, str(tmp_path))
    argv = seatbelt.wrap_argv("echo hi > f.txt")
    assert argv is not None
    assert argv[0] == seatbelt.SANDBOX_EXEC and argv[1] == "-f"
    assert argv[3:] == ["/bin/sh", "-c", "echo hi > f.txt"]
    with open(argv[2], encoding="utf-8") as fh:
        assert str(tmp_path.resolve()) in fh.read()


def test_profile_cached_per_workspace(monkeypatch, tmp_path):
    _force_capable(monkeypatch)
    seatbelt.set_context(True, str(tmp_path))
    a = seatbelt.wrap_argv("true")[2]
    b = seatbelt.wrap_argv("false")[2]
    assert a == b


# -- profile content ----------------------------------------------------------

def test_profile_denies_by_default_and_allows_workspace(tmp_path):
    text = seatbelt.profile_text(str(tmp_path))
    assert "(deny file-write*)" in text
    assert f'(subpath "{tmp_path.resolve()}")' in text
    assert "(allow default)" in text
    # temp + cache roots a real command stream needs (uv/pip cache on macOS
    # lives under ~/Library/Caches, not ~/.cache)
    for needle in ("/private/var/folders", "Library/Caches", ".chad"):
        assert needle in text


def test_profile_escapes_scheme_metachars(tmp_path):
    evil = tmp_path / 'we"ird'
    evil.mkdir()
    text = seatbelt.profile_text(str(evil))
    assert 'we\\"ird' in text
    assert 'we"ird")' not in text


def _split_at_deny_tail(text: str):
    """(allow-and-before, trailing deny block) — the deny block that carves paths
    back OUT of the writable allowlist sits last, because the last matching
    Seatbelt rule wins."""
    i = text.rindex("(deny file-write*")
    return text[:i], text[i:]


def _worktree_fixture(tmp_path):
    ws = tmp_path / "wt"
    ws.mkdir()
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n")
    (ws / ".git").write_text(f"gitdir: {gitdir}\n")
    return ws, gitdir.resolve(), (gitdir / "../..").resolve()


def test_profile_worktree_gitdir_carveout(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "seatbelt_protect_git")
    ws, gitdir, common = _worktree_fixture(tmp_path)
    head, tail = _split_at_deny_tail(seatbelt.profile_text(str(ws)))
    assert str(gitdir) in head and str(common) in head
    assert str(gitdir) not in tail and str(common) not in tail


def test_profile_protect_git_flips_worktree_gitdirs_to_deny(tmp_path):
    # the suite runs CHAD_ENABLE=all, so the protection tier is on here
    ws, gitdir, common = _worktree_fixture(tmp_path)
    head, tail = _split_at_deny_tail(seatbelt.profile_text(str(ws)))
    assert str(gitdir) in tail and str(common) in tail
    assert str(gitdir) not in head and str(common) not in head


def test_profile_protect_git_denies_workspace_dotgit(tmp_path):
    _head, tail = _split_at_deny_tail(seatbelt.profile_text(str(tmp_path)))
    ws = tmp_path.resolve()
    assert f'(subpath "{ws / ".git"}")' in tail


def test_profile_checkpoints_denied_even_without_git_tier(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "seatbelt_protect_git")
    text = seatbelt.profile_text(str(tmp_path))
    head, tail = _split_at_deny_tail(text)
    ckpt = os.path.join(os.path.expanduser("~"), ".chad", "checkpoints")
    assert f'(subpath "{ckpt}")' in tail
    assert "/.git" not in tail
    # ~/.chad itself stays on the allowlist; only the undo history is carved out
    assert os.path.join(os.path.expanduser("~"), ".chad") in head


def test_profile_cache_distinguishes_git_tier(monkeypatch, tmp_path):
    _force_capable(monkeypatch)
    seatbelt.set_context(True, str(tmp_path))
    with_tier = seatbelt.wrap_argv("true")[2]
    monkeypatch.setenv("CHAD_DISABLE", "seatbelt_protect_git")
    without_tier = seatbelt.wrap_argv("true")[2]
    assert with_tier != without_tier
    with open(with_tier, encoding="utf-8") as fh:
        assert str(tmp_path.resolve() / ".git") in fh.read()
    with open(without_tier, encoding="utf-8") as fh:
        assert str(tmp_path.resolve() / ".git") not in fh.read()


def test_profile_plain_repo_no_carveout(tmp_path):
    (tmp_path / ".git").mkdir()  # normal repo: .git is a dir, no pointer to chase
    assert seatbelt._worktree_gitdirs(str(tmp_path)) == []


# -- enforcement probe --------------------------------------------------------
# probe() must prove the profile DENIES, not merely that sandbox-exec runs: a
# profile that fails open would otherwise report confinement it does not have.

_real_run = subprocess.run


def _probe_with(monkeypatch, fake_run):
    monkeypatch.setattr(seatbelt, "available", lambda: True)
    monkeypatch.setattr(seatbelt.subprocess, "run", fake_run)
    return seatbelt.probe()


def test_probe_rejects_non_enforcing_sandbox(monkeypatch, caplog):
    """sandbox-exec runs the command fine but enforces nothing (both writes land):
    the probe must come back False, loudly — this is the fail-open case."""
    def fake(argv, **kw):
        return _real_run(argv[-3:], capture_output=True, check=False)
    with caplog.at_level("ERROR", logger="chad"):
        assert _probe_with(monkeypatch, fake) is False
    assert any("FAILED to enforce" in r.getMessage() for r in caplog.records)


def test_probe_rejects_sandbox_that_cannot_run(monkeypatch):
    """Nothing executes at all (nested sandbox): neither write lands -> False."""
    def fake(argv, **kw):
        return subprocess.CompletedProcess(argv, 1)
    assert _probe_with(monkeypatch, fake) is False


def test_probe_accepts_enforcing_sandbox(monkeypatch):
    """The allowed write lands and the denied one does not -> True. The fake
    simulates enforcement by executing only the allowed half of the command."""
    def fake(argv, **kw):
        return _real_run(["/bin/sh", "-c", argv[-1].split(";")[0]],
                         capture_output=True, check=False)
    assert _probe_with(monkeypatch, fake) is True


def test_probe_result_is_cached(monkeypatch):
    calls = []
    def fake(argv, **kw):
        calls.append(argv)
        return _real_run(["/bin/sh", "-c", argv[-1].split(";")[0]],
                         capture_output=True, check=False)
    assert _probe_with(monkeypatch, fake) is True
    assert seatbelt.probe() is True
    assert len(calls) == 1


# -- the spawned shell's environment (bash_env_guard) -------------------------

def test_bash_env_strips_credential_shaped_names(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "k")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "k")
    monkeypatch.setenv("MY_DB_PASSWORD", "k")
    monkeypatch.setenv("SOME_CLIENT_SECRET", "k")
    monkeypatch.setenv("TOKEN_COUNT", "5")            # TOKEN not at the end: keep
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "1")  # likewise
    env = tools._bash_env()
    assert env is not None
    for gone in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "GITHUB_TOKEN",
                 "MY_DB_PASSWORD", "SOME_CLIENT_SECRET"):
        assert gone not in env
    for kept in ("PATH", "TOKEN_COUNT", "TOKENIZERS_PARALLELISM"):
        assert kept in env


def test_bash_env_off_means_inherit(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "bash_env_guard")
    assert tools._bash_env() is None


def test_bash_env_guard_fires_only_when_something_dropped(monkeypatch):
    for k in list(os.environ):
        if tools._ENV_SECRET_RE.search(k):
            monkeypatch.delenv(k)
    before = levers.fire_counts().get("bash_env_guard", 0)
    assert tools._bash_env() is not None
    assert levers.fire_counts().get("bash_env_guard", 0) == before
    monkeypatch.setenv("SOME_API_KEY", "k")
    tools._bash_env()
    assert levers.fire_counts().get("bash_env_guard", 0) == before + 1


def test_bash_env_guard_end_to_end(monkeypatch):
    seatbelt.set_context(False, None)
    monkeypatch.setenv("SOME_API_KEY", "sekrit-value")
    monkeypatch.setenv("HARMLESS_SETTING", "visible-value")
    out = tools.tool_bash("printenv SOME_API_KEY; printenv HARMLESS_SETTING")
    assert "sekrit-value" not in out
    assert "visible-value" in out


# -- the tool_bash seam -------------------------------------------------------

def test_tool_bash_unwrapped_runs_plain_shell(monkeypatch):
    seatbelt.set_context(False, None)
    out = tools.tool_bash("echo plain")
    assert "plain" in out


def test_tool_bash_denial_note_and_fire(monkeypatch):
    """A wrapped command whose output shows the EPERM marker gets the explanatory
    note appended and fires the lever exactly once."""
    fake = ["/bin/sh", "-c", "echo 'x: Operation not permitted'; exit 1"]
    monkeypatch.setattr(seatbelt, "wrap_argv", lambda cmd: fake)
    before = levers.fire_counts().get("yolo_seatbelt", 0)
    out = tools.tool_bash("anything")
    assert "Operation not permitted" in out
    assert "seatbelt:" in out
    assert levers.fire_counts().get("yolo_seatbelt", 0) == before + 1


def test_tool_bash_wrapped_clean_run_no_note(monkeypatch):
    fake = ["/bin/sh", "-c", "echo all good"]
    monkeypatch.setattr(seatbelt, "wrap_argv", lambda cmd: fake)
    before = levers.fire_counts().get("yolo_seatbelt", 0)
    out = tools.tool_bash("anything")
    assert "all good" in out
    assert "seatbelt:" not in out
    assert levers.fire_counts().get("yolo_seatbelt", 0) == before


def test_unwrapped_denial_output_gets_no_note(monkeypatch):
    """'Operation not permitted' from an UNSANDBOXED command (plain EPERM) must not
    be blamed on the seatbelt."""
    seatbelt.set_context(False, None)
    out = tools.tool_bash("echo 'y: Operation not permitted'")
    assert "seatbelt:" not in out


# -- real sandbox e2e (darwin only, skipped wherever Seatbelt can't apply) -----

# Gate on the real enforcement probe, not a permissive-profile smoke test: inside a
# CI/harness sandbox a permissive profile can still apply while a deny profile fails
# open — exactly the environment where these tests must skip, not fail.
_can_sandbox = seatbelt.probe()
seatbelt._probe_result = None  # leave module state pristine for the tests above


@pytest.mark.skipif(not _can_sandbox, reason="Seatbelt cannot apply here")
def test_e2e_denies_outside_write_allows_inside(tmp_path, monkeypatch):
    seatbelt.set_context(True, str(tmp_path))
    argv = seatbelt.wrap_argv(f"echo ok > {tmp_path}/in.txt")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and (tmp_path / "in.txt").read_text().strip() == "ok"

    probe = os.path.expanduser("~/chad_seatbelt_test_probe")
    argv = seatbelt.wrap_argv(f"touch {probe}")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    try:
        assert r.returncode != 0
        assert not os.path.exists(probe)
    finally:
        if os.path.exists(probe):  # belt failed: don't leave droppings
            os.unlink(probe)


@pytest.mark.skipif(not _can_sandbox, reason="Seatbelt cannot apply here")
def test_e2e_enforcement_probe_green():
    assert seatbelt.probe() is True


@pytest.mark.skipif(not _can_sandbox, reason="Seatbelt cannot apply here")
def test_e2e_protect_git_denies_gitdir_write(tmp_path):
    (tmp_path / ".git").mkdir()
    seatbelt.set_context(True, str(tmp_path))
    argv = seatbelt.wrap_argv(f"touch {tmp_path}/.git/droppings")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert r.returncode != 0
    assert not (tmp_path / ".git" / "droppings").exists()
    # the workspace around it stays writable
    argv = seatbelt.wrap_argv(f"echo ok > {tmp_path}/normal.txt")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and (tmp_path / "normal.txt").read_text().strip() == "ok"
