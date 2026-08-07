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
import sys

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


def test_profile_worktree_gitdir_carveout(tmp_path):
    ws = tmp_path / "wt"
    ws.mkdir()
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n")
    (ws / ".git").write_text(f"gitdir: {gitdir}\n")
    text = seatbelt.profile_text(str(ws))
    assert str(gitdir.resolve()) in text
    common = (gitdir / "../..").resolve()
    assert str(common) in text


def test_profile_plain_repo_no_carveout(tmp_path):
    (tmp_path / ".git").mkdir()  # normal repo: .git is a dir, no pointer to chase
    assert seatbelt._worktree_gitdirs(str(tmp_path)) == []


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

_can_sandbox = (sys.platform == "darwin" and os.path.exists(seatbelt.SANDBOX_EXEC)
                and subprocess.run([seatbelt.SANDBOX_EXEC, "-p",
                                    "(version 1)(allow default)", "/usr/bin/true"],
                                   capture_output=True, check=False).returncode == 0)


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
