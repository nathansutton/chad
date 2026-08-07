"""Shadow-git checkpoints (checkpoint.py): snapshot → mutate → restore round trips.

Contract: the user's own .git is never opened or written; untracked-in-user-repo
files ARE snapshotted (they're exactly what an edit can destroy); restore puts
snapshotted content back but never deletes files created after the snapshot; every
failure path returns a value instead of raising (an edit must not die because the
checkpoint machinery hiccuped). The shadow lives under ~/.chad/history — pointed at
a tmp HOME here so tests never touch the real one.
"""
import os
import subprocess

import pytest

from chad import checkpoint


@pytest.fixture(autouse=True)
def _own_checkpoint_dir(tmp_path, monkeypatch):
    # conftest already redirects CHAD_CHECKPOINT_DIR for the whole suite; pin it
    # here too so this file stands alone (its assertions depend on an empty root).
    monkeypatch.setenv("CHAD_CHECKPOINT_DIR", str(tmp_path / "ckpt"))


@pytest.fixture()
def ws(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "a.py").write_text("A1\n")
    (d / "b.py").write_text("B1\n")
    return d


def test_snapshot_returns_hash_and_shadow_is_outside_ws(ws):
    ref = checkpoint.snapshot(str(ws), "before edit a.py")
    assert ref
    assert not (ws / ".git").exists()
    assert not checkpoint.shadow_dir(str(ws)).startswith(str(ws))
    assert checkpoint.shadow_dir(str(ws)).startswith(
        os.environ["CHAD_CHECKPOINT_DIR"])


def test_round_trip_restores_content(ws):
    checkpoint.snapshot(str(ws), "before edit")
    (ws / "a.py").write_text("A2 clobbered\n")
    (ws / "b.py").unlink()
    msg = checkpoint.restore(str(ws))
    assert msg.startswith("restored")
    assert (ws / "a.py").read_text() == "A1\n"
    assert (ws / "b.py").read_text() == "B1\n"


def test_restore_leaves_files_created_after_snapshot(ws):
    checkpoint.snapshot(str(ws), "s1")
    (ws / "new.py").write_text("created later\n")
    msg = checkpoint.restore(str(ws))
    assert msg.startswith("restored")
    assert (ws / "new.py").read_text() == "created later\n"


def test_restore_to_named_earlier_snapshot(ws):
    checkpoint.snapshot(str(ws), "s1")
    (ws / "a.py").write_text("A2\n")
    first = checkpoint.snapshots(str(ws))[-1][0]
    checkpoint.snapshot(str(ws), "s2")
    (ws / "a.py").write_text("A3\n")
    msg = checkpoint.restore(str(ws), first)
    assert msg.startswith("restored")
    assert (ws / "a.py").read_text() == "A1\n"


def test_unchanged_tree_snapshot_returns_prior_hash(ws):
    r1 = checkpoint.snapshot(str(ws), "s1")
    r2 = checkpoint.snapshot(str(ws), "s2 nothing changed")
    assert r1 == r2


def test_snapshots_listing_newest_first(ws):
    checkpoint.snapshot(str(ws), "first")
    (ws / "a.py").write_text("A2\n")
    checkpoint.snapshot(str(ws), "second")
    rows = checkpoint.snapshots(str(ws))
    assert [r[2] for r in rows] == ["second", "first"]


def test_user_git_repo_untouched(ws):
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    subprocess.run(["git", "-C", str(ws), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(ws), "-c", "user.email=u@u", "-c",
                    "user.name=u", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "user commit"], check=True)
    head_before = subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout
    checkpoint.snapshot(str(ws), "shadow snap")
    (ws / "a.py").write_text("A2\n")
    checkpoint.restore(str(ws))
    head_after = subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout
    log = subprocess.run(["git", "-C", str(ws), "log", "--format=%s"],
                         capture_output=True, text=True, check=True).stdout
    assert head_before == head_after
    assert "shadow snap" not in log


def test_default_excludes_keep_junk_out(ws):
    (ws / ".venv").mkdir()
    (ws / ".venv" / "huge.bin").write_text("x" * 10)
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "a.pyc").write_text("x")
    checkpoint.snapshot(str(ws), "s1")
    files = subprocess.run(
        ["git", "--git-dir", checkpoint.shadow_dir(str(ws)), "ls-tree", "-r",
         "--name-only", "HEAD"], capture_output=True, text=True, check=False).stdout
    assert ".venv" not in files and "__pycache__" not in files
    assert "a.py" in files


def test_restore_without_snapshots_is_a_message_not_a_crash(ws):
    msg = checkpoint.restore(str(ws))
    assert "no checkpoints" in msg


def test_restore_unknown_ref_is_a_message(ws):
    checkpoint.snapshot(str(ws), "s1")
    msg = checkpoint.restore(str(ws), "deadbeef")
    assert "no checkpoint named" in msg


def test_snapshot_failure_returns_none(ws, monkeypatch):
    monkeypatch.setattr(checkpoint, "shadow_dir",
                        lambda _ws: "/dev/null/not/a/dir/shadow.git")
    assert checkpoint.snapshot(str(ws), "s") is None
