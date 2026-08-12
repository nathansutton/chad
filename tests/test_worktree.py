"""Unit tests for worktree-per-session isolation (worktree.py) — lifecycle + apply
round-trips against real throwaway git repos. No model, no network.

Run: `uv run python -m pytest tests/test_worktree.py`
"""
import os
import subprocess

import pytest

from chad import worktree


def _git(cwd, *args, input_text=None):
    return subprocess.run(
        ["git", "-C", cwd, "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, input=input_text, check=False)


@pytest.fixture(autouse=True)
def _wt_root(tmp_path, monkeypatch):
    """Point the worktree store at a per-test dir and clear the active-worktree
    global — module state must not leak between tests."""
    monkeypatch.setenv("CHAD_WORKTREE_DIR", str(tmp_path / "wt"))
    monkeypatch.setattr(worktree, "_ACTIVE", None)
    yield


@pytest.fixture()
def repo(tmp_path):
    """A committed git repo with one file and one subdir."""
    r = tmp_path / "repo"
    (r / "pkg").mkdir(parents=True)
    (r / "a.txt").write_text("one\n")
    (r / "pkg" / "b.txt").write_text("two\n")
    _git(str(tmp_path), "init", "-q", "-b", "main", str(r))
    _git(str(r), "add", "-A")
    _git(str(r), "commit", "-q", "-m", "init")
    return str(r)


# -- eligibility --------------------------------------------------------------

def test_eligibility_non_git(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    ok, reason = worktree.eligibility(str(d))
    assert not ok and "not a git repository" in reason


def test_eligibility_clean_repo(repo):
    ok, reason = worktree.eligibility(repo)
    assert ok, reason


def test_eligibility_dirty_repo(repo):
    with open(os.path.join(repo, "a.txt"), "w") as f:
        f.write("edited\n")
    ok, reason = worktree.eligibility(repo)
    assert not ok and "uncommitted" in reason


def test_eligibility_untracked_counts_as_dirty(repo):
    # An untracked file could be exactly what the task is about; a worktree from
    # HEAD would not see it, so it must force in-place mode too.
    with open(os.path.join(repo, "new.txt"), "w") as f:
        f.write("scratch\n")
    ok, reason = worktree.eligibility(repo)
    assert not ok


def test_eligibility_empty_repo(tmp_path):
    r = tmp_path / "empty"
    r.mkdir()
    _git(str(tmp_path), "init", "-q", "-b", "main", str(r))
    ok, reason = worktree.eligibility(str(r))
    assert not ok and "no commits" in reason


def test_no_nesting_inside_chad_worktree(repo):
    meta, err = worktree.create(repo)
    assert meta, err
    ok, reason = worktree.eligibility(meta["path"])
    assert not ok and "already inside" in reason


# -- create / enter / adopt ---------------------------------------------------

def test_create_makes_branch_and_sidecar(repo):
    meta, err = worktree.create(repo)
    assert meta, err
    assert os.path.isdir(meta["path"])
    assert meta["branch"] == "chad/" + meta["id"]
    assert meta["origin_top"] == os.path.realpath(repo)
    assert meta["origin_rel"] == ""
    # the branch exists and points at base
    r = _git(repo, "rev-parse", meta["branch"])
    assert r.stdout.strip() == meta["base"]
    # sidecar round-trips through list/find
    assert worktree.find(repo, meta["id"])["path"] == meta["path"]
    # worktree contains the committed files
    assert os.path.isfile(os.path.join(meta["path"], "pkg", "b.txt"))


def test_create_from_subdir_records_rel(repo):
    sub = os.path.join(repo, "pkg")
    meta, err = worktree.create(sub)
    assert meta, err
    assert meta["origin_rel"] == "pkg"
    dest = worktree.enter(meta)
    assert dest == os.path.join(meta["path"], "pkg")
    assert worktree.session_anchor() == os.path.join(meta["origin_top"], "pkg")


def test_enter_sets_active_and_anchor(repo, monkeypatch):
    meta, _ = worktree.create(repo)
    dest = worktree.enter(meta)
    assert dest and os.path.realpath(os.getcwd()) == os.path.realpath(dest)
    assert worktree.current()["id"] == meta["id"]
    assert worktree.current_ref()["branch"] == meta["branch"]
    assert worktree.session_anchor() == meta["origin_top"]


def test_anchor_without_worktree_is_cwd(repo, monkeypatch):
    monkeypatch.chdir(repo)
    assert worktree.session_anchor() == os.getcwd()
    assert worktree.current_ref() is None


def test_adopt_from_inside_worktree(repo):
    meta, _ = worktree.create(repo)
    assert worktree.adopt(repo) is None          # origin checkout: nothing to adopt
    got = worktree.adopt(meta["path"])
    assert got and got["id"] == meta["id"]
    assert worktree.current()["id"] == meta["id"]


# -- diff / apply / remove ----------------------------------------------------

def _mutate(wt):
    """Edit + add + delete, the three shapes apply must carry."""
    with open(os.path.join(wt, "a.txt"), "w") as f:
        f.write("changed\n")
    with open(os.path.join(wt, "fresh.txt"), "w") as f:
        f.write("brand new\n")
    os.remove(os.path.join(wt, "pkg", "b.txt"))


def test_diff_stat_counts_all_change_shapes(repo):
    meta, _ = worktree.create(repo)
    n, stat = worktree.diff_stat(meta)
    assert n == 0 and stat == ""
    _mutate(meta["path"])
    n, stat = worktree.diff_stat(meta)
    assert n == 3, stat
    assert "fresh.txt" in stat


def test_apply_round_trip(repo):
    meta, _ = worktree.create(repo)
    _mutate(meta["path"])
    ok, msg = worktree.apply(meta)
    assert ok, msg
    assert "3" in msg
    with open(os.path.join(repo, "a.txt")) as f:
        assert f.read() == "changed\n"
    with open(os.path.join(repo, "fresh.txt")) as f:
        assert f.read() == "brand new\n"
    assert not os.path.exists(os.path.join(repo, "pkg", "b.txt"))
    # apply mutates the origin WORKING TREE only — never commits, never switches branch
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == meta["base"]
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


def test_apply_includes_worktree_commits(repo):
    # A session may run `git commit` itself mid-task; the diff is vs base, so
    # committed and uncommitted work land together.
    meta, _ = worktree.create(repo)
    with open(os.path.join(meta["path"], "a.txt"), "w") as f:
        f.write("committed change\n")
    _git(meta["path"], "add", "-A")
    _git(meta["path"], "commit", "-q", "-m", "wip")
    with open(os.path.join(meta["path"], "fresh.txt"), "w") as f:
        f.write("uncommitted\n")
    ok, msg = worktree.apply(meta)
    assert ok, msg
    with open(os.path.join(repo, "a.txt")) as f:
        assert f.read() == "committed change\n"
    assert os.path.isfile(os.path.join(repo, "fresh.txt"))


def test_apply_leaves_changes_unstaged(repo):
    # The supervision contract: what comes back is reviewable with plain `git diff`,
    # so the common (undrifted) apply must not touch the origin index.
    meta, _ = worktree.create(repo)
    with open(os.path.join(meta["path"], "a.txt"), "w") as f:
        f.write("changed\n")
    ok, msg = worktree.apply(meta)
    assert ok, msg
    staged = _git(repo, "diff", "--cached", "--name-only").stdout.strip()
    assert staged == "", f"apply staged files: {staged}"
    assert "a.txt" in _git(repo, "diff", "--name-only").stdout


def test_apply_3way_when_origin_drifted_compatibly(repo):
    # Origin gained an unrelated commit after the worktree was cut: plain apply may
    # still work; if not, the 3-way fallback must land the change rather than fail.
    meta, _ = worktree.create(repo)
    with open(os.path.join(meta["path"], "a.txt"), "w") as f:
        f.write("worktree change\n")
    with open(os.path.join(repo, "other.txt"), "w") as f:
        f.write("origin moved on\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "drift")
    ok, msg = worktree.apply(meta)
    assert ok, msg
    with open(os.path.join(repo, "a.txt")) as f:
        assert f.read() == "worktree change\n"
    with open(os.path.join(repo, "other.txt")) as f:
        assert f.read() == "origin moved on\n"


def test_apply_no_changes(repo):
    meta, _ = worktree.create(repo)
    ok, msg = worktree.apply(meta)
    assert ok and "no changes" in msg


def test_apply_conflict_keeps_worktree(repo):
    meta, _ = worktree.create(repo)
    with open(os.path.join(meta["path"], "a.txt"), "w") as f:
        f.write("worktree version\n")
    # origin diverges incompatibly on the same line after the worktree was cut
    with open(os.path.join(repo, "a.txt"), "w") as f:
        f.write("origin version\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "diverge")
    ok, msg = worktree.apply(meta)
    assert not ok, msg
    # the worktree (and the work) survives a failed apply
    assert os.path.isdir(meta["path"])
    with open(os.path.join(meta["path"], "a.txt")) as f:
        assert f.read() == "worktree version\n"


def test_remove_cleans_checkout_branch_sidecar(repo):
    meta, _ = worktree.create(repo)
    worktree.enter(meta)
    os.chdir(repo)  # cannot remove the cwd out from under the process
    msg = worktree.remove(meta)
    assert msg.startswith("removed"), msg
    assert not os.path.isdir(meta["path"])
    assert _git(repo, "rev-parse", "--verify", "--quiet",
                meta["branch"]).returncode != 0
    assert worktree.find(repo, meta["id"]) is None
    assert worktree.current() is None  # removing the active worktree deactivates it


def test_list_prunes_stale_sidecars(repo):
    m1, _ = worktree.create(repo)
    m2, _ = worktree.create(repo)
    assert {m["id"] for m in worktree.list_for(repo)} == {m1["id"], m2["id"]}
    # simulate a user rm -rf'ing the checkout behind our back
    _git(repo, "worktree", "remove", "--force", m1["path"])
    metas = worktree.list_for(repo)
    assert [m["id"] for m in metas] == [m2["id"]]
    assert worktree.find(repo, m1["id"]) is None


def test_find_by_suffix(repo):
    meta, _ = worktree.create(repo)
    assert worktree.find(repo, meta["id"][-4:])["id"] == meta["id"]
    assert worktree.find(repo, "nope-nope") is None


def test_describe_mentions_changes(repo):
    meta, _ = worktree.create(repo)
    _mutate(meta["path"])
    d = worktree.describe(meta)
    assert meta["id"] in d and "3 file(s) changed" in d
