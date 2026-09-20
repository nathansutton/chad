"""Workspaces without a future.

A trial works in a real checkout of a real repository at the instance's base commit. The
danger is everything else that a clone normally carries: the branch the fix landed on,
the tag after it, the reflog, the remote it came from. A model that runs `git log --all`
in an ordinary clone of django can read the patch it was asked to write, and it would be
right to — the information was handed to it.

So a workspace is built by export, not by checkout. `git archive` writes the tree at the
base commit and nothing else; the destination then gets a fresh `git init` and exactly
one commit. There is no remote, no second commit, no object in the store that postdates
the task. The model still gets a working `git` — diffing, staging, stashing and
committing all behave — which matters, because taking git away would be a harness
difference rather than a fixed environment.

`assert_history_free` is the check, and it runs on every trial rather than once in a
test: this is the kind of safeguard that is worth almost nothing if it is only believed.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

CLONE_URLS = {
    "django/django": "https://github.com/django/django.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
}

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$", re.M)


def repos_dir(root: str) -> str:
    return os.path.join(root, "_repos")


def bare_path(root: str, repo: str) -> str:
    return os.path.join(repos_dir(root), repo.replace("/", "__") + ".git")


def _git(args: list[str], cwd: str = "", check: bool = True) -> str:
    out = subprocess.run(["git", *args], cwd=cwd or None, capture_output=True,
                         text=True, check=False)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def clone(root: str, repo: str) -> str:
    """One bare clone per repository, fetched once and reused by every trial."""
    dest = bare_path(root, repo)
    if os.path.isdir(dest):
        return dest
    os.makedirs(repos_dir(root), exist_ok=True)
    _git(["clone", "--bare", "--quiet", CLONE_URLS[repo], dest])
    return dest


def has_commit(root: str, repo: str, sha: str) -> bool:
    out = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                         cwd=bare_path(root, repo), capture_output=True, check=False)
    return out.returncode == 0


def export(root: str, repo: str, sha: str, dest: str) -> str:
    """The tree at `sha`, and only the tree, into an empty `dest`."""
    bare = bare_path(root, repo)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar = tmp.name
    try:
        _git(["archive", "--format=tar", "-o", tar, sha], cwd=bare)
        subprocess.run(["tar", "-xf", tar, "-C", dest], check=True)
    finally:
        os.unlink(tar)
    return dest


def materialize(root: str, repo: str, sha: str, dest: str, label: str) -> str:
    """Export the base commit and seed a single-commit repository over it.

    The `git init` is deliberately template-free and identity-pinned: a user's global
    git config can carry commit hooks, a signing key, or an `init.templateDir` that
    would make one machine's workspace differ from another's.
    """
    export(root, repo, sha, dest)
    with tempfile.TemporaryDirectory(prefix="swb-empty-template-") as empty:
        _git(["init", "--quiet", f"--template={empty}", "-b", "main"], cwd=dest)
    _git(["config", "user.email", "workbench@localhost"], cwd=dest)
    _git(["config", "user.name", "workbench"], cwd=dest)
    _git(["config", "commit.gpgsign", "false"], cwd=dest)
    _git(["add", "-A"], cwd=dest)
    _git(["commit", "--quiet", "--no-verify", "-m", f"{label} @ {sha[:12]}"], cwd=dest)
    assert_history_free(dest)
    return dest


def assert_history_free(dest: str) -> None:
    """One commit, no remotes, no other refs. Raises rather than warns."""
    count = _git(["rev-list", "--all", "--count"], cwd=dest).strip()
    if count != "1":
        raise RuntimeError(f"{dest}: workspace history has {count} commits, expected 1")
    remotes = _git(["remote"], cwd=dest).strip()
    if remotes:
        raise RuntimeError(f"{dest}: workspace has remotes {remotes!r}")
    refs = [r for r in _git(["for-each-ref", "--format=%(refname)"], cwd=dest).split()
            if r != "refs/heads/main"]
    if refs:
        raise RuntimeError(f"{dest}: workspace carries extra refs {refs}")


def diff(dest: str) -> str:
    """Everything the trial changed, as a patch the official harness can apply.

    Staged first so that files the model created are included; `--binary` so that an
    accidental binary change produces an applicable patch rather than a `Binary files
    differ` line that fails silently at grading time.
    """
    _git(["add", "-A"], cwd=dest)
    return _git(["diff", "--cached", "--binary", "HEAD"], cwd=dest)


def reset(dest: str) -> None:
    """Back to the seed commit — used between the two halves of a validation check."""
    _git(["checkout", "--", "."], cwd=dest)
    _git(["clean", "-qfdx"], cwd=dest)


def apply_patch(dest: str, patch: str) -> bool:
    """Apply a patch in the workspace. False when it does not apply cleanly."""
    if not patch.strip():
        return False
    out = subprocess.run(["git", "apply", "-v", "-"], cwd=dest, input=patch,
                         capture_output=True, text=True, check=False)
    return out.returncode == 0


def changed_files(patch: str) -> tuple[str, ...]:
    """Paths a patch touches, in the `b/` spelling (the post-image)."""
    return tuple(sorted({m.group("b") for m in _DIFF_HEADER.finditer(patch)}))
