"""The task catalog: 225 Exercism exercises in six languages, read from a pinned upstream.

The set is Aider's polyglot benchmark — the hardest exercises from six Exercism tracks,
picked upstream because frontier models were failing them. Nothing is vendored here:
`fetch()` clones the upstream repository at `UPSTREAM_COMMIT` into `_data/`, so the
exercises stay byte-identical to what everyone else measures and this repository does
not carry someone else's 225 problem sets.

An exercise's `.meta/config.json` names three file groups, and the whole kit is built on
keeping them apart:

* `solution` — the stub the model edits;
* `test` (plus `editor`, read-only helpers the tests import) — owned by the harness,
  restored from upstream before every verification;
* `example` — the reference solution. It lives under `.meta/`, which is never copied
  into a workspace, and is used only by the gold gate (`gold.py`).

Stdlib only, no model, no chad import: the catalog has to load on a machine that will
never run a trial.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing_extensions import TypeIs

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "_data")
UPSTREAM_URL = "https://github.com/Aider-AI/polyglot-benchmark"
UPSTREAM_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")

JsonValue = Union[None, bool, int, float, str, Sequence["JsonValue"], Mapping[str, "JsonValue"]]


def is_object(value: JsonValue) -> TypeIs[Mapping[str, JsonValue]]:
    return isinstance(value, Mapping)


def is_array(value: JsonValue) -> TypeIs[Sequence[JsonValue]]:
    return isinstance(value, Sequence) and not isinstance(value, str)


def is_text(value: JsonValue) -> TypeIs[str]:
    return isinstance(value, str)


def is_number(value: JsonValue) -> TypeIs[int | float]:
    """A JSON number. `bool` is an `int` in Python and is not one here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CatalogError(RuntimeError):
    """The upstream checkout is missing, at the wrong commit, or malformed."""


@dataclass(frozen=True)
class Task:
    """One exercise. Paths in the three groups are relative to `source`."""

    name: str                    # "<language>/<slug>", unique across the catalog
    language: str
    slug: str
    source: str                  # absolute path of the upstream exercise directory
    solution: tuple[str, ...]
    tests: tuple[str, ...]       # `test` + `editor`: everything the harness restores
    example: tuple[str, ...]


def upstream_dir() -> str:
    return os.path.join(DATA, "upstream")


def _git(args: list[str], cwd: str) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise CatalogError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def fetch() -> str:
    """Clone upstream at the pinned commit (idempotent) and return its path.

    Fetches the one commit by hash rather than a branch tip, so a force-push or a new
    exercise upstream cannot change what a run measures."""
    dest = upstream_dir()
    if not os.path.isdir(os.path.join(dest, ".git")):
        os.makedirs(dest, exist_ok=True)
        _git(["init", "-q"], dest)
        _git(["remote", "add", "origin", UPSTREAM_URL], dest)
    if _head(dest) != UPSTREAM_COMMIT:
        _git(["fetch", "-q", "--depth", "1", "origin", UPSTREAM_COMMIT], dest)
        _git(["checkout", "-q", "--detach", "FETCH_HEAD"], dest)
    return dest


def _head(dest: str) -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True,
                         text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 else ""


def _paths(files: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    group = files.get(key)
    if group is None:
        return ()
    if not is_array(group):
        raise CatalogError(f"`files.{key}` is not a list")
    return tuple(p for p in group if is_text(p))


def _read_task(language: str, slug: str, source: str) -> Task:
    with open(os.path.join(source, ".meta", "config.json"), encoding="utf-8") as f:
        config: JsonValue = json.load(f)
    files = config.get("files") if is_object(config) else None
    if not is_object(files):
        raise CatalogError(f"{language}/{slug}: config.json has no `files` object")
    task = Task(name=f"{language}/{slug}", language=language, slug=slug, source=source,
                solution=_paths(files, "solution"),
                tests=_paths(files, "test") + _paths(files, "editor"),
                example=_paths(files, "example"))
    if not (task.solution and task.tests and task.example):
        raise CatalogError(f"{task.name}: a file group is empty")
    return task


def load_tasks(languages: Sequence[str] = LANGUAGES) -> list[Task]:
    """Every exercise of `languages`, sorted by name. Refuses a checkout that is not the
    pinned commit: a catalog that silently drifted is a different benchmark."""
    root = upstream_dir()
    if _head(root) != UPSTREAM_COMMIT:
        raise CatalogError(
            f"no upstream checkout at {UPSTREAM_COMMIT[:12]} under {root}; "
            "run `python benchmarks/polyglot/prepare.py` first")
    tasks = []
    for language in languages:
        practice = os.path.join(root, language, "exercises", "practice")
        for slug in sorted(os.listdir(practice)):
            source = os.path.join(practice, slug)
            if os.path.isfile(os.path.join(source, ".meta", "config.json")):
                tasks.append(_read_task(language, slug, source))
    return tasks


def manifest_path() -> str:
    return os.path.join(ROOT, "manifest.json")


def load_manifest() -> list[str]:
    """Names of the tasks that passed the gold gate, as committed. A run is scored over
    exactly this list, so two people's pass rates share a denominator."""
    with open(manifest_path(), encoding="utf-8") as f:
        doc: JsonValue = json.load(f)
    names = doc.get("tasks") if is_object(doc) else None
    if not is_array(names):
        raise CatalogError("manifest.json has no `tasks` list")
    return [n for n in names if is_text(n)]
