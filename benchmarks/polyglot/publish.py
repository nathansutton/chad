"""Bundle a finished run for publication: rows always, trajectories on request, no local paths.

    uv run python benchmarks/polyglot/publish.py --label baseline
    uv run python benchmarks/polyglot/publish.py --label baseline --with-trajectories
    uv run python benchmarks/polyglot/publish.py --label baseline --upload

Run output never enters git. A run a write-up cites is published as a bundle — `meta.json`,
`trials.jsonl` and, when asked, `trajectories.tar.xz` — to a dataset repository, and one
hand-pasted row of `RUNS.md` points at it with the sha256 of its rows. Nothing is uploaded
without `--upload`.

A run records where the agent stood: every workspace path, the model directory, the
interpreter, chad's spill files. Before anything is written, each file is rewritten so the
trial's workspace is `.`, a CLI arm's throwaway home is `<home>` (including the `~` such an
arm writes, which is that home), the kit is `<kit>`, the checkout is `<repo>` and the home
directory is `~`, and then checked. The bundle is refused if a `/Users/<name>` path or the
home directory survives, if a path under `~` leads somewhere a trial has no business being, or if
a string looks like a credential. A refusal names the file: the fix is a look at that trial,
not a looser rule.

Stdlib only, except `--upload`, which needs `huggingface_hub` (already a chad dependency).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog  # noqa: E402
from catalog import JsonValue, is_number, is_object, is_text  # noqa: E402

DATASET = "nathansutton/chad-polyglot-runs"
TRAJECTORIES = "trajectories.tar.xz"
RUNS_HEADER = ("| label | date | chad | harness | model | tasks × reps | dataset path "
               "| trials.jsonl sha256 |")

# A path segment inside a JSON string: it ends at a slash, a quote, whitespace or the
# backslash of an escape.
_SEGMENT = r"[^/\s\"'\\]+"
# A prefix only matches whole: `/Users/nate` must not rewrite `/Users/nathan`.
_WHOLE = r"(?![\w.-])"


def _trial_dir(kind: str) -> re.Pattern[str]:
    """A trial's `_work/` or `_home/` directory under whichever root its arm ran in: the
    kit for chad in process, the system temp directory for a CLI arm. A match begins
    where an absolute path begins, so a relative mention of one stays as written."""
    return re.compile(rf"(?<![\w.~-])(?:/{_SEGMENT})*?/_{kind}/{_SEGMENT}/rep\d+"
                      rf"/{_SEGMENT}/{_SEGMENT}{_WHOLE}")


_WORKSPACE, _TRIAL_HOME = _trial_dir("work"), _trial_dir("home")
_HOME_PATH = re.compile(r"(?<![\w.])~/[^\s\"'\\]*")
# A concrete account's directory. The model's own elisions (`/Users/.../bob/`) name nobody.
_USER_DIR = re.compile(r"/Users/[\w-]")
# Where a trial reaches under `~` by design: chad's own spill files and the toolchains'
# package caches. Anything else under the home directory is a trial worth reading first.
_HOME_ALLOWED = ("~/.cache/chad/", "~/.cargo/", "~/.rustup/", "~/go/pkg/")
_SECRET = re.compile(
    r"AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}"
    r"|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")


class PublishError(RuntimeError):
    """A bundle would carry something that must not leave this machine."""


@dataclass(frozen=True)
class Roots:
    """The absolute prefixes a run embeds. Injected so a test can use its own."""

    home: str
    repo: str
    kit: str

    @classmethod
    def local(cls) -> Roots:
        return cls(home=os.path.expanduser("~"),
                   repo=os.path.dirname(os.path.dirname(catalog.ROOT)), kit=catalog.ROOT)


@dataclass(frozen=True)
class Bundle:
    path: str
    sha256: str                  # of trials.jsonl, the value RUNS.md records
    runs_row: str


def redact(text: str, roots: Roots, isolated: bool = False) -> str:
    """Every local prefix rewritten, most specific first. `isolated`: the arm ran in a
    throwaway home (`harness/cli.py`), so a `~` in what it wrote — a harness describing
    its own config directory, say — is that home, not the maintainer's, and is written
    as such rather than left to look like a path into someone's account."""
    text = _WORKSPACE.sub(".", text)
    text = _TRIAL_HOME.sub("<home>", text)
    if isolated:
        # Before the home prefix is folded to `~`, or a real path into the maintainer's
        # account would end up spelled as the throwaway home and stop being a refusal.
        text = _HOME_PATH.sub(lambda m: "<home>" + m.group(0)[1:], text)
    for prefix, name in ((roots.kit, "<kit>"), (roots.repo, "<repo>"), (roots.home, "~")):
        text = re.sub(re.escape(prefix) + _WHOLE, name, text)
    return text


def problems(text: str, roots: Roots) -> list[str]:
    """Why `text` (already redacted) cannot be published; empty when it can."""
    found = []
    if _USER_DIR.search(text) or roots.home in text:
        found.append("an absolute home path survived the rewrite")
    outside = sorted({p for p in _HOME_PATH.findall(text) if not p.startswith(_HOME_ALLOWED)})
    if outside:
        found.append("paths outside the workspace: " + ", ".join(outside[:5]))
    if _SECRET.search(text):
        found.append("a credential-shaped string")
    return found


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_meta(text: str) -> dict[str, JsonValue]:
    doc: JsonValue = json.loads(text)
    return dict(doc) if is_object(doc) else {}


def _row_shape(rows: str) -> tuple[int, int]:
    """(tasks, reps) as the rows actually are, which a block cut short makes smaller than
    what `meta.json` planned."""
    tasks: set[str] = set()
    reps = 0
    for line in rows.splitlines():
        if not line.strip():
            continue
        row: JsonValue = json.loads(line)
        if is_object(row):
            task, rep = row.get("task"), row.get("rep")
            if is_text(task):
                tasks.add(task)
            if is_number(rep):
                reps = max(reps, int(rep))
    return len(tasks), reps


def runs_row(label: str, meta: dict[str, JsonValue], rows: str, sha256: str, dataset: str) -> str:
    def text(key: str, default: str = "") -> str:
        value = meta.get(key)
        return value if is_text(value) else default

    model = text("model")
    if model.startswith(("<repo>", "~", "/")):
        model = os.path.basename(model)
    dirty = "+dirty" if meta.get("git_dirty") is True else ""
    chad = f"{text('chad_version')} ({text('git_rev')}{dirty})"
    tasks, reps = _row_shape(rows)
    return (f"| {label} | {text('started')[:10]} | {chad} | {text('harness', 'chad')} | {model} "
            f"| {tasks} × {reps} | hf://datasets/{dataset}/polyglot/{label} | {sha256} |")


def _tar(files: list[tuple[str, bytes]]) -> bytes:
    """A reproducible archive: same trajectories, same bytes, same hash."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def bundle(run_dir: str, out_dir: str, roots: Roots, with_trajectories: bool = False,
           dataset: str = DATASET) -> Bundle:
    """Rewrite and check every file of the run in memory, then write the bundle. Nothing is
    written unless every file passes, so a refused run leaves no half-bundle behind."""
    label = os.path.basename(os.path.normpath(run_dir))
    sources = [("trials.jsonl", os.path.join(run_dir, "trials.jsonl")),
               ("meta.json", os.path.join(run_dir, "meta.json"))]
    if with_trajectories:
        base = os.path.join(run_dir, "trajectories")
        for root, _dirs, files in sorted(os.walk(base)):
            sources += [(os.path.relpath(os.path.join(root, f), run_dir), os.path.join(root, f))
                        for f in sorted(files) if f.endswith(".json")]
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        # An arm with its own throwaway home says so; chad in process does not.
        isolated = "isolation" in _read_meta(f.read())
    clean: dict[str, str] = {}
    refused = []
    for name, path in sources:
        with open(path, encoding="utf-8") as f:
            text = redact(f.read(), roots, isolated)
        refused += [f"{name}: {why}" for why in problems(text, roots)]
        clean[name] = text
    if refused:
        raise PublishError(f"{label} not bundled:\n  " + "\n  ".join(refused))

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    for name in ("trials.jsonl", "meta.json"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(clean[name])
    trajectories = [(n, t.encode()) for n, t in clean.items() if n.startswith("trajectories/")]
    if trajectories:
        with open(os.path.join(out_dir, TRAJECTORIES), "wb") as archive:
            archive.write(_tar(trajectories))
    sha256 = _sha256(clean["trials.jsonl"].encode())
    return Bundle(out_dir, sha256, runs_row(label, _read_meta(clean["meta.json"]),
                                            clean["trials.jsonl"], sha256, dataset))


def upload(b: Bundle, label: str, dataset: str) -> None:
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=dataset, repo_type="dataset", folder_path=b.path,
                          path_in_repo=f"polyglot/{label}", commit_message=f"polyglot: {label}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="the run under _runs/<label>/")
    ap.add_argument("--with-trajectories", action="store_true",
                    help="also bundle every trajectory (rows alone carry what stats.py needs)")
    ap.add_argument("--dataset", default=DATASET, help="dataset repo id for the RUNS.md row")
    ap.add_argument("--upload", action="store_true", help="push the bundle to the dataset repo")
    args = ap.parse_args()

    run_dir = os.path.join(catalog.ROOT, "_runs", args.label)
    out_dir = os.path.join(catalog.ROOT, "_publish", args.label)
    try:
        b = bundle(run_dir, out_dir, Roots.local(), args.with_trajectories, args.dataset)
    except PublishError as e:
        raise SystemExit(str(e)) from None
    print(f"bundle: {b.path}\ntrials.jsonl sha256: {b.sha256}\n\nRUNS.md row:\n{b.runs_row}")
    if args.upload:
        upload(b, args.label, args.dataset)
        print(f"uploaded to hf://datasets/{args.dataset}/polyglot/{args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
