"""Download a published run into `_runs/<label>/` and prove it is the run RUNS.md names.

    uv run python benchmarks/polyglot/fetch.py --label baseline
    python benchmarks/polyglot/stats.py compare _runs/baseline/trials.jsonl _runs/mine/trials.jsonl

`RUNS.md` is the ledger: each row names a bundle and the sha256 of its `trials.jsonl`. The
download is checked against that hash before it lands, so `stats.py compare` never pairs a
local arm against rows that differ from the ones a write-up cited. A `file://` path works
too, for a bundle that was never uploaded or a mirror of one.

Never overwrites: a label already under `_runs/` is someone's run, fetched or not.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import posixpath
import shutil
import sys
import tarfile
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog  # noqa: E402
from publish import TRAJECTORIES  # noqa: E402

LEDGER = os.path.join(catalog.ROOT, "RUNS.md")


class FetchError(RuntimeError):
    """The ledger has no such run, or the bundle is not the one it records."""


@dataclass(frozen=True)
class Entry:
    label: str
    dataset_path: str            # hf://datasets/<owner>/<repo>/<path> or file:///<dir>
    sha256: str


def read_ledger(path: str) -> dict[str, Entry]:
    """RUNS.md's table, keyed by label. Columns are found by header, not position."""
    entries: dict[str, Entry] = {}
    header: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not header:
                header = cells
            elif not set("".join(cells)) <= set("-: "):
                row = dict(zip(header, cells))
                entries[row["label"]] = Entry(row["label"], row["dataset path"],
                                              row["trials.jsonl sha256"])
    return entries


def locate(dataset_path: str) -> str:
    """A local directory holding the bundle: the path itself for `file://`, a download into
    the Hugging Face cache for `hf://datasets/`."""
    if dataset_path.startswith("file://"):
        return dataset_path.removeprefix("file://")
    prefix = "hf://datasets/"
    if not dataset_path.startswith(prefix):
        raise FetchError(f"unsupported dataset path: {dataset_path}")
    owner, repo, sub = dataset_path.removeprefix(prefix).split("/", 2)
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(repo_id=f"{owner}/{repo}", repo_type="dataset",
                                 allow_patterns=[f"{sub}/*"])
    return os.path.join(snapshot, sub)


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _unpack(archive: str, dest: str) -> None:
    """Regular files under `trajectories/` only: a member that names an absolute path, a
    parent directory or a link is refused rather than trusted."""
    with tarfile.open(archive, mode="r:xz") as tar:
        for member in tar.getmembers():
            name = posixpath.normpath(member.name)
            if not (member.isfile() and name.startswith("trajectories/")):
                raise FetchError(f"{archive}: refusing archive member {member.name!r}")
            source = tar.extractfile(member)
            if source is None:
                raise FetchError(f"{archive}: unreadable member {member.name!r}")
            target = os.path.join(dest, *name.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)


def fetch(entry: Entry, runs_dir: str) -> str:
    """Verify, then land the bundle at `runs_dir/<label>/` in the layout `run.py` writes."""
    dest = os.path.join(runs_dir, entry.label)
    if os.path.exists(dest):
        raise FetchError(f"{dest} already exists; move it aside or fetch under another name")
    source = locate(entry.dataset_path)
    got = _sha256(os.path.join(source, "trials.jsonl"))
    if got != entry.sha256:
        raise FetchError(f"{entry.label}: trials.jsonl sha256 is {got}, RUNS.md records "
                         f"{entry.sha256}")
    staging = dest + ".part"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    for name in ("trials.jsonl", "meta.json"):
        shutil.copyfile(os.path.join(source, name), os.path.join(staging, name))
    archive = os.path.join(source, TRAJECTORIES)
    if os.path.exists(archive):
        _unpack(archive, staging)
    os.replace(staging, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="a row of RUNS.md")
    ap.add_argument("--ledger", default=LEDGER)
    args = ap.parse_args()
    entries = read_ledger(args.ledger)
    if args.label not in entries:
        raise SystemExit(f"{args.label} is not in {args.ledger}")
    try:
        dest = fetch(entries[args.label], os.path.join(catalog.ROOT, "_runs"))
    except FetchError as e:
        raise SystemExit(str(e)) from None
    print(f"fetched {args.label} into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
