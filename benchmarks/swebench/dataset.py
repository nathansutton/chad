"""The benchmark, pinned: 50 SWE-bench Verified instances, fetched once and split in two.

`MariusHobbhahn/swe-bench-verified-mini` is a published 50-instance subset of SWE-bench
Verified — 25 django/django (all at version 3.1, so one environment) and 25
sphinx-doc/sphinx spread over 12 versions. It is pinned here by commit revision AND by
the sha256 of the parquet, because a benchmark that can be edited under you is not a
benchmark: the digest is what a reader re-checks before believing any number in the kit.

THE SPLIT IS THE POINT
----------------------
`fetch()` writes two files, and nothing reads the wrong one:

  _data/instances.jsonl   everything — gold patch, test patch, the test lists. Read by
                          `prepare.py validate` and by `grade.py`, which run when no
                          model is resident.
  _data/tasks.jsonl       what the solver is allowed to see: id, repo, base commit,
                          version, problem statement. No `patch`, no `test_patch`, no
                          `hints_text`, no test names.

`run.py` imports `load_tasks` and there is no gold field for it to leak by accident. A
solver that cannot read the answer cannot be accused of having read it, and that is
cheaper to defend than any amount of care taken at the call site. `hints_text` is
dropped at fetch time rather than merely unused: the field never lands on this disk.

CONTAMINATION, SAID PLAINLY
---------------------------
These issues are dated 2017-2023 and both repos are famous. The model has very likely
seen them, and probably some of the fixes. That is not fatal for the comparison this kit
exists to make — it is chad against chad on the same 50 tasks, so memorized solutions
cancel — but it does cost discrimination: a task the model simply knows lands at 5/5 for
both arms and carries no information about the harness. It shrinks the band of tasks
that can go either way, which is the same as shrinking the effective sample.

Parquet is read once, at fetch time, by an ephemeral `uv` subprocess. pyarrow does not
enter this project's dependencies and nothing in the kit needs it again: every later
step reads jsonl with the standard library.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import urllib.request

DATASET_ID = "MariusHobbhahn/swe-bench-verified-mini"
# The dataset commit, not `main`: `resolve/main` would follow a re-upload.
REVISION = "b316c349947c29963fce3f4a65967c9807a4b673"
PARQUET_PATH = "data/test-00000-of-00001.parquet"
PARQUET_SHA256 = "f9ba19dea78884f1081355d2d8afb671899981f24180aa0c4c1aa14d2c23e855"
URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{PARQUET_PATH}"

# What the grader is pointed at. The mini set is a subset of Verified by instance id, so
# the official harness grades it against the parent dataset and never needs the mirror.
GRADING_DATASET = "princeton-nlp/SWE-bench_Verified"

N_INSTANCES = 50

# Fields that must never reach the solver's view. Named, rather than listing the fields
# that may, so that a column added upstream is excluded by default.
_GOLD_FIELDS = ("patch", "test_patch", "hints_text", "FAIL_TO_PASS", "PASS_TO_PASS")
_TASK_FIELDS = ("instance_id", "repo", "base_commit", "version", "problem_statement")


@dataclasses.dataclass(frozen=True)
class Task:
    """What the solver sees. Constructed only from `tasks.jsonl`."""
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str

    @property
    def repo_key(self) -> str:
        """`django__django` — the instance-id prefix, and the bare clone's directory."""
        return self.repo.replace("/", "__")

    @property
    def env_key(self) -> str:
        """`django__django-3.1` — one native workbench environment per repo+version."""
        return f"{self.repo_key}-{self.version}"


@dataclasses.dataclass(frozen=True)
class Instance:
    """The full row: everything `Task` has, plus the answer and the test lists."""
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str
    environment_setup_commit: str
    patch: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]

    @property
    def repo_key(self) -> str:
        return self.repo.replace("/", "__")

    @property
    def env_key(self) -> str:
        return f"{self.repo_key}-{self.version}"


def data_dir(root: str) -> str:
    return os.path.join(root, "_data")


def _download(dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_to_rows(parquet: str) -> list[dict]:
    """Read the parquet in a throwaway interpreter and hand back plain dicts.

    pyarrow is a large dependency that this kit needs exactly once, so it is rented for
    one subprocess rather than added to the project. The child prints jsonl on stdout;
    a failure here is a hard error, never a partial dataset.
    """
    script = (
        "import json, sys, pyarrow.parquet as pq\n"
        "t = pq.read_table(sys.argv[1]).to_pylist()\n"
        "sys.stdout.write('\\n'.join(json.dumps(r) for r in t))\n"
    )
    out = subprocess.run(
        ["uv", "run", "--no-project", "--quiet", "--with", "pyarrow",
         "python", "-c", script, parquet],
        capture_output=True, text=True, check=True, timeout=600)
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def _split_tests(raw: str | list) -> tuple[str, ...]:
    """FAIL_TO_PASS / PASS_TO_PASS ship as a JSON-encoded list inside a string column."""
    if isinstance(raw, str):
        return tuple(json.loads(raw))
    return tuple(raw)


def fetch(root: str, force: bool = False) -> tuple[str, str]:
    """Download, verify, and write `_data/instances.jsonl` and `_data/tasks.jsonl`.

    Returns the two paths. Re-running is a no-op unless `force`; the digest is checked
    every time, so a corrupted or swapped parquet is caught on the run that uses it and
    not on the one that fetched it.
    """
    out = data_dir(root)
    parquet = os.path.join(out, "mini.parquet")
    if force or not os.path.exists(parquet):
        _download(parquet)
    got = digest(parquet)
    if got != PARQUET_SHA256:
        raise ValueError(
            f"{DATASET_ID} parquet digest mismatch: expected {PARQUET_SHA256}, got {got}. "
            "Refusing to build the task set from an unpinned benchmark.")

    rows = _parquet_to_rows(parquet)
    if len(rows) != N_INSTANCES:
        raise ValueError(f"expected {N_INSTANCES} instances, parquet has {len(rows)}")

    inst_path = os.path.join(out, "instances.jsonl")
    task_path = os.path.join(out, "tasks.jsonl")
    with open(inst_path, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: str(r["instance_id"])):
            fh.write(json.dumps({
                "instance_id": r["instance_id"], "repo": r["repo"],
                "base_commit": r["base_commit"], "version": str(r["version"]),
                "problem_statement": r["problem_statement"],
                "environment_setup_commit": r["environment_setup_commit"],
                "patch": r["patch"], "test_patch": r["test_patch"],
                "fail_to_pass": list(_split_tests(r["FAIL_TO_PASS"])),
                "pass_to_pass": list(_split_tests(r["PASS_TO_PASS"])),
            }) + "\n")
    with open(task_path, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: str(r["instance_id"])):
            fh.write(json.dumps({
                "instance_id": r["instance_id"], "repo": r["repo"],
                "base_commit": r["base_commit"], "version": str(r["version"]),
                "problem_statement": r["problem_statement"],
            }) + "\n")
    return inst_path, task_path


def load_instances(root: str) -> list[Instance]:
    """Every field, gold included. Callers must not be holding a model."""
    path = os.path.join(data_dir(root), "instances.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [Instance(
        instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
        version=r["version"], problem_statement=r["problem_statement"],
        environment_setup_commit=r["environment_setup_commit"],
        patch=r["patch"], test_patch=r["test_patch"],
        fail_to_pass=tuple(r["fail_to_pass"]), pass_to_pass=tuple(r["pass_to_pass"]),
    ) for r in rows]


def load_tasks(root: str) -> list[Task]:
    """The solver's view. This function is the only one `run.py` imports from here."""
    path = os.path.join(data_dir(root), "tasks.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    for r in rows:
        leaked = sorted(set(r) & set(_GOLD_FIELDS))
        if leaked:
            raise ValueError(f"tasks.jsonl carries gold fields {leaked} — regenerate it")
        missing = sorted(set(_TASK_FIELDS) - set(r))
        if missing:
            raise ValueError(f"tasks.jsonl row {r.get('instance_id')} missing {missing}")
    return [Task(instance_id=r["instance_id"], repo=r["repo"],
                 base_commit=r["base_commit"], version=r["version"],
                 problem_statement=r["problem_statement"]) for r in rows]


def tabulate(instances: list[Instance]) -> str:
    """The shape of the set, printed after a fetch so the pin is visible in the log."""
    by_env: dict[str, int] = {}
    for i in instances:
        by_env[i.env_key] = by_env.get(i.env_key, 0) + 1
    files = [i.patch.count("\ndiff --git ") + i.patch.startswith("diff --git ")
             for i in instances]
    spread: dict[int, int] = {}
    for n in files:
        spread[n] = spread.get(n, 0) + 1
    lines = [f"{DATASET_ID} @ {REVISION[:12]} — {len(instances)} instances",
             f"  environments: {len(by_env)}"]
    for key in sorted(by_env):
        lines.append(f"    {key:<28} {by_env[key]:>3}")
    lines.append("  gold patch files touched: "
                 + ", ".join(f"{n} file(s) x{spread[n]}" for n in sorted(spread)))
    return "\n".join(lines)
