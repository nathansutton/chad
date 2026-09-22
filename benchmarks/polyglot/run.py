"""Run a block of trials: one arm, many exercises, one row per trial.

    uv run python benchmarks/polyglot/run.py --label baseline            # all valid tasks
    uv run python benchmarks/polyglot/run.py --label baseline --reps 3   # k=3, rep-major
    uv run python benchmarks/polyglot/run.py --label arm-b \
        --tasks-file benchmarks/polyglot/_runs/baseline/pool.txt
    CHAD_DISABLE=env_manifest uv run python benchmarks/polyglot/run.py --label no-manifest
    uv run python benchmarks/polyglot/run.py --label pi-pool --harness pi \
        --tasks-file benchmarks/polyglot/_runs/baseline/pool.txt --server http://127.0.0.1:8080

The default arm is the agent a user gets: chad in this process, with the shipped model,
the shipped sampler preset, the RAM-aware context limit, yolo mode, the default step
budget. An arm is whatever differs in the environment or the checkout when the block is
launched; the label names it and `meta.json` records what it actually was.

`--harness <name>` runs another entry of `harnesses.py` instead: a coding agent's own
command line, in a throwaway home under Seatbelt (`harness/cli.py`), against the
llama-server already listening at `--server`. The tasks, the prompt, the wall cap, the
verification and the row are the same loop whoever is solving (`harness/__init__.py`).

The weights load once per block. On a 24 GB laptop a load is minutes and load/teardown
cycling is the one thing here with kernel panics on its record, so a block is resumable
instead: rows already in `trials.jsonl` are skipped, and re-running the same command
picks up where a crash, a lid close or a Ctrl-C left it.

Every chad trial also leaves an ATIF trajectory under `_runs/<label>/trajectories/`,
rewritten as the trial runs — including the step still being generated — so a trial
stuck in a long think can be read while it is stuck: `python benchmarks/polyglot/trace.py
<file>`. A CLI arm's own output goes to `_runs/<label>/output/`.

With `--reps N` the order is rep-major — every task once, then every task again — so a
block cut short still has balanced coverage instead of k=3 on the first third of the
alphabet.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog  # noqa: E402
import harnesses  # noqa: E402
import workspace  # noqa: E402
from catalog import JsonValue, Task, is_object  # noqa: E402
from harness import AtifDoc, Harness, Solved, Trial  # noqa: E402
from harness.chad_inprocess import ChadInProcess  # noqa: E402
from harness.cli import CliHarness, Endpoint, HarnessError, server_context  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(catalog.ROOT))
DEFAULT_WALL_CAP_S = 1200
DEFAULT_SERVER = "http://127.0.0.1:8080"

_DOCS = ("introduction.md", "instructions.md", "instructions.append.md")


def prompt_for(task: Task) -> str:
    """The exercise's own statement — introduction, instructions, track-specific appendix,
    the same three files Aider sends and never the hints file — then the contract."""
    parts = []
    for name in _DOCS:
        path = os.path.join(task.source, ".docs", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                parts.append(f.read().strip())
    files = ", ".join(f"`{s}`" for s in task.solution)
    return ("\n\n".join(parts) + "\n\n---\n\n"
            f"Implement the solution in {files}, in the directory you are standing in. "
            f"`sh {workspace.SCRIPT}` runs the full test suite; iterate until it passes. "
            "Do not edit the tests or that script — both are restored before grading.")


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=False)
    return out.stdout.strip()


def thermal() -> str:
    """macOS's CPU speed limit at the end of a trial: a throttled run should be visible
    in its rows, not inferred afterwards from a suspicious wall time. Empty off macOS."""
    if not shutil.which("pmset"):
        return ""
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=False)
    return next((ln.strip() for ln in out.stdout.splitlines() if "CPU_Speed_Limit" in ln), "")


def done_keys(trials_path: str) -> set[tuple[str, int]]:
    """(task, rep) pairs already recorded — what a resumed block skips."""
    keys: set[tuple[str, int]] = set()
    if not os.path.exists(trials_path):
        return keys
    with open(trials_path, encoding="utf-8") as f:
        for line in f:
            row: JsonValue = json.loads(line)
            if catalog.is_object(row):
                task, rep = row.get("task"), row.get("rep")
                if catalog.is_text(task) and catalog.is_number(rep):
                    keys.add((task, int(rep)))
    return keys


def select_tasks(only: str, tasks_file: str, limit: int) -> list[Task]:
    valid = set(catalog.load_manifest())
    if tasks_file:
        with open(tasks_file, encoding="utf-8") as f:
            wanted = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        unknown = wanted - valid
        if unknown:
            raise SystemExit(f"--tasks-file names tasks outside the manifest: {sorted(unknown)}")
        valid = wanted
    tasks = [t for t in catalog.load_tasks() if t.name in valid and t.name.startswith(only)]
    return tasks[:limit] if limit else tasks


def row(task: Task, label: str, rep: int, solved: Solved, verdict: workspace.Verdict,
        doc: AtifDoc | None, trajectory: str, machine: str) -> dict[str, JsonValue]:
    """One trial's record: the same frame for every arm, with the arm's own fields in the
    middle. `trajectory` is the document's path relative to the run."""
    final = doc.get("final_metrics") if doc is not None else None
    return {
        "task": task.name, "language": task.language, "label": label, "rep": rep,
        "passed": verdict.passed, "capped": solved.capped,
        "wall_s": round(solved.wall_s, 1), "verify_s": round(verdict.seconds, 1),
        **solved.fields,
        "thermal": machine,
        # The trajectory's own trial totals — cache hit rate, tok/s, draft acceptance
        # by phase, compactions — so a score table never has to open the documents.
        "metrics": final if is_object(final) else {},
        "trajectory": trajectory if doc is not None else None,
        "test_tail": "" if verdict.passed else verdict.tail[-600:],
    }


class Block:
    def __init__(self, label: str, harness: Harness, wall_cap: int,
                 runs: str = os.path.join(catalog.ROOT, "_runs"), tasks_file: str = ""):
        self.label, self.harness, self.wall_cap = label, harness, wall_cap
        self.tasks_file = tasks_file
        self.out_dir = os.path.join(runs, label)
        self.trials_path = os.path.join(self.out_dir, "trials.jsonl")
        os.makedirs(self.out_dir, exist_ok=True)

    def write_meta(self, arm: Mapping[str, JsonValue], n_tasks: int, reps: int) -> None:
        import chad
        meta: dict[str, JsonValue] = {
            "label": self.label, "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "harness": self.harness.name, "harness_version": self.harness.version(),
            "chad_version": chad.__version__, "git_rev": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--", "src")),
            "wall_cap_s": self.wall_cap, "tasks": n_tasks, "reps": reps,
            "upstream_commit": catalog.UPSTREAM_COMMIT,
            **arm,
        }
        if self.tasks_file:
            # A subset or pool is only pre-registered if the rows name exactly which one.
            with open(self.tasks_file, "rb") as f:
                meta["tasks_sha256"] = hashlib.sha256(f.read()).hexdigest()
            meta["tasks_file"] = os.path.relpath(os.path.abspath(self.tasks_file), REPO_ROOT)
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")

    def trial(self, task: Task, rep: int) -> Trial:
        where = (self.label, f"rep{rep}", task.language, task.slug)
        name = f"{task.slug}.rep{rep}"
        return Trial(workspace=os.path.join(self.harness.root, "_work", *where),
                     home=os.path.join(self.harness.root, "_home", *where),
                     prompt=prompt_for(task), wall_cap_s=self.wall_cap,
                     trajectory=os.path.join(self.out_dir, "trajectories", task.language,
                                             f"{name}.json"),
                     output=os.path.join(self.out_dir, "output", task.language, f"{name}.log"))

    def solve(self, task: Task, rep: int) -> dict[str, JsonValue]:
        trial = self.trial(task, rep)
        workspace.materialize(task, trial.workspace)
        self.harness.prepare(trial.home)
        solved = self.harness.solve(trial)
        verdict = workspace.verify(task, trial.workspace)
        return row(task, self.label, rep, solved, verdict, self.harness.trajectory(),
                   os.path.relpath(trial.trajectory, self.out_dir), thermal())

    def run(self, tasks: list[Task], reps: int) -> int:
        todo = [(rep, t) for rep in range(1, reps + 1) for t in tasks]
        done = done_keys(self.trials_path)
        todo = [(rep, t) for rep, t in todo if (t.name, rep) not in done]
        print(f"[{self.label}] {len(todo)} trials to run, {len(done)} already recorded")
        if not todo:
            return 0
        self.write_meta(self.harness.start(), len(tasks), reps)
        passed = 0
        for i, (rep, task) in enumerate(todo, 1):
            record = self.solve(task, rep)
            with open(self.trials_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            passed += bool(record["passed"])
            steps = f" {record['steps']} steps" if "steps" in record else ""
            print(f"[{i}/{len(todo)}] {'PASS' if record['passed'] else 'fail'} {task.name} "
                  f"rep{rep} {record['wall_s']}s{steps} | running {passed}/{i}", flush=True)
        return 0


def tokenizer_dir() -> str:
    """The shipped model's directory, for an arm that renders token ids itself (chad on
    llama-server). Resolved here, where HOME is the real one: inside a trial a repo id
    would resolve against the throwaway home's empty cache. Empty when it is nowhere
    local, which only an arm that needs it refuses."""
    from chad.cli import _pick_model
    model = _pick_model()[0]
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model, local_files_only=True)
    except (OSError, ValueError):      # not in the cache, or not a repo id at all
        return ""


def build_harness(args: argparse.Namespace) -> Harness:
    if args.harness == "chad":
        model = args.model
        if not model:
            from chad.cli import _pick_model
            model = _pick_model()[0]
        return ChadInProcess(model, not args.no_think, args.max_steps)
    spec = harnesses.SPECS[args.harness]
    if spec.unsupported:               # before asking for a server it would never use
        raise HarnessError(f"{spec.name} is marked unsupported: {spec.unsupported}")
    server = args.server.rstrip("/")
    endpoint = Endpoint(server, args.served_model, server_context(server), tokenizer_dir())
    return CliHarness(spec, endpoint, harnesses.load_lock().get(spec.name))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="names the arm; rows go to _runs/<label>/")
    ap.add_argument("--harness", default="chad", choices=["chad", *harnesses.SPECS],
                    help="who solves: chad in this process (default), or an entry of harnesses.py")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", default="", help="task-name prefix, e.g. `rust/`")
    ap.add_argument("--tasks-file", default="", help="one task name per line (a pinned pool)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wall-cap", type=int, default=DEFAULT_WALL_CAP_S, help="seconds per trial")
    chad = ap.add_argument_group("chad in process")
    chad.add_argument("--model", default="", help="HF repo or local dir (default: the shipped model)")
    chad.add_argument("--no-think", action="store_true")
    chad.add_argument("--max-steps", type=int, default=None, help="default: the agent's own")
    cli = ap.add_argument_group("any other harness")
    cli.add_argument("--server", default=DEFAULT_SERVER,
                     help="origin of the llama-server every trial of the block talks to")
    cli.add_argument("--served-model", default=harnesses.SERVED_MODEL,
                     help="the model id that server answers to (its --alias)")
    args = ap.parse_args()
    if args.harness != "chad" and (args.model or args.no_think or args.max_steps is not None):
        ap.error("--model, --no-think and --max-steps apply to --harness chad only")

    tasks = select_tasks(args.only, args.tasks_file, args.limit)
    if not tasks:
        raise SystemExit("no tasks selected")
    try:
        block = Block(args.label, build_harness(args), args.wall_cap, tasks_file=args.tasks_file)
        return block.run(tasks, args.reps)
    except HarnessError as e:
        raise SystemExit(f"[{args.label}] {e}") from None


if __name__ == "__main__":
    raise SystemExit(main())
