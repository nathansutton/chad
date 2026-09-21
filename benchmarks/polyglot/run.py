"""Run a block of trials: one model load, many exercises, one row per trial.

    uv run python benchmarks/polyglot/run.py --label baseline            # all valid tasks
    uv run python benchmarks/polyglot/run.py --label baseline --reps 3   # k=3, rep-major
    uv run python benchmarks/polyglot/run.py --label arm-b --tasks-file results/pool.txt
    CHAD_DISABLE=env_manifest uv run python benchmarks/polyglot/run.py --label no-manifest

The agent under test is the one a user gets: the shipped model, the shipped sampler
preset, the RAM-aware context limit, yolo mode, the default step budget. An arm is
whatever differs in the environment or the checkout when the block is launched; the label
names it and `meta.json` records what it actually was.

The weights load once per block. On a 24 GB laptop a load is minutes and load/teardown
cycling is the one thing here with kernel panics on its record, so a block is resumable
instead: rows already in `trials.jsonl` are skipped, and re-running the same command
picks up where a crash, a lid close or a Ctrl-C left it.

Every trial also leaves an ATIF trajectory under `_runs/<label>/trajectories/`, rewritten
as the trial runs — including the step still being generated — so a trial stuck in a long
think can be read while it is stuck: `python benchmarks/polyglot/trace.py <file>`.

With `--reps N` the order is rep-major — every task once, then every task again — so a
block cut short still has balanced coverage instead of k=3 on the first third of the
alphabet.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import io
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog  # noqa: E402
import workspace  # noqa: E402
from catalog import JsonValue, Task  # noqa: E402

if TYPE_CHECKING:
    from chad.engine import Engine

REPO_ROOT = os.path.dirname(os.path.dirname(catalog.ROOT))
DEFAULT_WALL_CAP_S = 1200
KV_CACHE_MAX_BYTES = 4 * 1024**3

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
    in its rows, not inferred afterwards from a suspicious wall time."""
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=False)
    return next((ln.strip() for ln in out.stdout.splitlines() if "CPU_Speed_Limit" in ln), "")


@contextlib.contextmanager
def _stdin_from_devnull() -> Iterator[None]:
    """Launched from a terminal, a confirm prompt would go to `input()` behind the
    redirected stdout — an invisible question no timeout interrupts. Closed at the file
    descriptor as well as `sys.stdin`, because spawned shells inherit fd 0."""
    saved_fd, saved_stdin = os.dup(0), sys.stdin
    devnull = open(os.devnull, encoding="utf-8")
    try:
        os.dup2(devnull.fileno(), 0)
        sys.stdin = devnull
        yield
    finally:
        sys.stdin = saved_stdin
        os.dup2(saved_fd, 0)
        os.close(saved_fd)
        devnull.close()


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


class Block:
    def __init__(self, label: str, model: str, thinking: bool, max_steps: int | None,
                 wall_cap: int):
        self.label, self.model, self.thinking = label, model, thinking
        self.max_steps, self.wall_cap = max_steps, wall_cap
        self.out_dir = os.path.join(catalog.ROOT, "_runs", label)
        self.trials_path = os.path.join(self.out_dir, "trials.jsonl")
        os.makedirs(self.out_dir, exist_ok=True)

    def load_engine(self) -> tuple[Engine, int, float]:
        """The engine as the shipped CLI builds it: preset first, CHAD_* env second."""
        from chad.cli import _compute_ctx_limit, apply_sampler_env, apply_sampler_preset
        from chad.engine import Engine
        eng = Engine(model_id=self.model, cache_dir=os.path.join(catalog.ROOT, "_kv"),
                     kv_cache_max_bytes=KV_CACHE_MAX_BYTES)
        apply_sampler_preset(eng, thinking=self.thinking)
        apply_sampler_env(eng)
        load_s = eng.load()
        return eng, _compute_ctx_limit(eng), load_s

    def write_meta(self, ctx_limit: int, load_s: float, n_tasks: int, reps: int) -> None:
        import chad
        from chad import levers
        from chad.tools import active_schemas
        meta: dict[str, JsonValue] = {
            "label": self.label, "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "chad_version": chad.__version__, "git_rev": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--", "src")),
            "model": self.model, "thinking": self.thinking, "ctx_limit": ctx_limit,
            "max_steps": self.max_steps, "wall_cap_s": self.wall_cap,
            "load_s": round(load_s, 1), "tasks": n_tasks, "reps": reps,
            "upstream_commit": catalog.UPSTREAM_COMMIT,
            "levers_active": levers.active(),
            "chad_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("CHAD_")},
            "tools": sorted(s["function"]["name"] for s in active_schemas()),
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")

    def trajectory_path(self, task: Task, rep: int) -> str:
        return os.path.join(self.out_dir, "trajectories", task.language,
                            f"{task.slug}.rep{rep}.json")

    def solve(self, eng: Engine, ctx_limit: int, task: Task, rep: int) -> dict[str, JsonValue]:
        from chad import atif, levers
        from chad.agent import Agent

        ws = os.path.join(catalog.ROOT, "_work", self.label, f"rep{rep}", task.language,
                          task.slug)
        workspace.materialize(task, ws)
        deadline = time.time() + self.wall_cap
        fires_before = levers.fire_counts()
        start_dir, saved_path = os.getcwd(), os.environ.get("PATH", "")
        try:
            os.chdir(ws)
            os.environ["PATH"] = workspace.trial_env()["PATH"]
            traj = self.trajectory_path(task, rep)
            # Before the Agent: it binds the recorder it is built with.
            recorder = atif.start(traj)
            build = functools.partial(Agent, eng, thinking=self.thinking, mode="yolo",
                                      ctx_limit=ctx_limit,
                                      should_stop=lambda: time.time() > deadline,
                                      emit=lambda _kind, _text: None)
            # No --max-steps keeps the agent's own default rather than restating it here.
            agent = build() if self.max_steps is None else build(max_steps=self.max_steps)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()), _stdin_from_devnull():
                agent.run_turn(prompt_for(task), stream=False)
            wall = time.time() - t0
        finally:
            os.chdir(start_dir)
            os.environ["PATH"] = saved_path
        verdict = workspace.verify(task, ws)
        fired = {k: v - fires_before.get(k, 0) for k, v in levers.fire_counts().items()
                 if v > fires_before.get(k, 0)}
        return {
            "task": task.name, "language": task.language, "label": self.label, "rep": rep,
            # The deadline stops the agent a moment BEFORE the cap, so the wall time alone
            # reads a capped trial as one that finished at 1199.8 s.
            "passed": verdict.passed, "capped": agent.interrupted,
            "wall_s": round(wall, 1), "verify_s": round(verdict.seconds, 1),
            "steps": sum(1 for m in agent.messages if m.get("role") == "assistant"),
            "tool_dispatches": agent.tool_dispatches,
            "gen_tokens": agent.gen_tokens, "think_tokens": agent.think_tokens,
            "prefill_tokens": agent.prefill_tokens, "peak_ctx": agent.peak_ctx,
            "gen_time_s": round(agent.gen_time, 1),
            "levers_fired": fired, "thermal": thermal(),
            # The trajectory's own trial totals — cache hit rate, tok/s, draft acceptance
            # by phase, compactions — so a score table never has to open the documents.
            "metrics": recorder.to_dict().get("final_metrics", {}),
            "trajectory": os.path.relpath(traj, self.out_dir),
            "test_tail": "" if verdict.passed else verdict.tail[-600:],
        }

    def run(self, tasks: list[Task], reps: int) -> int:
        todo = [(rep, t) for rep in range(1, reps + 1) for t in tasks]
        done = done_keys(self.trials_path)
        todo = [(rep, t) for rep, t in todo if (t.name, rep) not in done]
        print(f"[{self.label}] {len(todo)} trials to run, {len(done)} already recorded")
        if not todo:
            return 0
        eng, ctx_limit, load_s = self.load_engine()
        self.write_meta(ctx_limit, load_s, len(tasks), reps)
        passed = 0
        for i, (rep, task) in enumerate(todo, 1):
            row = self.solve(eng, ctx_limit, task, rep)
            with open(self.trials_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            passed += bool(row["passed"])
            print(f"[{i}/{len(todo)}] {'PASS' if row['passed'] else 'fail'} {task.name} "
                  f"rep{rep} {row['wall_s']}s {row['steps']} steps | running {passed}/{i}",
                  flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True, help="names the arm; rows go to _runs/<label>/")
    ap.add_argument("--model", default="", help="HF repo or local dir (default: the shipped model)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--only", default="", help="task-name prefix, e.g. `rust/`")
    ap.add_argument("--tasks-file", default="", help="one task name per line (a pinned pool)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="default: the agent's own")
    ap.add_argument("--wall-cap", type=int, default=DEFAULT_WALL_CAP_S, help="seconds per trial")
    args = ap.parse_args()

    tasks = select_tasks(args.only, args.tasks_file, args.limit)
    if not tasks:
        raise SystemExit("no tasks selected")
    model = args.model
    if not model:
        from chad.cli import _pick_model
        model = _pick_model()[0]
    block = Block(args.label, model, not args.no_think, args.max_steps, args.wall_cap)
    return block.run(tasks, args.reps)


if __name__ == "__main__":
    raise SystemExit(main())
