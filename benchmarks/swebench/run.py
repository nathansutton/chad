"""Solve one block: one arm, one replicate, 50 instances, one model load.

    uv run --project <tree> python benchmarks/swebench/run.py --arm lean --rep 1

A block is the unit the campaign is scheduled in. It loads the weights once and keeps
them resident for all 50 trials, because on a 24 GB laptop the load is minutes and the
teardown-and-reload cycle is the one thing on this machine with GPU panics on its record.
Ten loads for a whole campaign, not five hundred.

The arms interleave by block — L F / F L / L F — so that heat, OS updates and the
calendar fall on both arms equally, and a campaign cut short is still balanced. Inside a
block the task order is fixed: position effects are identical for both arms of a
replicate, so the pairing removes them.

WHAT MAKES THE TWO ARMS COMPARABLE
----------------------------------
Everything outside the harness is owned by this file, not by either tree:

* the same weights, named on the command line and passed to both trees;
* the same sampler, applied through each tree's own shipped preset helper so it is
  the configuration a user actually gets rather than one invented for the benchmark;
* the same workspace, exported with no history (`workspace.py`);
* the same confinement, injected on `PATH` so neither tree needs to know about it
  (`sandbox.py`);
* the same prompt, the same wall cap, the same step cap;
* the same environment on `PATH`, and neither arm is told a test command.

WHAT IS RECORDED
----------------
`predictions.jsonl` is exactly the official three-field shape and nothing else, so it can
be handed to the grader untouched. Everything this measurement actually cares about —
prefill tokens, round trips, wall clock, which tools were called, which levers fired —
goes to `trials.jsonl` beside it. Nothing here reads the gold patch: file-level
localization is scored later by joining `changed_files` against the instance rows, which
keeps the solver's view provably free of the answer.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402
import envspec  # noqa: E402
import prepare  # noqa: E402
import sandbox  # noqa: E402
import workspace  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))

# The task prompt, frozen. `problem_statement` verbatim, then one paragraph that says
# what a solution is. It names no file, no test command and no tool: discovering how the
# project is tested is part of the task, and telling one arm would be telling it the
# answer to the question the benchmark is asking.
PROMPT_SUFFIX = (
    "\n\n---\n\n"
    "Fix this issue in the repository you are standing in. Do not edit tests — the "
    "change is judged by the project's own test suite, which will be restored before "
    "it runs. When you are done, leave the fix in the working tree; do not commit it."
)

DEFAULT_WALL_CAP_S = 3600
DEFAULT_MAX_STEPS = 60

ARMS = ("lean", "legacy-full", "legacy-tools")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")


def lever_env(arm: str, manifest: str) -> dict[str, str]:
    """The lever configuration for an arm.

    lean          2.x levers are ON by default and there is no enable list; the shipped
                  configuration is the empty environment.
    legacy-full   1.x levers are OFF by default. `CHAD_ENABLE=all` is not a
                  configuration — a manifest naming every lever ON or OFF with a reason
                  is, and it is frozen by sha before the first scored trial.
    legacy-tools  the 1.x shipped default: the 22-tool surface with every lever off.
    """
    if arm != "legacy-full":
        return {}
    if not manifest or not os.path.exists(manifest):
        raise SystemExit(
            f"--arm legacy-full needs a lever manifest (got {manifest!r}). It is the "
            "arm's pre-registered configuration: one row per lever, ON or OFF, with a "
            "reason. Build it with the legacy tree.")
    on = []
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise SystemExit(f"malformed manifest row: {line!r}")
            if parts[1].strip().upper() == "ON":
                on.append(parts[0].strip())
    if not on:
        raise SystemExit(f"{manifest} switches nothing on")
    return {"CHAD_ENABLE": ",".join(sorted(on))}


def preflight(allow_docker: bool = False) -> list[str]:
    """Refuse to start a block on a machine that would make the numbers meaningless."""
    problems = []
    if sys.platform != "darwin":
        problems.append("this benchmark is about an Apple Silicon laptop; "
                        f"this is {sys.platform}")
    if not sandbox.available():
        problems.append(f"{sandbox.SANDBOX_EXEC} is missing — trials would run unconfined")
    if not allow_docker:
        docker = shutil.which("docker")
        if docker:
            up = subprocess.run([docker, "info"], capture_output=True, check=False,
                                timeout=60)
            if up.returncode == 0:
                problems.append("Docker is running — quit it before solving; the model "
                                "and a grading VM must never share the 24 GB")
    power = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True,
                           check=False)
    if "Battery Power" in power.stdout:
        problems.append("running on battery — mains power only, the clock is a measurement")
    return problems


def default_model() -> str:
    """The shipped weights, resolved once from THIS tree and passed to both arms.

    Resolved in a subprocess rather than by importing chad here: the arm's own tree is
    what this process is about to import from, and a stray import of the release tree
    first would decide which `chad` package wins for the rest of the run. Resolving it
    from one place is also the point — the comparison is only about the harness, so
    neither arm may pick its own weights.
    """
    out = subprocess.run(
        ["uv", "run", "--project", REPO_ROOT, "--quiet", "python", "-c",
         "from chad.cli import _pick_model; print(_pick_model()[0])"],
        capture_output=True, text=True, check=False, timeout=600)
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 else ""


def thermal() -> str:
    """CPU speed limit as the OS reports it. Sampled per trial so a throttled stretch is
    visible in the rows rather than inferred from a suspicious wall time."""
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                         check=False)
    for line in out.stdout.splitlines():
        if "CPU_Speed_Limit" in line:
            return line.strip()
    return ""


@contextlib.contextmanager
def _stdin_from_devnull():
    """Every trial runs with stdin on /dev/null, at the file-descriptor level.

    Launched from a terminal, `sys.stdin.isatty()` is true and a confirm prompt goes to
    `input()` behind the redirected stdout — an invisible question that no timeout
    interrupts. fd 0 as well as `sys.stdin`, because the spawned shell inherits it.
    """
    saved_fd, saved_stdin = os.dup(0), sys.stdin
    devnull = open(os.devnull)
    try:
        os.dup2(devnull.fileno(), 0)
        sys.stdin = devnull
        yield
    finally:
        sys.stdin = saved_stdin
        os.dup2(saved_fd, 0)
        os.close(saved_fd)
        devnull.close()


def _tool_calls(messages: list[dict]) -> dict[str, int]:
    """How often each tool actually returned a result this trial — the reachability
    question: is bash still the whole surface when twenty-two tools are on offer?"""
    counts: dict[str, int] = {}
    for m in messages:
        if m.get("role") == "tool":
            name = str(m.get("name") or "")
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


class Block:
    """One arm, one replicate: the model stays loaded across every trial in it."""

    def __init__(self, arm: str, rep: int, tree: str, model: str, out_dir: str,
                 wall_cap: int, max_steps: int, manifest: str) -> None:
        self.arm = arm
        self.rep = rep
        self.tree = tree
        self.model = model
        self.out_dir = out_dir
        self.wall_cap = wall_cap
        self.max_steps = max_steps
        self.lever_env = lever_env(arm, manifest)
        self.shim_dir = os.path.join(out_dir, "bin")
        self.base_path = os.environ.get("PATH", "")

    # -- environment ------------------------------------------------------

    def install_shim(self) -> None:
        """Put the confining shell first on PATH.

        Must happen before chad is imported: both trees resolve their shell once, at
        import, with `which("bash")`. An import that has already happened would keep the
        real /bin/bash and every trial would run unconfined.
        """
        sandbox.write_shim(self.shim_dir)
        os.environ["PATH"] = self.shim_dir + os.pathsep + self.base_path
        if os.environ.get("CHAD_NO_SEATBELT") != "1":
            # Seatbelt profiles do not nest. chad's own confinement would fail to apply
            # inside the runner's and turn every command into an error.
            os.environ["CHAD_NO_SEATBELT"] = "1"

    def trial_env(self, task: dataset.Task, ws: str, profile: str) -> None:
        """Point the process environment at this trial: its sandbox profile, its
        workbench environment, and a git that cannot read the machine's config."""
        env_bin = os.path.join(prepare.envs_dir(ROOT), task.env_key, "venv", "bin")
        os.environ["PATH"] = os.pathsep.join([self.shim_dir, env_bin, self.base_path])
        os.environ[sandbox.PROFILE_ENV] = profile
        os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TMPDIR"] = os.path.join(ws, ".swb-tmp")
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
        # PYTHONPATH stands in for the container's editable install (envspec deviation 4):
        # without it `import django` fails from the test runner and the agent would be
        # solving a packaging puzzle the benchmark does not contain.
        os.environ.update(envspec.workspace_env(ws))
        os.environ.update(self.lever_env)

    # -- the model --------------------------------------------------------

    def load_engine(self, thinking: bool):
        """Build and load the engine the way the shipped CLI does, from the tree under
        test. Both arms go through each tree's own preset helper, so the sampler is the
        one a user gets and not one invented here."""
        sys.path.insert(0, os.path.join(self.tree, "src"))
        from chad.cli import _compute_ctx_limit, apply_sampler_env, apply_sampler_preset
        from chad.engine import Engine
        eng = Engine(model_id=self.model)
        apply_sampler_preset(eng, thinking=thinking)
        apply_sampler_env(eng)
        load_s = eng.load()
        ctx_limit = _compute_ctx_limit(eng)
        return eng, ctx_limit, load_s

    # -- one trial --------------------------------------------------------

    def solve(self, eng, ctx_limit, task: dataset.Task, thinking: bool) -> dict:
        from chad import levers
        from chad.agent import Agent

        ws = os.path.join(ROOT, "_work", f"{self.arm}-rep{self.rep}", task.instance_id)
        workspace.materialize(ROOT, task.repo, task.base_commit, ws, task.instance_id)
        profile = sandbox.write_profile(
            os.path.join(ROOT, "_work", f"{self.arm}-rep{self.rep}",
                         f"{task.instance_id}.sb"),
            os.path.realpath(ws), prepare.trial_read_roots(ROOT, task.env_key))
        self.trial_env(task, ws, profile)

        deadline = [0.0]
        first_token = [0.0]

        def should_stop() -> bool:
            return deadline[0] > 0 and time.time() > deadline[0]

        def emit(kind: str, text: str) -> None:
            if not first_token[0] and kind == "stream" and text:
                first_token[0] = time.time()

        start_dir = os.getcwd()
        fires_before = dict(levers.fire_counts())
        try:
            os.chdir(ws)
            agent = Agent(eng, max_steps=self.max_steps, thinking=thinking, mode="yolo",
                          ctx_limit=ctx_limit, should_stop=should_stop, emit=emit)
            t0 = time.time()
            deadline[0] = t0 + self.wall_cap
            with contextlib.redirect_stdout(io.StringIO()), _stdin_from_devnull():
                agent.run_turn(task.problem_statement + PROMPT_SUFFIX, stream=False)
            wall = time.time() - t0
        finally:
            os.chdir(start_dir)

        # A capped trial submits whatever it had; that is the honest reading of a wall
        # budget, and the budget is ours and disclosed.
        patch = workspace.diff(ws)
        row = {
            "instance_id": task.instance_id, "arm": self.arm, "rep": self.rep,
            "env": task.env_key, "wall_s": round(wall, 1),
            "capped": wall >= self.wall_cap,
            "first_token_s": round(first_token[0] - t0, 2) if first_token[0] else None,
            "steps": sum(1 for m in agent.messages if m.get("role") == "assistant"),
            "tool_calls": _tool_calls(agent.messages),
            "tool_dispatches": agent.tool_dispatches,
            "prefill_tokens": agent.prefill_tokens, "peak_ctx": agent.peak_ctx,
            "gen_tokens": agent.gen_tokens, "gen_time": round(agent.gen_time, 1),
            "think_tokens": agent.think_tokens, "forwards": agent.forwards,
            "draft_proposed": agent.draft_proposed,
            "draft_accepted": agent.draft_accepted,
            "patch_bytes": len(patch), "patch_empty": not patch.strip(),
            "changed_files": list(workspace.changed_files(patch)),
            "levers_active": levers.active(),
            "levers_fired": {k: v - fires_before.get(k, 0)
                             for k, v in levers.fire_counts().items()
                             if v - fires_before.get(k, 0) > 0},
            "thermal": thermal(),
        }
        return {"row": row, "patch": patch, "workspace": ws}

    # -- the block --------------------------------------------------------

    def run(self, tasks: list[dataset.Task], thinking: bool, keep: bool) -> int:
        os.makedirs(self.out_dir, exist_ok=True)
        eng, ctx_limit, load_s = self.load_engine(thinking)
        meta = {
            "arm": self.arm, "rep": self.rep, "tree": self.tree,
            "tree_sha": subprocess.run(["git", "-C", self.tree, "rev-parse", "HEAD"],
                                       capture_output=True, text=True,
                                       check=False).stdout.strip(),
            "model": self.model, "thinking": thinking,
            "sampler": {"temp": eng.temp, "top_p": eng.top_p, "top_k": eng.top_k,
                        "min_p": eng.min_p},
            "ctx_limit": ctx_limit, "load_s": round(load_s, 1),
            "wall_cap_s": self.wall_cap, "max_steps": self.max_steps,
            "lever_env": self.lever_env, "prompt_suffix": PROMPT_SUFFIX,
            "dataset": dataset.DATASET_ID, "dataset_revision": dataset.REVISION,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_tasks": len(tasks),
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.write("\n")

        name = f"chad-{self.arm}"
        preds = open(os.path.join(self.out_dir, "predictions.jsonl"), "a",
                     encoding="utf-8")
        trials = open(os.path.join(self.out_dir, "trials.jsonl"), "a", encoding="utf-8")
        try:
            for n, task in enumerate(tasks, 1):
                print(f"[{n}/{len(tasks)}] {self.arm} rep{self.rep} "
                      f"{task.instance_id} ...", flush=True)
                out = self.solve(eng, ctx_limit, task, thinking)
                preds.write(json.dumps({"instance_id": task.instance_id,
                                        "model_name_or_path": name,
                                        "model_patch": out["patch"]}) + "\n")
                preds.flush()
                trials.write(json.dumps(out["row"], sort_keys=True) + "\n")
                trials.flush()
                r = out["row"]
                print(f"      {r['wall_s']:>6.0f}s  {r['steps']:>3} steps  "
                      f"{r['prefill_tokens']:>8,} prefill  "
                      f"{'CAPPED ' if r['capped'] else ''}"
                      f"{'no patch' if r['patch_empty'] else str(len(r['changed_files'])) + ' file(s)'}",
                      flush=True)
                if not keep:
                    shutil.rmtree(out["workspace"], ignore_errors=True)
        finally:
            preds.close()
            trials.close()
        print(f"\nblock done: {self.out_dir}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--rep", type=int, required=True, help="replicate number (1..k)")
    ap.add_argument("--tree", default=REPO_ROOT,
                    help="the harness checkout to import chad from")
    ap.add_argument("--model", default="",
                    help="weights for BOTH arms; defaults to the tree's shipped model")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "legacy-manifest.tsv"))
    ap.add_argument("--out", default="", help="run directory (default _runs/<arm>-rep<N>-<date>)")
    ap.add_argument("--wall-cap", type=int, default=DEFAULT_WALL_CAP_S)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--only", default="", help="comma-separated instance ids (a pilot)")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (a pilot)")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--keep-workspaces", action="store_true",
                    help="keep each trial's checkout for a trajectory read")
    ap.add_argument("--allow-docker", action="store_true",
                    help="skip the Docker check (only for a dry run with no model)")
    args = ap.parse_args()

    problems = preflight(allow_docker=args.allow_docker)
    if problems:
        for p in problems:
            print(f"preflight: {p}", file=sys.stderr)
        return 2

    tasks = dataset.load_tasks(ROOT)
    if args.only:
        want = set(args.only.split(","))
        tasks = [t for t in tasks if t.instance_id in want]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    model = args.model or default_model()
    if not model:
        print("could not resolve the shipped model; pass --model", file=sys.stderr)
        return 2

    out_dir = args.out or os.path.join(ROOT, "_runs",
                                       f"{args.arm}-rep{args.rep}-{_now()}")
    block = Block(arm=args.arm, rep=args.rep, tree=os.path.abspath(args.tree),
                  model=model, out_dir=out_dir, wall_cap=args.wall_cap,
                  max_steps=args.max_steps, manifest=args.manifest)
    block.install_shim()

    # Hold the machine awake for the block without touching the display setting; the
    # child dies with the runner, so an interrupted block does not leave the laptop
    # pinned awake.
    with subprocess.Popen(["caffeinate", "-is"]) as awake:
        try:
            return block.run(tasks, thinking=not args.no_think,
                             keep=args.keep_workspaces)
        finally:
            awake.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
