"""Grading, in the official container, with no model on the machine.

    uv run python benchmarks/swebench/grade.py gate          # the 50 gold patches
    uv run python benchmarks/swebench/grade.py run _runs/lean-rep1-20260919

SWE-bench separates solving from grading, and that separation is what lets a 24 GB
laptop run this benchmark at all. The agent solves on the host, against a real checkout,
with the model resident. The patch it produces is then graded later by the official
harness inside Docker, with no model loaded — so the weights and a grading VM never
share the memory. The rule is enforced here rather than remembered: `gate` and `run`
both refuse to start while a solving block is alive.

The grade is always the official one. Nothing in this kit decides whether an instance is
resolved; the container does, from the project's own tests, exactly as it would for any
other submission. The native workbench the agent works in is a convenience for the
agent, never a scoring authority.

THE GOLD GATE
-------------
Before any agent trial is scored, the 50 gold patches must resolve 50/50 on this
machine. Anything less means the grader here is not the public grader — an image that
built differently, a test that depends on the host, an architecture difference — and a
number produced by it would not be comparable to anything. It is the first stop
condition in the plan for a reason: it is cheap, and everything downstream assumes it.

ARCHITECTURE
------------
Images are built locally for arm64 (`--namespace ''`), because the published images are
x86_64 and emulating 500 test runs under Rosetta is slow enough to change what a timeout
means. If the arm64 build cannot pass the gold gate, the fallback is the published
x86_64 images under emulation — slower, but the same harness and the same tests, and the
route actually used is recorded in the report.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

# The grading harness, pinned. 4.x is the last line that carries the environment specs
# in-process, which is what makes the native workbench in `envspec.py` auditable against
# the same release that does the grading.
HARNESS = "swebench==4.0.5"

ROUTES = {
    # namespace None -> build images locally; the empty string is how the CLI spells it.
    "arm64": ["--namespace", ""],
    # the published images, under emulation
    "x86_64": ["--namespace", "swebench"],
}


def _model_is_resident() -> str:
    """A crude, deliberate check: is a solving block running right now?

    Docker Desktop's VM and 12 GB of weights do not fit together on this machine, and
    the failure mode is not a clean error — it is swap, then a frozen laptop. So this
    refuses rather than warns.
    """
    out = subprocess.run(["pgrep", "-fl", "swebench/run.py"], capture_output=True,
                         text=True, check=False)
    live = [ln for ln in out.stdout.splitlines() if "pgrep" not in ln]
    return live[0] if live else ""


def _docker_up() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    return subprocess.run([docker, "info"], capture_output=True, check=False,
                          timeout=120).returncode == 0


def preflight() -> list[str]:
    problems = []
    live = _model_is_resident()
    if live:
        problems.append(f"a solving block is running ({live.strip()}) — grade between "
                        "blocks, never beside one")
    if not _docker_up():
        problems.append("Docker is not running — start it for grading, and quit it "
                        "again before the next block")
    return problems


def _instance_ids() -> list[str]:
    return [i.instance_id for i in dataset.load_instances(ROOT)]


def evaluate(predictions: str, run_id: str, route: str, report_dir: str,
             workers: int = 4, timeout: int = 1800) -> int:
    """Drive the official harness. Its dependencies are rented for the call, so nothing
    the grader needs is ever installed beside the model."""
    os.makedirs(report_dir, exist_ok=True)
    argv = ["uv", "run", "--no-project", "--quiet", "--with", HARNESS,
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset.GRADING_DATASET,
            "--predictions_path", predictions,
            "--run_id", run_id,
            "--max_workers", str(workers),
            "--timeout", str(timeout),
            "--report_dir", report_dir,
            "--instance_ids", *_instance_ids(),
            *ROUTES[route]]
    print(" ".join(argv[:12]) + " ...", flush=True)
    return subprocess.run(argv, check=False).returncode


def read_report(report_dir: str, model_name: str, run_id: str) -> dict:
    """The harness writes `<model>.<run_id>.json`; slashes in a model name become
    double underscores."""
    safe = model_name.replace("/", "__")
    path = os.path.join(report_dir, f"{safe}.{run_id}.json")
    if not os.path.exists(path):
        candidates = [f for f in os.listdir(report_dir) if f.endswith(f".{run_id}.json")]
        if not candidates:
            raise FileNotFoundError(f"no report for run {run_id} in {report_dir}")
        path = os.path.join(report_dir, candidates[0])
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summarize(report: dict) -> str:
    resolved = report.get("resolved_instances", 0)
    total = report.get("total_instances", 0)
    lines = [f"resolved {resolved}/{total}"]
    for key in ("unresolved_instances", "empty_patch_instances", "error_instances"):
        n = report.get(key, 0)
        if n:
            lines.append(f"  {key.replace('_', ' ')}: {n}")
    return "\n".join(lines)


def cmd_gate(route: str, workers: int) -> int:
    """The 50 gold patches must resolve 50/50 before any agent trial is scored."""
    report_dir = os.path.join(ROOT, "_runs", "_gate")
    run_id = f"gold-{route}-{time.strftime('%Y%m%d')}"
    rc = evaluate("gold", run_id, route, report_dir, workers=workers)
    if rc != 0:
        print(f"harness exited {rc}", file=sys.stderr)
    report = read_report(report_dir, "gold", run_id)
    print(summarize(report))
    resolved = report.get("resolved_instances", 0)
    if resolved != dataset.N_INSTANCES:
        missed = sorted(set(_instance_ids()) - set(report.get("resolved_ids", [])))
        print(f"\nGATE FAILED on the {route} route: {resolved}/{dataset.N_INSTANCES}.")
        print(f"  did not resolve: {missed}")
        print("  this grader is not the public grader; try the other image route "
              "before any trial is scored")
        return 1
    print(f"\nGATE PASSED on the {route} route: the official harness reproduces all "
          f"{dataset.N_INSTANCES} gold patches on this machine")
    return 0


def cmd_run(run_dir: str, route: str, workers: int) -> int:
    predictions = os.path.join(run_dir, "predictions.jsonl")
    if not os.path.exists(predictions):
        print(f"no predictions at {predictions}", file=sys.stderr)
        return 2
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    model_name = f"chad-{meta['arm']}"
    run_id = f"{meta['arm']}-rep{meta['rep']}-{route}"
    report_dir = os.path.join(run_dir, "report")
    rc = evaluate(predictions, run_id, route, report_dir, workers=workers)
    if rc != 0:
        print(f"harness exited {rc}", file=sys.stderr)
    report = read_report(report_dir, model_name, run_id)
    print(summarize(report))
    dest = os.path.join(run_dir, "resolved.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"arm": meta["arm"], "rep": meta["rep"], "route": route,
                   "run_id": run_id,
                   "resolved": sorted(report.get("resolved_ids", [])),
                   "resolved_count": report.get("resolved_instances", 0),
                   "total": report.get("total_instances", 0)},
                  fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"  {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="grade the 50 gold patches")
    r = sub.add_parser("run", help="grade one block's predictions")
    r.add_argument("run_dir")
    for p in (g, r):
        p.add_argument("--route", default="arm64", choices=sorted(ROUTES))
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--force", action="store_true",
                       help="skip the preflight (it exists for a reason)")
    args = ap.parse_args()

    if not args.force:
        problems = preflight()
        if problems:
            for p in problems:
                print(f"preflight: {p}", file=sys.stderr)
            return 2
    if args.cmd == "gate":
        return cmd_gate(args.route, args.workers)
    return cmd_run(args.run_dir, args.route, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
