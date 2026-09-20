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

ARCHITECTURE: THE FALLBACK IS THE ROUTE
---------------------------------------
The plan's first choice was to build the images here for arm64 (`--namespace ''`) and
keep the published x86_64 images under emulation as a fallback. Measured on this
machine, the first choice does not work and the fallback is the only honest route.

A locally built image is built *today*, and SWE-bench's image recipe installs the
project with `pip install -e .[test]` and no date pin. For sphinx that resolves
`docutils>=0.12` — unbounded in sphinx 3.x and 5.x — to a 2026 docutils, and the suite
dies on `No module named 'docutils.utils.roman'`, which has not existed since docutils
0.21. It is the same failure the native workbench hit, and `envspec.py` fixes it there
with `--exclude-newer`; it cannot be fixed here, because patching upstream's image build
is precisely how a grader stops being the public grader. Probed on two gold patches:
arm64 local build 1/2, published x86_64 2/2, the failure being sphinx.

So: `--route x86_64`, the images published when those dependency graphs still resolved,
run under emulation. Slower, and the wall cap in `run.py` is ours anyway and disclosed.
`--route arm64` is kept so the finding can be reproduced rather than taken on trust.

Published arm64 images do exist on Docker Hub (`swebench/sweb.eval.arm64.…`, and they
pull), but neither swebench 4.x nor 5.x can address them: both hardcode the image
architecture and expose no flag for it. Renaming an arm64 image to the x86_64 name the
harness expects would work and is not done — a grader that has been tricked about what
it is running is not a grader.
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
    # Build the images here. MEASURED TO FAIL THE GOLD GATE — see the module docstring.
    # Kept because the finding is worth being able to reproduce.
    "arm64": ["--namespace", ""],
    # The published images. This is the route that works.
    "x86_64": ["--namespace", "swebench"],
}

# Published images have to be fetched by hand on Apple Silicon. `docker pull` without an
# explicit platform resolves the manifest list for the host and reports "no matching
# manifest for linux/arm64/v8"; the harness's own fetch then inspects an image that is
# not there and fails the instance with a 404 rather than a diagnosis.
PUBLISHED_NAMESPACE = "swebench"
PUBLISHED_PLATFORM = {"x86_64": "linux/amd64"}


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


def published_image(instance_id: str, route: str) -> str:
    """The published image for an instance, spelled the way the harness spells it —
    Docker Hub forbids a double underscore in a repository name, so upstream substitutes
    `_1776_`."""
    key = f"sweb.eval.{route}.{instance_id}:latest".replace("__", "_1776_")
    return f"{PUBLISHED_NAMESPACE}/{key}".lower()


def prepull(instance_ids: list[str], route: str) -> list[str]:
    """Fetch the published images this run needs; returns the ids that could not be got.

    Done here rather than left to the harness because the harness cannot do it on this
    machine: it asks Docker for an image whose manifest list has no arm64 entry, gets
    nothing, and reports a 404 per instance at grading time — after the run has started.
    Pulling first turns that into one clear failure before anything is scored.
    """
    platform = PUBLISHED_PLATFORM.get(route)
    if not platform:
        return []
    missing = []
    for n, iid in enumerate(instance_ids, 1):
        image = published_image(iid, route)
        have = subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, check=False, timeout=120)
        if have.returncode == 0:
            continue
        print(f"  pulling [{n}/{len(instance_ids)}] {iid} ...", flush=True)
        out = subprocess.run(["docker", "pull", "-q", "--platform", platform, image],
                             capture_output=True, text=True, check=False, timeout=3600)
        if out.returncode != 0:
            missing.append(iid)
            print(f"    FAILED: {out.stderr.strip()[:160]}")
    return missing


def docker_image_bytes() -> int:
    """Total size of the local image store, so a route's disk cost can be measured as a
    delta rather than guessed. The campaign shares a laptop disk with the weights."""
    out = subprocess.run(
        ["docker", "system", "df", "--format", "{{.Type}}\t{{.Size}}"],
        capture_output=True, text=True, check=False, timeout=120)
    for line in out.stdout.splitlines():
        kind, _, size = line.partition("\t")
        if kind.strip() == "Images":
            return _parse_size(size.strip())
    return 0


def _parse_size(text: str) -> int:
    units = {"B": 1, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12}
    for suffix in sorted(units, key=len, reverse=True):
        if text.upper().endswith(suffix):
            try:
                return int(float(text[:-len(suffix)].strip()) * units[suffix])
            except ValueError:
                return 0
    return 0


def evaluate(predictions: str, run_id: str, route: str, report_dir: str,
             instance_ids: list[str], workers: int = 4, timeout: int = 1800) -> int:
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
            "--instance_ids", *instance_ids,
            *ROUTES[route]]
    print(" ".join(argv[:12]) + " ...", flush=True)
    # Run FROM the report directory. The harness honours `--report_dir` for the summary
    # but writes its per-instance `logs/run_evaluation/...` tree relative to wherever it
    # was invoked, and on some releases the summary lands there too. Pointing its working
    # directory at the run's own folder keeps every artifact together and stops the
    # repository root collecting `logs/` and stray `gold.*.json` after a grading pass.
    return subprocess.run(argv, cwd=report_dir, check=False).returncode


def read_report(report_dir: str, model_name: str, run_id: str) -> dict:
    """The harness writes `<model>.<run_id>.json`; slashes in a model name become
    double underscores."""
    safe = model_name.replace("/", "__")
    path = os.path.join(report_dir, f"{safe}.{run_id}.json")
    if not os.path.exists(path):
        # The naming has moved between releases; match on the run id, which is ours.
        candidates = sorted(f for f in os.listdir(report_dir)
                            if f.endswith(f".{run_id}.json"))
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


def cmd_gate(route: str, workers: int, only: str = "", label: str = "") -> int:
    """The 50 gold patches must resolve 50/50 before any agent trial is scored.

    `--only` grades a subset. It cannot pass the gate — the gate is the whole set — but
    it is how the route gets tried before hours of image builds are spent on it, and how
    the per-instance disk cost is measured on a laptop that has to hold the weights too.
    """
    ids = sorted(set(only.split(","))) if only else _instance_ids()
    report_dir = os.path.join(ROOT, "_runs", "_gate")
    run_id = f"gold-{route}-{label or time.strftime('%Y%m%d')}"
    before = docker_image_bytes()
    missing = prepull(ids, route)
    if missing:
        print(f"could not fetch {len(missing)} published image(s): {missing[:5]}",
              file=sys.stderr)
        return 1
    rc = evaluate("gold", run_id, route, report_dir, ids, workers=workers)
    if rc != 0:
        print(f"harness exited {rc}", file=sys.stderr)
    grew = docker_image_bytes() - before
    report = read_report(report_dir, "gold", run_id)
    print(summarize(report))
    print(f"  image store grew {grew / 1e9:.1f} GB over {len(ids)} instance(s)")
    resolved = report.get("resolved_instances", 0)
    if only:
        print(f"\nSUBSET only: {resolved}/{len(ids)} on the {route} route. The gate is "
              f"all {dataset.N_INSTANCES}; run it without --only to pass it.")
        return 0 if resolved == len(ids) else 1
    if resolved != dataset.N_INSTANCES:
        missed = sorted(set(ids) - set(report.get("resolved_ids", [])))
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
    os.makedirs(report_dir, exist_ok=True)
    missing = prepull(_instance_ids(), route)
    if missing:
        print(f"could not fetch {len(missing)} published image(s): {missing[:5]}",
              file=sys.stderr)
        return 1
    rc = evaluate(predictions, run_id, route, report_dir, _instance_ids(),
                  workers=workers)
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
    g.add_argument("--only", default="",
                   help="comma-separated instance ids — tries the route cheaply; "
                        "cannot pass the gate")
    g.add_argument("--label", default="", help="run id suffix (default: today)")
    r = sub.add_parser("run", help="grade one block's predictions")
    r.add_argument("run_dir")
    for p in (g, r):
        # x86_64 by default: the locally built arm64 images are measured NOT to pass the
        # gold gate (see the module docstring), so defaulting to them would hand a new
        # reader the broken route first.
        p.add_argument("--route", default="x86_64", choices=sorted(ROUTES))
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
        return cmd_gate(args.route, args.workers, only=args.only, label=args.label)
    return cmd_run(args.run_dir, args.route, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
