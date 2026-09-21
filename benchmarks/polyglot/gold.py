"""The gold gate: prove every task is solvable here, and unsolved as shipped. No model.

    uv run python benchmarks/polyglot/gold.py            # all languages, writes manifest.json
    uv run python benchmarks/polyglot/gold.py --only rust/

Each exercise is verified twice with the same `workspace.verify` a trial is scored by:

* the untouched stub must FAIL — an exercise whose stub already passes is a free point,
  and a checkout someone left a solution in is the same thing by accident;
* upstream's reference solution must PASS — otherwise a model's failure on that task
  says something about this machine's toolchain, not about the model.

Only tasks that clear both go into `manifest.json`. The rest are listed with the reason,
and stay out of every score's denominator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog  # noqa: E402
import prepare  # noqa: E402
import workspace  # noqa: E402


@dataclass(frozen=True)
class GateRow:
    task: str
    ok: bool
    reason: str
    seconds: float


def check(task: catalog.Task) -> GateRow:
    ws = os.path.join(catalog.ROOT, "_work", "gold", task.language, task.slug)
    workspace.materialize(task, ws)
    stub = workspace.verify(task, ws)
    if stub.passed:
        return GateRow(task.name, False, "stub already passes", stub.seconds)
    workspace.apply_gold(task, ws)
    gold = workspace.verify(task, ws)
    seconds = stub.seconds + gold.seconds
    if gold.timed_out:
        return GateRow(task.name, False, "reference solution timed out", seconds)
    if not gold.passed:
        return GateRow(task.name, False, "reference solution fails: " + gold.tail[-400:], seconds)
    return GateRow(task.name, True, "", seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="task-name prefix, e.g. `go/` or `rust/acc`")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    skipped = set(prepare.missing_toolchains())
    tasks = [t for t in catalog.load_tasks()
             if t.name.startswith(args.only) and t.language not in skipped]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(check, tasks))
    for row in rows:
        if not row.ok:
            print(f"FAIL {row.task}: {row.reason}")
    passed = sorted(r.task for r in rows if r.ok)
    print(f"gold gate: {len(passed)}/{len(rows)} tasks valid "
          f"({sum(r.seconds for r in rows):.0f} s of test time)")
    if not args.only:
        doc = {"upstream_commit": catalog.UPSTREAM_COMMIT, "tasks": passed,
               "excluded": {r.task: r.reason.split(":")[0] for r in rows if not r.ok}}
        with open(catalog.manifest_path(), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print(f"wrote {catalog.manifest_path()}")
    return 0 if len(passed) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
