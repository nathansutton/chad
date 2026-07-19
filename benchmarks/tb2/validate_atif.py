#!/usr/bin/env python
"""Validate ATIF trajectories against harbor's OWN pydantic models.

The Terminal-Bench leaderboard requires an ATIF trajectory for every PASSING trial. chad
emits one via `CHAD_TRAJECTORY_JSON` (src/chad/atif.py), but chad cannot import harbor,
so its unit tests can only assert the invariants structurally. This is the real contract
check.

Run it with harbor's interpreter (it needs `harbor` importable), NOT chad's venv. With
harbor installed as a uv tool:

    "$(uv tool dir)/harbor/bin/python" validate_atif.py jobs/<job>/
    "$(uv tool dir)/harbor/bin/python" validate_atif.py path/to/trajectory.json

With a job directory it walks every trial, and FAILS if a trial with reward>0 has no
trajectory — that is precisely the condition that gets a submission rejected.
Exit 0 = submittable. Exit 1 = would be rejected.
"""

import glob
import os
import sys

try:
    from harbor.models.trajectories.trajectory import Trajectory
except ImportError:
    sys.exit("run me with harbor's python, e.g. "
             '"$(uv tool dir)/harbor/bin/python" validate_atif.py <path>')


def validate_file(path: str) -> tuple[bool, str]:
    try:
        t = Trajectory.model_validate_json(open(path).read())
    except Exception as e:
        return False, str(e).splitlines()[0][:160]
    n_tc = sum(len(s.tool_calls or []) for s in t.steps)
    return True, (f"{t.schema_version} · {len(t.steps)} steps · {n_tc} tool calls · "
                  f"agent={t.agent.name}/{t.agent.version} model={t.agent.model_name}")


def main(target: str) -> int:
    if os.path.isfile(target):
        ok, msg = validate_file(target)
        print(("PASS " if ok else "FAIL ") + target + "  " + msg)
        return 0 if ok else 1

    trials = sorted(d for d in glob.glob(os.path.join(target, "*__*")) if os.path.isdir(d))
    if not trials:
        print(f"no trial dirs under {target}")
        return 1

    fails = 0
    for td in trials:
        name = os.path.basename(td)
        rw = os.path.join(td, "verifier", "reward.txt")
        reward = open(rw).read().strip() if os.path.exists(rw) else None
        traj = os.path.join(td, "agent", "trajectory.json")
        passed = reward not in (None, "", "0", "0.0", "0.000")

        if not os.path.exists(traj):
            # Only PASSING trials are required to carry one. A failing trial without a
            # trajectory is untidy; a passing one is a rejected submission.
            level = "FAIL" if passed else "warn"
            fails += passed
            print(f"{level} {name}: reward={reward} NO trajectory.json")
            continue
        ok, msg = validate_file(traj)
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: reward={reward}  {msg}")

    print(f"\n{len(trials) - fails}/{len(trials)} trials OK")
    if fails:
        print("SUBMISSION WOULD BE REJECTED")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
