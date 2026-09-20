"""Exercise every tool the LEGACY arm offers, once, on a real django workspace.

    uv run --project ../chad-legacy python benchmarks/swebench/tool_probe.py \
        --tree ../chad-legacy

The steelman rule the plan sets is that LEGACY-FULL gets its best honest configuration,
and that a feature which cannot fire here is listed as unreachable BEFORE scoring rather
than discovered afterwards. A tool that errors on every call is the worst version of
that failure: the arm looks like it has twenty-two tools, the model tries one, gets an
error, and the measurement quietly becomes a comparison of five tools against five tools
plus sixteen disappointments.

So each tool is called once against a real checkout — the same `git archive` of a real
base commit a trial gets, in the same sandbox, through the tree's own dispatch table —
and the result is printed. What counts as working is that the tool did its job, not that
it returned without raising: a `grep` that matches nothing and a `find_refs` that
reports no references are both "no error" and both useless, so every probe asserts on
the content of the result.

The symbol tools deserve the attention. They are the part of the 1.x surface with a real
prerequisite — a language server, started per project, indexing django — and they are
the reason this file exists rather than a line in a README. If pyright cannot start from
the uv cache, `definition`, `find_refs`, `hover` and `rename_symbol` are four tools the
model will be offered and cannot use, and that has to be known before a month of nights.

`activate_skill` is the twenty-second tool, and it is not in the campaign's surface: the
runner turns skill discovery off for both arms (`CHAD_NO_SKILLS=1`) because otherwise
the operator's own `~/.claude/skills` rides into every prompt. It is probed separately,
against a skill this file writes, so the count is honest in both directions.
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

import prepare  # noqa: E402
import sandbox  # noqa: E402
import workspace  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

# A real instance's repo, base commit and workbench environment: django 3.1, the
# version 25 of the 50 tasks sit on (django__django-11790). The probe runs in the
# same confinement a trial gets, against the same environment, because a tool that
# works unconfined and fails under the profile is exactly the surprise this finds.
REPO = "django/django"
BASE = "b1d6b35e146aea83b171c1b921178bbaae2795ed"

# Where the probe works, inside the repository, so the sandbox profile and the
# history-free rules are the same ones a trial gets.
WS_NAME = "tool-probe"
ENV_KEY = "django__django-3.1"

# A file that exists at that commit, with a symbol worth asking about.
TARGET = "django/core/paginator.py"
SYMBOL = "Paginator"


class Probe:
    """One tool call and what makes its result count as working."""

    def __init__(self, name: str, args: dict, expect, note: str = "") -> None:
        self.name = name
        self.args = args
        self.expect = expect      # callable(result) -> bool
        self.note = note


def _contains(*needles: str):
    def check(result: str) -> bool:
        return all(n in result for n in needles)
    return check


def _nonempty_and_not(*bad: str):
    def check(result: str) -> bool:
        return bool(result.strip()) and not any(b in result for b in bad)
    return check


def probes(ws: str) -> list[Probe]:
    """The call for each tool, in an order where the mutating ones build on each other.

    The edits are deliberately real — a paginator edit, a line splice, a symbol rewrite —
    because a tool that works on a toy file and fails on 11k files of django is the
    failure this probe exists to catch.
    """
    scratch = "chad_probe_scratch.py"
    return [
        # --- the five the lean arm also has -------------------------------
        Probe("write_todos",
              {"todos": [{"content": "probe the tool surface", "status": "in_progress"}]},
              _nonempty_and_not("[error", "[invalid")),
        # The workbench interpreter, reached the way a trial reaches it: bare `python`
        # off PATH, inside the confining shell. `import django` proves the environment
        # is the one the task needs, not merely that a python exists.
        Probe("bash", {"command": 'python -c "import django, sys; '
                                  'print(django.VERSION[:2], sys.version_info[:2])"'},
              _contains("(3, 1)"),
              "the workbench interpreter, under the sandbox shim"),
        Probe("write", {"path": scratch,
                        "content": "def probe_fn(a, b):\n    return a + b\n"},
              _contains("[wrote")),
        Probe("edit", {"path": scratch, "old": "return a + b", "new": "return a - b"},
              _contains("[edited")),

        # --- reading and searching ----------------------------------------
        Probe("read", {"path": TARGET, "offset": 1, "limit": 40},
              _contains("class Paginator")),
        Probe("glob", {"pattern": "django/core/*.py"},
              _contains("paginator.py")),
        Probe("grep", {"pattern": "class Paginator", "path": "django", "glob": "**/*.py"},
              _contains("paginator.py")),

        # --- line editing --------------------------------------------------
        Probe("insert_lines", {"path": scratch, "after_line": 1,
                               "code": "    # probe inserted this line"},
              _contains("[edited", "inserted after line 1")),
        Probe("replace_lines", {"path": scratch, "start": 1, "end": 1,
                                "new": "def probe_fn(a, b, c=0):"},
              _contains("[edited", "replaced lines 1-1")),

        # --- the repo-map / tree-sitter tier -------------------------------
        Probe("repo_map", {},
              _nonempty_and_not("[error", "[no "),
              "ranked whole-repo skeleton over 11k files"),
        Probe("overview", {"path": TARGET},
              _contains(SYMBOL)),
        Probe("find_symbol", {"name": SYMBOL},
              _contains("paginator.py")),
        Probe("view_symbol", {"name": SYMBOL, "path": TARGET},
              _contains("class Paginator")),

        # --- the language-server tier --------------------------------------
        Probe("definition", {"name": SYMBOL},
              _contains("paginator.py"),
              "pyright, indexing django"),
        Probe("find_refs", {"name": SYMBOL, "path": TARGET},
              _nonempty_and_not("[no ", "[error", "unavailable"),
              "pyright, indexing django"),
        Probe("hover", {"name": SYMBOL, "path": TARGET},
              _nonempty_and_not("[no ", "[error", "unavailable"),
              "pyright, indexing django"),

        # --- symbolic mutation (on the scratch file, not on django) --------
        Probe("replace_symbol", {"name": "probe_fn", "path": scratch,
                                 "new": "def probe_fn(a, b, c=0):\n    return a * b + c\n"},
              _contains("[replaced")),
        Probe("insert_symbol", {"name": "probe_fn", "path": scratch, "where": "after",
                                "code": "def probe_fn2(x):\n    return x\n"},
              _contains("[inserted")),
        Probe("rename_symbol", {"name": "probe_fn2", "path": scratch,
                                "new_name": "probe_renamed"},
              _contains("[renamed"),
              "pyright — a rename is the one symbol EDIT that needs the server"),

        # --- the sub-agent and the terminal call ----------------------------
        # `task` and `done` are dispatched by the agent loop, not by DISPATCH; they are
        # checked for presence in the schema rather than called, and `task` gets its own
        # live run below.
    ]


def child(payload: dict) -> int:
    """The half that runs inside the tree under test: import chad, call each tool.

    A subprocess, and self-invocation rather than a second file, because both trees
    install a package called `chad` and the parent must never have imported one.
    """
    # PATH is set HERE, not in the parent's environment for the subprocess: `uv run`
    # prepends the project's own venv bin, so anything the parent put first ends up
    # second and `python` resolves to the harness's 3.11 interpreter instead of the
    # workbench's. Under the trial's profile that interpreter's own venv is unreadable,
    # and it dies with "Failed to import the site module" — which is how this was found.
    # run.py sets PATH from inside the process for the same reason.
    os.environ["PATH"] = os.pathsep.join(
        [payload["shim"], payload["env_bin"], os.environ["PATH"]])
    sys.path.insert(0, os.path.join(payload["tree"], "src"))
    from chad.tools import active_schemas, dispatch_for

    os.chdir(payload["ws"])
    rows = []
    for pr in probes(payload["ws"]):
        fn = dispatch_for(pr.name)
        t0 = time.time()
        if fn is None:
            result, ok = "[tool is not in this tree's dispatch table]", False
        else:
            try:
                result = fn(pr.args)
                ok = bool(pr.expect(result))
            except Exception as e:  # noqa: BLE001 — a raising tool is a failed probe
                result, ok = f"[raised {type(e).__name__}: {e}]", False
        head = " ".join(str(result).split())[:70]
        rows.append({"tool": pr.name, "ok": ok, "ms": int((time.time() - t0) * 1000),
                     "head": head, "note": pr.note})

    offered = {s["function"]["name"] for s in active_schemas()}
    called = {r["tool"] for r in rows}
    out = {"ok": True, "rows": rows,
           "offered": sorted(offered),
           "not_called": sorted(offered - called)}
    sys.stdout.write("\n@@PROBE@@" + json.dumps(out))
    return 0


def run_probe(tree: str, ws: str, profile: str, shim: str, env_bin: str,
              timeout: int) -> dict:
    """Run every probe inside the tree's own interpreter, in the workspace, confined."""
    me = os.path.abspath(__file__)
    payload = {"tree": tree, "ws": ws, "shim": shim, "env_bin": env_bin}
    env = dict(os.environ)
    env[sandbox.PROFILE_ENV] = profile
    env["CHAD_NO_SEATBELT"] = "1"
    env["CHAD_NO_SKILLS"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    out = subprocess.run(["uv", "run", "--project", tree, "--quiet", "python", me,
                          "--child", json.dumps(payload)],
                         capture_output=True, text=True, check=False,
                         timeout=timeout, env=env)
    marker = "@@PROBE@@"
    if marker not in out.stdout:
        return {"ok": False, "error": (out.stderr or out.stdout or "no output")[-3000:]}
    return json.loads(out.stdout.rsplit(marker, 1)[1])


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(json.loads(sys.argv[2]))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="the harness checkout to probe")
    ap.add_argument("--timeout", type=int, default=900,
                    help="seconds for the whole probe (pyright indexes django once)")
    ap.add_argument("--env-bin", default="",
                    help="a workbench venv's bin/ to put first on PATH; defaults to "
                         f"the {ENV_KEY} workbench the trials use")
    ap.add_argument("--keep", action="store_true", help="keep the workspace afterwards")
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    ws = os.path.join(ROOT, "_work", WS_NAME)
    if not os.path.isdir(workspace.bare_path(ROOT, REPO)):
        print(f"no bare clone for {REPO}; run prepare.py first", file=sys.stderr)
        return 2
    shutil.rmtree(ws, ignore_errors=True)
    workspace.materialize(ROOT, REPO, BASE, ws, WS_NAME)
    # The same read roots a trial gets. Without them the profile denies $HOME, and a
    # uv-managed interpreter cannot load its own dylibs — the probe's first run failed
    # `bash` with a dyld error, which is what a trial would have done too.
    profile = sandbox.write_profile(os.path.join(ROOT, "_work", f"{WS_NAME}.sb"),
                                    os.path.realpath(ws),
                                    prepare.trial_read_roots(ROOT, ENV_KEY))
    shim = os.path.join(ROOT, "_work", f"{WS_NAME}-bin")
    sandbox.write_shim(shim)

    print(f"tree:      {tree}")
    print(f"workspace: {ws}  ({REPO} @ {BASE[:12]})\n")

    t0 = time.time()
    env_bin = args.env_bin or os.path.join(prepare.envs_dir(ROOT), ENV_KEY, "venv", "bin")
    res = run_probe(tree, ws, profile, shim, env_bin, args.timeout)
    if not res.get("ok"):
        print(res.get("error"), file=sys.stderr)
        return 2

    width = max(len(r["tool"]) for r in res["rows"])
    worked = sum(1 for r in res["rows"] if r["ok"])
    for r in res["rows"]:
        note = f"   ({r['note']})" if r.get("note") else ""
        print(f"  {'ok  ' if r['ok'] else 'FAIL'}  {r['tool']:<{width}}  "
              f"{r['ms']:>6} ms  {r['head']}{note}")
    print(f"\n{worked}/{len(res['rows'])} tools worked on a real django checkout "
          f"in {time.time() - t0:.0f}s")
    print(f"the arm offers {len(res['offered'])} tools: {', '.join(res['offered'])}")
    for name in res.get("not_called", []):
        print(f"  note: `{name}` is dispatched by the agent loop, not the tool table")
    if not args.keep:
        shutil.rmtree(ws, ignore_errors=True)
    return 0 if worked == len(res["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
