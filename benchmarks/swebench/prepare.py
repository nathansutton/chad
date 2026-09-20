"""Everything the campaign needs before a model is ever loaded.

    uv run python benchmarks/swebench/prepare.py all

Six steps, each runnable on its own and each safe to re-run:

  specs      dump SWE-bench's own environment recipes to `envs/upstream_specs.json`
  dataset    fetch and pin the 50 instances, split into gold and solver views
  clone      one bare clone per repository
  envs       build the native arm64 workbench environments and write their lockfiles
  canary     prove the sandbox denies what it claims to deny — and allows what it must
  validate   the workbench validation gate: do these environments actually reproduce
             the benchmark's own pass/fail structure on this machine

Nothing here loads a model, and nothing here needs Docker. That is deliberate: the
expensive, fragile parts of a laptop campaign are the ones that can be finished and
checked while the GPU is idle.

WHAT `validate` PROVES, AND WHAT IT DOES NOT
--------------------------------------------
Per instance, at the base commit with the official test patch applied: the
FAIL_TO_PASS tests must fail, a sample of PASS_TO_PASS must pass, and with the gold
patch applied FAIL_TO_PASS must pass. An instance that behaves that way is one where a
correct fix is reachable with the tools the agent is given.

It is a diagnostic, not a filter. Every one of the 50 stays in the scored set whatever
this says, because the official grade comes from the container and not from here —
dropping tasks because our workbench is imperfect would quietly redefine the benchmark.
The status is recorded per instance so the analysis can report a sensitivity check
restricted to the instances that validated.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402
import envspec  # noqa: E402
import sandbox  # noqa: E402
import workspace  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

# A PASS_TO_PASS list can run to 200 tests; the gate only needs enough of them to show
# the environment is not broken, and the full list would cost hours across 50 instances.
P2P_SAMPLE = 8
VALIDATE_TIMEOUT = 900


def envs_dir(root: str) -> str:
    return os.path.join(root, "_envs")


def env_python(root: str, env_key: str) -> str:
    return os.path.join(envs_dir(root), env_key, "venv", "bin", "python")


def _run(argv: list[str], cwd: str = "", timeout: int = 1800,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd or None, capture_output=True, text=True,
                          timeout=timeout, check=False, env=env)


# ---------------------------------------------------------------- specs

def cmd_specs(root: str) -> int:
    """Dump upstream's environment recipes verbatim, so the translation is auditable.

    Read from `swebench` itself rather than transcribed by hand, in a rented interpreter
    so the grading harness's dependency tree never enters this project. The release is
    pinned in `envspec.SPEC_SOURCE` and recorded in the output: a reader can re-run this
    and diff.
    """
    instances = dataset.load_instances(root)
    want: dict[str, list[str]] = {}
    for i in instances:
        want.setdefault(i.repo, [])
        if i.version not in want[i.repo]:
            want[i.repo].append(i.version)
    script = (
        "import json, sys\n"
        "from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as M\n"
        "want = json.loads(sys.argv[1])\n"
        "out = {r: {v: M[r][v] for v in sorted(vs)} for r, vs in want.items()}\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    out = _run(["uv", "run", "--no-project", "--quiet", "--with", envspec.SPEC_SOURCE,
                "python", "-c", script, json.dumps(want)], timeout=900)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        return 1
    dest = os.path.join(root, "envs", "upstream_specs.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"source": envspec.SPEC_SOURCE,
                   "dataset": dataset.DATASET_ID,
                   "specs": json.loads(out.stdout)}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    recipes = envspec.recipes(root)
    print(f"wrote {dest}: {len(recipes)} environments from {envspec.SPEC_SOURCE}")
    for key in sorted(recipes):
        r = recipes[key]
        note = ""
        if r.python != r.python_upstream:
            note = f"  (upstream python {r.python_upstream} -> {r.python})"
        if r.unreachable:
            note += f"  UNREACHABLE: {'; '.join(r.unreachable)}"
        print(f"  {key:<28} python {r.python} runner {r.runner}{note}")
    return 0


# ---------------------------------------------------------------- dataset

def cmd_dataset(root: str, force: bool = False) -> int:
    inst_path, task_path = dataset.fetch(root, force=force)
    instances = dataset.load_instances(root)
    tasks = dataset.load_tasks(root)
    print(dataset.tabulate(instances))
    print(f"  gold view   {inst_path}")
    print(f"  solver view {task_path}  ({len(tasks)} tasks, no patch / tests / hints)")
    return 0


# ---------------------------------------------------------------- clone

def cmd_clone(root: str) -> int:
    instances = dataset.load_instances(root)
    repos = sorted({i.repo for i in instances})
    for repo in repos:
        print(f"cloning {repo} ...", flush=True)
        path = workspace.clone(root, repo)
        print(f"  {path}")
    missing = [i.instance_id for i in instances
               if not workspace.has_commit(root, i.repo, i.base_commit)]
    if missing:
        print(f"  {len(missing)} base commits missing from the clones: {missing[:5]}")
        return 1
    print(f"  all {len(instances)} base commits present")
    return 0


# ---------------------------------------------------------------- envs

def _env_setup_commits(instances: list[dataset.Instance]) -> dict[str, str]:
    """One environment-setup commit per environment key; upstream guarantees it, and a
    violation would mean two instances silently sharing mismatched dependencies."""
    seen: dict[str, str] = {}
    for i in instances:
        prior = seen.setdefault(i.env_key, i.environment_setup_commit)
        if prior != i.environment_setup_commit:
            raise RuntimeError(
                f"{i.env_key} has two environment_setup_commits: {prior} and "
                f"{i.environment_setup_commit}")
    return seen


def _setup_commit_date(root: str, repo: str, sha: str) -> str:
    """The environment-setup commit's date, which is the date the dependency graph is
    resolved as of (envspec deviation 7). Derived, never chosen."""
    out = _run(["git", "-C", workspace.bare_path(root, repo), "show", "-s",
                "--format=%cs", sha], timeout=120)
    date = out.stdout.strip()
    if not date:
        raise RuntimeError(f"no commit date for {repo}@{sha}")
    return date


def _pip(venv_python: str, args: list[str], cwd: str = "", as_of: str = "",
         timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """`as_of` freezes the index to a date and hands the build the environment's own
    setuptools — a period-correct backend cannot build an editable install."""
    pin = ["--exclude-newer", as_of, "--no-build-isolation"] if as_of else []
    return _run(["uv", "pip", "install", "--python", venv_python, *pin, *args],
                cwd=cwd, timeout=timeout)


def _build_requires(src: str) -> list[str]:
    """`[build-system] requires` from the project's own pyproject.toml.

    With build isolation off, uv will not fetch the backend, so it has to already be in
    the environment. sphinx switched to flit at 5.2, and reading the declaration is the
    general answer — hardcoding `flit_core` would break again at the next backend.
    """
    path = os.path.join(src, "pyproject.toml")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        blob = tomllib.load(fh)
    build = blob.get("build-system", {})
    return [str(r) for r in build.get("requires", [])]


def _install_each(venv_python: str, specs: list[str], cwd: str, as_of: str = "") -> list[str]:
    """Install one at a time and report what refused.

    Used as the fallback when a requirements file will not resolve as a whole. django's
    test requirements include packages that never shipped an arm64 wheel and have no
    reason to build here; losing one costs the few tests that import it, and the
    validation gate is what says whether those tests were in scope. Installing them
    one by one turns "the environment failed" into a named, short list.
    """
    failed: list[str] = []
    for spec in specs:
        if _pip(venv_python, [spec], cwd=cwd, as_of=as_of, timeout=900).returncode != 0:
            failed.append(spec)
    return failed


def _requirement_lines(path: str) -> list[str]:
    out: list[str] = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("-"):
            out.append(s)
    return out


def build_env(root: str, key: str, recipe: envspec.NativeRecipe, setup_commit: str,
              rebuild: bool = False) -> dict:
    """One native workbench environment: a template checkout plus a venv of its deps."""
    base = os.path.join(envs_dir(root), key)
    venv = os.path.join(base, "venv")
    src = os.path.join(base, "src")
    if rebuild and os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)

    workspace.export(root, recipe.repo, setup_commit, src)
    # The dependency pins upstream expresses as edits to setup.py are applied to this
    # template only. Trial workspaces are exported fresh from the base commit and never
    # see them: the agent must work on the repository as it actually was.
    patched: list[str] = []
    for p in recipe.patches:
        target = os.path.join(src, p.path)
        if not os.path.exists(target):
            continue
        with open(target, encoding="utf-8") as fh:
            before = fh.read()
        after = p.apply(before)
        if after != before:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(after)
            patched.append(f"{p.path}: {p.pattern} -> {p.replacement}")

    # only-managed: a host interpreter that happens to match the version (macOS ships a
    # 3.9) would make this environment depend on the machine it was built on. uv's own
    # builds are the same bytes everywhere, which is what a repro kit owes a reader.
    mk = _run(["uv", "venv", "--python", recipe.python,
               "--python-preference", "only-managed", venv], timeout=900)
    if mk.returncode != 0:
        return {"env": key, "ok": False, "error": mk.stderr.strip()[-600:]}
    python = env_python(root, key)
    as_of = _setup_commit_date(root, recipe.repo, setup_commit)

    # Deviation 8, plus whatever backend the project declares: with isolation off, uv
    # will not fetch it, so it has to be installed here first.
    boot = _pip(python, [*envspec.BOOTSTRAP_PACKAGES, *_build_requires(src)], cwd=src)
    if boot.returncode != 0:
        return {"env": key, "ok": False, "error": boot.stderr.strip()[-600:]}

    failed: list[str] = []
    # Upstream's pip_packages are exact and deliberately post-date the repo, so they are
    # the one thing installed without the date pin (deviation 7's first carve-out).
    if recipe.pip_packages:
        if _pip(python, list(recipe.pip_packages), cwd=src).returncode != 0:
            failed += _install_each(python, list(recipe.pip_packages), src)
    if recipe.requirements_file:
        req = os.path.join(src, recipe.requirements_file)
        if not os.path.exists(req):
            return {"env": key, "ok": False,
                    "error": f"missing {recipe.requirements_file} at {setup_commit[:12]}"}
        if _pip(python, ["-r", recipe.requirements_file], cwd=src,
                as_of=as_of).returncode != 0:
            failed += _install_each(python, _requirement_lines(req), src, as_of=as_of)
    if recipe.extras:
        target = ".[" + ",".join(recipe.extras) + "]"
        inst = _pip(python, ["-e", target], cwd=src, as_of=as_of)
        if inst.returncode != 0:
            return {"env": key, "ok": False, "error": inst.stderr.strip()[-600:]}
        # Deviation 4: keep what the install resolved, drop the link to this checkout.
        _run(["uv", "pip", "uninstall", "--python", python, recipe.uninstall],
             cwd=src, timeout=600)

    freeze = _run(["uv", "pip", "freeze", "--python", python], cwd=src, timeout=300)
    lock = os.path.join(root, "envs", f"{key}.lock")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(f"# {key}: python {recipe.python} "
                 f"(upstream {recipe.python_upstream}), {envspec.SPEC_SOURCE}\n")
        fh.write(f"# dependencies resolved as of {as_of}, the date of environment "
                 f"setup commit {setup_commit[:12]}\n")
        fh.write(freeze.stdout)

    ver = _run([python, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"])
    report = {"env": key, "ok": True, "python": ver.stdout.strip(),
              "python_upstream": recipe.python_upstream,
              "setup_commit": setup_commit, "resolved_as_of": as_of,
              "runner": recipe.runner,
              "patched": patched, "unreachable": list(recipe.unreachable),
              # Recorded because it is the deviation most likely to be doubted: the
              # package under test must come from the workspace, never from site-packages.
              "unlinked": recipe.uninstall if recipe.extras else "",
              "failed_packages": failed, "lock": os.path.relpath(lock, root)}
    with open(os.path.join(base, "env.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def cmd_envs(root: str, only: str = "", rebuild: bool = False) -> int:
    instances = dataset.load_instances(root)
    setup = _env_setup_commits(instances)
    recipes = envspec.recipes(root)
    keys = sorted(setup)
    if only:
        keys = [k for k in keys if k in set(only.split(","))]
    bad = 0
    for key in keys:
        t0 = time.time()
        print(f"building {key} (python {recipes[key].python}) ...", flush=True)
        report = build_env(root, key, recipes[key], setup[key], rebuild=rebuild)
        if not report["ok"]:
            bad += 1
            print(f"  FAILED: {report['error']}")
            continue
        notes = []
        if report["failed_packages"]:
            notes.append(f"{len(report['failed_packages'])} package(s) would not "
                         f"install: {', '.join(report['failed_packages'][:4])}")
        if report["unreachable"]:
            notes.append(f"unreachable: {'; '.join(report['unreachable'])}")
        print(f"  python {report['python']} in {time.time() - t0:.0f}s"
              + ("" if not notes else "\n  " + "\n  ".join(notes)))
    print(f"{len(keys) - bad}/{len(keys)} environments built")
    return 1 if bad else 0


# ---------------------------------------------------------------- canary

def read_roots() -> tuple[str, ...]:
    """Paths under `$HOME` a confined trial legitimately reads: uv's managed
    interpreters (the venv's python is a symlink into that store) and uv's own cache,
    which holds the installed wheels the venv links against."""
    home = os.path.expanduser("~")
    return (os.path.join(home, ".local", "share", "uv"),
            os.path.join(home, ".cache", "uv"),
            os.path.join(home, "Library", "Caches", "uv"))


def trial_read_roots(root: str, env_key: str) -> tuple[str, ...]:
    return (os.path.join(envs_dir(root), env_key), *read_roots())


def cmd_canary(root: str) -> int:
    """Prove the boundary before trusting a month of nights to it."""
    if not sandbox.available():
        print("sandbox-exec is not available on this machine")
        return 1
    area = os.path.join(root, "_canary")
    ws = os.path.join(area, "workspace")
    outside = os.path.join(area, "outside")
    shutil.rmtree(area, ignore_errors=True)
    os.makedirs(ws)
    os.makedirs(outside)
    # The planted "fix" sits beside the workspace, under $HOME, exactly where a real
    # leak would live: another checkout of the same project on the same disk.
    secret = os.path.join(outside, "FIX.patch")
    with open(secret, "w", encoding="utf-8") as fh:
        fh.write("diff --git a/leaked b/leaked\n")
    with open(os.path.join(ws, "README"), "w", encoding="utf-8") as fh:
        fh.write("workspace\n")

    profile = sandbox.write_profile(os.path.join(area, "canary.sb"),
                                    os.path.realpath(ws), read_roots())
    results = sandbox.run_probes(profile, os.path.realpath(ws), secret,
                                 os.path.join(outside, "written"))
    print(f"sandbox profile: {profile}")
    print(sandbox.format_probes(results))
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"{len(failed)} probe(s) failed — no trials until this is clean")
        return 1
    print(f"all {len(results)} probes behaved as declared")
    return 0


# ---------------------------------------------------------------- validate

def _sample(tests: tuple[str, ...], n: int) -> tuple[str, ...]:
    """A deterministic sample: the list is already in upstream's order, so take a
    stride through it rather than the head, which in both repos clusters inside one
    test class."""
    if len(tests) <= n:
        return tests
    step = len(tests) / n
    return tuple(tests[int(i * step)] for i in range(n))


def _run_tests(root: str, inst: dataset.Instance, recipe: envspec.NativeRecipe,
               ws: str, tests: tuple[str, ...], profile: str) -> bool:
    """True iff every named test passed. Run under the same confinement a trial gets,
    so an environment that only works unconfined is caught here and not in the campaign."""
    if not tests:
        return True
    python = env_python(root, inst.env_key)
    argv = envspec.test_argv(recipe, python, tests)
    env = dict(os.environ)
    env.update(envspec.workspace_env(ws))
    env["PATH"] = os.path.join(envs_dir(root), inst.env_key, "venv", "bin") \
        + os.pathsep + env.get("PATH", "")
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    quoted = " ".join(shlex.quote(a) for a in argv)
    out = sandbox.run_confined(profile, quoted, cwd=ws, timeout=VALIDATE_TIMEOUT, env=env)
    return out.returncode == 0


def validate_instance(root: str, inst: dataset.Instance,
                      recipe: envspec.NativeRecipe) -> dict:
    """base: FAIL_TO_PASS must fail and PASS_TO_PASS must pass; gold: both must pass."""
    ws = os.path.join(root, "_validate", inst.instance_id)
    workspace.materialize(root, inst.repo, inst.base_commit, ws, inst.instance_id)
    profile = sandbox.write_profile(
        os.path.join(root, "_validate", f"{inst.instance_id}.sb"),
        os.path.realpath(ws), trial_read_roots(root, inst.env_key))
    row = {"instance_id": inst.instance_id, "env": inst.env_key}
    try:
        if not workspace.apply_patch(ws, inst.test_patch):
            row["status"] = "test-patch-failed"
            return row
        # Some entries are not runnable by name here; sampling AFTER the filter keeps
        # the sample the size it claims to be.
        f2p = envspec.runnable(recipe, inst.fail_to_pass)
        p2p = _sample(envspec.runnable(recipe, inst.pass_to_pass), P2P_SAMPLE)
        row["unrunnable_entries"] = ((len(inst.fail_to_pass) - len(f2p))
                                     + (len(inst.pass_to_pass)
                                        - len(envspec.runnable(recipe, inst.pass_to_pass))))
        if not f2p:
            row["status"] = "no-runnable-f2p"
            return row
        row["f2p_fails_at_base"] = not _run_tests(root, inst, recipe, ws, f2p, profile)
        row["p2p_passes_at_base"] = _run_tests(root, inst, recipe, ws, p2p, profile)
        # Back to the seed commit before the gold half. A sphinx test run writes built
        # documentation into `tests/roots/`, and the second run reads it: without the
        # reset, the gold phase inherits the base phase's output and reports a gold
        # patch as not fixing what it plainly fixes.
        workspace.reset(ws)
        if not workspace.apply_patch(ws, inst.test_patch):
            row["status"] = "test-patch-failed-on-reset"
            return row
        if not workspace.apply_patch(ws, inst.patch):
            row["status"] = "gold-patch-failed"
            return row
        row["f2p_passes_with_gold"] = _run_tests(root, inst, recipe, ws, f2p, profile)
        row["status"] = ("valid" if (row["f2p_fails_at_base"]
                                     and row["p2p_passes_at_base"]
                                     and row["f2p_passes_with_gold"]) else "invalid")
        return row
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def cmd_validate(root: str, only: str = "") -> int:
    instances = dataset.load_instances(root)
    recipes = envspec.recipes(root)
    if only:
        want = set(only.split(","))
        instances = [i for i in instances if i.instance_id in want]
    out_path = os.path.join(root, "_data", "workbench_validation.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for inst in instances:
            t0 = time.time()
            row = validate_instance(root, inst, recipes[inst.env_key])
            row["wall_s"] = round(time.time() - t0, 1)
            rows.append(row)
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            print(f"  {row['instance_id']:<32} {row['status']:<18} {row['wall_s']:>6.0f}s",
                  flush=True)
    valid = sum(1 for r in rows if r["status"] == "valid")
    print(f"\nworkbench validation: {valid}/{len(rows)} instances reproduce the "
          f"benchmark's structure natively")
    print(f"  rows: {out_path}")
    if valid < 0.8 * len(rows):
        print("  below the gate — the agent would be handed broken tooling; fix the "
              "environments before spending nights on trials")
        return 1
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["specs", "dataset", "clone", "envs", "canary",
                                     "validate", "all"])
    ap.add_argument("--only", default="", help="comma-separated env keys or instance ids")
    ap.add_argument("--force", action="store_true", help="re-download the dataset")
    ap.add_argument("--rebuild", action="store_true", help="discard and rebuild envs")
    args = ap.parse_args()

    if args.step == "dataset":
        return cmd_dataset(ROOT, force=args.force)
    if args.step == "specs":
        return cmd_specs(ROOT)
    if args.step == "clone":
        return cmd_clone(ROOT)
    if args.step == "envs":
        return cmd_envs(ROOT, only=args.only, rebuild=args.rebuild)
    if args.step == "canary":
        return cmd_canary(ROOT)
    if args.step == "validate":
        return cmd_validate(ROOT, only=args.only)
    for step in (lambda: cmd_dataset(ROOT, force=args.force),
                 lambda: cmd_specs(ROOT),
                 lambda: cmd_clone(ROOT),
                 lambda: cmd_envs(ROOT, only=args.only, rebuild=args.rebuild),
                 lambda: cmd_canary(ROOT),
                 lambda: cmd_validate(ROOT, only=args.only)):
        rc = step()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
