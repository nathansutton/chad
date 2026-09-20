"""benchmarks/swebench: the properties a reader has to be able to trust without a GPU.

Four of them, and each one is a thing that would silently corrupt the measurement rather
than break it loudly:

  1. the solver cannot see the answer — the task view carries no gold field, and the
     loader refuses one that does;
  2. a workspace has no future in it — one commit, no remotes, no extra refs;
  3. the sandbox profile says what the module docstring says it says, and the canary's
     own targets are really outside the allowed set;
  4. upstream's environment recipes are translated exactly, and an upstream command in a
     shape the translator does not recognise raises instead of being skipped.

Stdlib only, no model, no Docker, no network, and nothing here needs the kit to have been
prepared: every test builds what it needs. The sandbox tests that actually apply a
profile are skipped off macOS, where `sandbox-exec` does not exist.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, "benchmarks", "swebench")
DARWIN = sys.platform == "darwin"


def _load(name: str):
    """Import a kit module by path. The kit is a directory of scripts, not a package —
    it is meant to be run with `uv run python benchmarks/swebench/<step>.py` — so the
    tests load it the same way the scripts load each other."""
    if KIT not in sys.path:
        sys.path.insert(0, KIT)
    spec = importlib.util.spec_from_file_location(
        f"swebench_kit_{name}", os.path.join(KIT, f"{name}.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Registered before execution: `dataclasses` resolves a field's annotations through
    # `sys.modules[cls.__module__]`, and a module that is not there yet makes every
    # frozen dataclass in the file fail to build.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dataset = _load("dataset")
envspec = _load("envspec")
sandbox = _load("sandbox")
workspace = _load("workspace")


# ----------------------------------------------------------------- the split

def test_task_view_has_no_gold_field():
    """The solver's row carries the problem statement and nothing that answers it."""
    assert set(dataset.Task.__dataclass_fields__) == set(dataset._TASK_FIELDS)
    for field in dataset._GOLD_FIELDS:
        assert field not in dataset.Task.__dataclass_fields__


def test_load_tasks_refuses_a_row_carrying_gold(tmp_path):
    """Not merely unused: a gold field in the solver's file is an error.

    The guard matters because `tasks.jsonl` is a generated file. Someone regenerating it
    by hand, or a future column added upstream, would otherwise put the patch in front of
    the model and nothing would complain.
    """
    data = tmp_path / "_data"
    data.mkdir()
    row = {"instance_id": "x__y-1", "repo": "x/y", "base_commit": "a" * 40,
           "version": "1.0", "problem_statement": "boom"}
    (data / "tasks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert dataset.load_tasks(str(tmp_path))[0].instance_id == "x__y-1"

    row["patch"] = "diff --git a/f b/f"
    (data / "tasks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gold fields"):
        dataset.load_tasks(str(tmp_path))


def test_task_keys_name_the_environment():
    t = dataset.Task(instance_id="django__django-11790", repo="django/django",
                     base_commit="a" * 40, version="3.1", problem_statement="")
    assert t.repo_key == "django__django"
    assert t.env_key == "django__django-3.1"


# ----------------------------------------------------------------- workspaces

def _seed_repo(path: str) -> str:
    """A tiny repository with two commits, the second standing in for the fix."""
    os.makedirs(path)
    sh = lambda *a: subprocess.run(["git", *a], cwd=path, check=True,  # noqa: E731
                                   capture_output=True)
    sh("init", "--quiet", "-b", "main")
    sh("config", "user.email", "t@localhost")
    sh("config", "user.name", "t")
    with open(os.path.join(path, "mod.py"), "w", encoding="utf-8") as fh:
        fh.write("def f():\n    return 1\n")
    sh("add", "-A")
    sh("commit", "--quiet", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True,
                          text=True, check=True).stdout.strip()
    with open(os.path.join(path, "mod.py"), "w", encoding="utf-8") as fh:
        fh.write("def f():\n    return 2\n")
    sh("commit", "--quiet", "-am", "THE FIX")
    return base


def test_runner_scratch_never_reaches_a_prediction(tmp_path):
    """A prediction is the one artifact the grader sees, so it carries source changes
    and nothing else.

    The trial's TMPDIR sits inside the workspace — the one place the sandbox profile
    certainly lets it write — which puts the runner's own scratch one `git add -A` away
    from the patch. Measured on the first live legacy trial: node, spawned by the
    language server, left 81 compile-cache files there and the prediction came out as
    207 KB of them with no source change in it at all.
    """
    origin = str(tmp_path / "origin")
    base = _seed_repo(origin)
    root = str(tmp_path / "kit")
    os.makedirs(os.path.join(root, "_repos"))
    subprocess.run(["git", "clone", "--bare", "--quiet", origin,
                    workspace.bare_path(root, "x/y")], check=True, capture_output=True)
    ws = str(tmp_path / "ws")
    workspace.materialize(root, "x/y", base, ws, "x__y-1")

    junk = os.path.join(ws, workspace.SCRATCH, "node-compile-cache", "v1")
    os.makedirs(junk)
    for i in range(5):
        with open(os.path.join(junk, f"c{i}"), "w", encoding="utf-8") as fh:
            fh.write("cachejunk" * 200)
    with open(os.path.join(ws, "mod.py"), "w", encoding="utf-8") as fh:
        fh.write("def f():\n    return 3\n")

    patch = workspace.diff(ws)
    assert workspace.changed_files(patch) == ("mod.py",)
    assert workspace.SCRATCH not in patch
    # ...and the model does not see the runner's scratch in its own `git status`.
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ws,
                            capture_output=True, text=True, check=True).stdout
    assert workspace.SCRATCH not in status


def test_materialized_workspace_has_no_future(tmp_path):
    """The point of exporting instead of cloning: the fix is not in the object store."""
    origin = str(tmp_path / "origin")
    base = _seed_repo(origin)
    root = str(tmp_path / "kit")
    os.makedirs(os.path.join(root, "_repos"))
    subprocess.run(["git", "clone", "--bare", "--quiet", origin,
                    workspace.bare_path(root, "x/y")], check=True, capture_output=True)

    ws = str(tmp_path / "ws")
    workspace.materialize(root, "x/y", base, ws, "x__y-1")

    log = subprocess.run(["git", "log", "--all", "--oneline"], cwd=ws,
                         capture_output=True, text=True, check=True).stdout
    assert "THE FIX" not in log
    assert len(log.strip().splitlines()) == 1
    with open(os.path.join(ws, "mod.py"), encoding="utf-8") as fh:
        assert "return 1" in fh.read()
    workspace.assert_history_free(ws)


def test_assert_history_free_rejects_a_second_commit(tmp_path):
    origin = str(tmp_path / "origin")
    base = _seed_repo(origin)
    root = str(tmp_path / "kit")
    os.makedirs(os.path.join(root, "_repos"))
    subprocess.run(["git", "clone", "--bare", "--quiet", origin,
                    workspace.bare_path(root, "x/y")], check=True, capture_output=True)
    ws = str(tmp_path / "ws")
    workspace.materialize(root, "x/y", base, ws, "x__y-1")
    subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", "extra"],
                   cwd=ws, check=True, capture_output=True)
    with pytest.raises(RuntimeError, match="2 commits"):
        workspace.assert_history_free(ws)


def test_diff_includes_files_the_trial_created(tmp_path):
    """A prediction that omitted new files would grade as an unexplained failure."""
    origin = str(tmp_path / "origin")
    base = _seed_repo(origin)
    root = str(tmp_path / "kit")
    os.makedirs(os.path.join(root, "_repos"))
    subprocess.run(["git", "clone", "--bare", "--quiet", origin,
                    workspace.bare_path(root, "x/y")], check=True, capture_output=True)
    ws = str(tmp_path / "ws")
    workspace.materialize(root, "x/y", base, ws, "x__y-1")

    with open(os.path.join(ws, "mod.py"), "w", encoding="utf-8") as fh:
        fh.write("def f():\n    return 2\n")
    with open(os.path.join(ws, "new.py"), "w", encoding="utf-8") as fh:
        fh.write("helper = 1\n")
    patch = workspace.diff(ws)
    assert workspace.changed_files(patch) == ("mod.py", "new.py")


# ----------------------------------------------------------------- the sandbox

def test_profile_denies_network_writes_and_home_reads(tmp_path):
    text = sandbox.profile_text(str(tmp_path), (str(tmp_path / "env"),))
    assert "(deny network*)" in text
    assert "(deny file-write*)" in text
    assert f'(subpath "{os.path.expanduser("~")}")' in text
    # Metadata is the last rule, so stat survives while contents stay denied.
    assert text.rstrip().endswith("(allow file-read-metadata)")
    assert text.index("(deny file-read*") < text.index("(allow file-read-metadata)")


def test_shim_fails_closed_without_a_profile(tmp_path):
    """An unconfined trial must be impossible, not merely unlikely: it would be
    indistinguishable in the results from a confined one."""
    shim = sandbox.write_shim(str(tmp_path / "bin"))
    assert os.access(shim, os.X_OK)
    env = {k: v for k, v in os.environ.items() if k != sandbox.PROFILE_ENV}
    out = subprocess.run([shim, "-c", "echo leaked"], capture_output=True, text=True,
                         env=env, check=False)
    assert out.returncode == sandbox.SHIM_MISSING_PROFILE_EXIT
    assert "leaked" not in out.stdout


def test_canary_refuses_targets_that_are_not_actually_outside(tmp_path):
    """A planted secret in /tmp is readable by design; a probe that reported it as a leak
    would teach the reader to ignore the canary."""
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    with pytest.raises(ValueError, match="temp root"):
        sandbox.probes(ws, "/tmp/planted-fix", str(tmp_path / "out"))
    inside = os.path.join(ws, "fix")
    with pytest.raises(ValueError, match="inside the workspace"):
        sandbox.probes(ws, inside, str(tmp_path / "out"))


@pytest.mark.skipif(not DARWIN, reason="sandbox-exec is macOS-only")
def test_profile_actually_enforces(tmp_path):
    """Applying a profile proves nothing about confinement unless both directions are
    checked: the denied write must not land AND the allowed one must."""
    ws = tmp_path / "ws"
    ws.mkdir()
    profile = sandbox.write_profile(str(tmp_path / "p.sb"), os.path.realpath(ws), ())

    # This repository is the leak the profile exists to prevent: another checkout of a
    # project under test on the same disk. `tmp_path` would not do — it lives under a
    # temp root, which is writable and readable on purpose.
    secret = os.path.join(ROOT, "README.md")
    denied = sandbox.run_confined(profile, f"cat {secret}", cwd=str(ws))
    assert denied.returncode != 0
    assert not denied.stdout.strip()

    allowed = sandbox.run_confined(profile, "printf ok > inside && cat inside",
                                   cwd=str(ws))
    assert allowed.returncode == 0
    assert allowed.stdout.strip() == "ok"

    # stat must survive, or pytest cannot find its rootdir and collection aborts.
    stat = sandbox.run_confined(profile, f"test -e {secret}", cwd=str(ws))
    assert stat.returncode == 0

    outside_write = sandbox.run_confined(
        profile, f"touch {os.path.join(ROOT, '.swb-should-not-exist')}", cwd=str(ws))
    assert outside_write.returncode != 0
    assert not os.path.exists(os.path.join(ROOT, ".swb-should-not-exist"))


# ----------------------------------------------------------------- env recipes

def test_sed_translation_matches_gnu_sed_semantics():
    p = envspec._parse_sed("sed -i 's/Jinja2>=2.3/Jinja2<3.0/' setup.py")
    assert p.path == "setup.py"
    # No `g` flag upstream, so at most one substitution per line.
    assert p.apply("a Jinja2>=2.3 b Jinja2>=2.3\n") == "a Jinja2<3.0 b Jinja2>=2.3\n"


def test_conditional_pre_install_becomes_two_guarded_patches():
    yes, no = envspec._parse_conditional(
        "grep -q 'sphinxcontrib-htmlhelp>=2.0.0' setup.py && "
        "sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py || "
        "sed -i 's/sphinxcontrib-htmlhelp/sphinxcontrib-htmlhelp<=2.0.4/' setup.py")
    assert yes.guard_present is True and no.guard_present is False
    present = "install_requires = ['sphinxcontrib-htmlhelp>=2.0.0']\n"
    assert yes.applies(present) and not no.applies(present)
    absent = "install_requires = ['sphinxcontrib-htmlhelp']\n"
    assert no.applies(absent) and not yes.applies(absent)


def test_unrecognised_upstream_command_raises():
    """A skipped dependency pin surfaces much later as a version conflict nobody can
    place, so the translator refuses rather than ignores."""
    with pytest.raises(ValueError, match="unrecognised"):
        envspec.translate_pre_install(("python -m pip install cython",))
    with pytest.raises(ValueError, match="back-reference"):
        envspec._parse_sed("sed -i 's/pytest/& -rA/' tox.ini")


def test_apt_get_is_recorded_as_unreachable_not_dropped():
    patches, unreachable = envspec.translate_pre_install(
        ("sed -i 's/pytest/pytest -rA/' tox.ini",
         "apt-get update && apt-get install -y graphviz"))
    assert len(patches) == 1
    assert unreachable == ("apt-get update && apt-get install -y graphviz",)


def test_our_own_patches_ride_after_upstreams_and_are_self_limiting():
    """EXTRA_PATCHES are declared deviations, applied the same way upstream's are. Each
    is a no-op where its text is absent, which is what keeps it scoped to the versions
    that need it."""
    spec = envspec.UpstreamSpec(repo="sphinx-doc/sphinx", version="3.1", python="3.9",
                                install="python -m pip install -e .[test]",
                                test_cmd="tox --current-env -epy39 -v --",
                                pre_install=("sed -i 's/pytest/pytest -rA/' tox.ini",))
    r = envspec.native(spec)
    assert len(r.patches) == 1 + len(envspec.EXTRA_PATCHES)
    marker = r.patches[-1]
    assert marker.apply("        'typed_ast',  # for py35-37\n") == (
        "        \"typed_ast; python_version < '3.8'\",  # for py35-37\n")
    # sphinx 3.4 already carries the marker, so the same patch changes nothing there.
    already = "        \"typed_ast; python_version < '3.8'\",\n"
    assert marker.apply(already) == already


def test_python_deviation_is_recorded_on_the_recipe():
    spec = envspec.UpstreamSpec(repo="django/django", version="3.1", python="3.6",
                                install="python -m pip install -e .",
                                test_cmd="./tests/runtests.py", packages="requirements.txt")
    r = envspec.native(spec)
    assert (r.python, r.python_upstream) == ("3.8", "3.6")
    assert r.requirements_file == envspec.DJANGO_REQUIREMENTS
    assert r.extras == ()          # django's package is never installed at all
    assert r.runner == "django"


def test_django_labels_and_dropped_docstring_entries():
    assert (envspec.django_label("test_x (auth_tests.test_forms.T)")
            == "auth_tests.test_forms.T.test_x")
    assert envspec.django_label("a.b.C.test_x") == "a.b.C.test_x"
    # A docstring line upstream's log parser captured as if it were a test.
    assert envspec.django_label("{% get_admin_log %} works without a user") == ""

    spec = envspec.UpstreamSpec(repo="django/django", version="3.1", python="3.6",
                                install="python -m pip install -e .",
                                test_cmd="./tests/runtests.py", packages="requirements.txt")
    r = envspec.native(spec)
    tests = ("test_x (a.b.C)", "documents the test above")
    assert envspec.runnable(r, tests) == ("test_x (a.b.C)",)
    argv = envspec.test_argv(r, "/py", tests)
    assert argv[-1] == "a.b.C.test_x"
    assert "documents the test above" not in argv


def test_pytest_argv_carries_no_flag_that_postdates_the_pinned_pytest():
    """The environments install period-correct pytest; `--no-header` arrived in 6.0 and
    makes 5.x exit with a usage error, which reads as a failing test."""
    spec = envspec.UpstreamSpec(repo="sphinx-doc/sphinx", version="3.1", python="3.9",
                                install="python -m pip install -e .[test]",
                                test_cmd="tox --current-env -epy39 -v --")
    r = envspec.native(spec)
    argv = envspec.test_argv(r, "/py", ("tests/test_a.py::test_b",))
    assert argv == ["/py", "-m", "pytest", "-rA", "-p", "no:cacheprovider",
                    "tests/test_a.py::test_b"]
    assert r.extras == ("test",) and r.uninstall == "Sphinx"


def test_workspace_env_points_the_import_at_the_trial(tmp_path):
    """Deviation 4's mechanism: without it `import django` fails from the test runner."""
    env = envspec.workspace_env(str(tmp_path))
    assert env["PYTHONPATH"] == str(tmp_path)
    assert env["LC_ALL"] == "en_US.UTF-8"


# ----------------------------------------------------------------- committed files

def test_upstream_specs_are_committed_and_pinned():
    path = os.path.join(KIT, "envs", "upstream_specs.json")
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob["source"] == envspec.SPEC_SOURCE
    assert blob["dataset"] == dataset.DATASET_ID
    recipes = {k: envspec.native(v)
               for k, v in envspec.load_upstream(path).items()}
    assert len(recipes) == 13
    assert recipes["django__django-3.1"].python == "3.8"
    # Every environment has a committed lockfile naming its resolution date.
    for key in recipes:
        lock = os.path.join(KIT, "envs", f"{key}.lock")
        with open(lock, encoding="utf-8") as fh:
            head = fh.read(400)
        assert "resolved as of" in head, key
