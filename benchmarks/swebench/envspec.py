"""SWE-bench's own environment recipe, translated to a native arm64 workbench.

The agent solves on this Mac, on a real checkout, with a real interpreter — that is the
whole premise of the measurement, and it is why the trials do not run in the grading
container. But the grade must still be the official one, so the environment the agent
works in has to be as close to the container as an arm64 host can get, and every place
it is NOT close has to be written down here rather than discovered in the results.

Upstream ships one recipe per (repo, version) in `MAP_REPO_VERSION_TO_SPECS`: an
interpreter, a list of pip packages, a list of shell commands that pin transitive
dependencies by editing `setup.py`, and an install command. `prepare.py specs` dumps
those verbatim into `envs/upstream_specs.json` (pinned to one swebench release, recorded
in the file). This module translates them and names the differences.

THE DEVIATIONS, IN FULL
-----------------------
Each has a reason, and each is visible in the kit before any trial is scored:

1. **django 3.1 runs on Python 3.8, not upstream's 3.6.** There is no arm64 build of
   CPython 3.6 — the platform predates Apple Silicon. 3.8 is the oldest interpreter that
   both exists here and still runs django 3.1's test suite. sphinx needs no deviation:
   upstream already asks for 3.9, which builds natively.
2. **`sed -i` is applied in-process, not by sed.** Upstream's commands are GNU sed;
   BSD sed on macOS reads `-i` differently and would write backup files. They are parsed
   and applied as text substitutions instead, and an upstream command in any shape this
   module does not recognise raises — a silently skipped dependency pin would show up
   much later as a mysterious version conflict.
3. **`apt-get` steps cannot run.** sphinx 7.2 wants graphviz from a Debian repository.
   It is recorded as unreachable; the workbench validation gate reports whichever tests
   need it rather than the env pretending to have it.
4. **The package under test is never installed; `PYTHONPATH` stands in for the editable
   install.** Upstream does `pip install -e .`, which binds one environment to one
   checkout; here a single environment is shared by every trial of that version, and
   each trial has its own throwaway workspace. So the dependencies are resolved exactly
   as upstream resolves them (including the editable install, so setuptools does the
   work), the package's own link is then removed, and every command run in a trial gets
   `PYTHONPATH` set to that trial's workspace root.
   Relying on the working directory instead does not work and is worth stating, because
   it looks like it should: django's `tests/runtests.py` is executed as a script, so
   `sys.path[0]` is `tests/` and `import django` fails outright. An agent that had to
   discover that would be solving a packaging puzzle this benchmark does not contain —
   in the official container the package is importable from anywhere, and it must be
   here too, or the measurement is of something else.
5. **`tox --current-env` becomes plain pytest.** That flag exists to make tox run in the
   ambient environment instead of building one; running the ambient pytest directly is
   the same thing with one fewer moving part.
6. **Locale `eval_commands` are dropped.** They are `locale-gen` against `/etc/locale.gen`,
   which is Debian-only. macOS ships UTF-8 locales; the runner exports them directly.
9. **A short, declared list of our own patches** (`EXTRA_PATCHES`), applied the same way
   upstream's are, where a period-correct source distribution no longer compiles here.

7. **Dependencies resolve as of the environment-setup commit's date.** This one is not
   cosmetic, and without it nothing in the sphinx half of the benchmark runs at all.
   sphinx 3.x asks for `docutils>=0.12` with no upper bound, so a resolution performed
   today installs docutils 0.23 into a 2020 project and the test suite dies during
   collection — `docutils.utils.roman` has not existed since 0.21. The published images
   do not have this problem because their resolution was frozen the day they were built.
   Ours is frozen too, at the date of the commit the environment is defined by, via
   `uv pip install --exclude-newer`. The date is per environment, derived from the data
   rather than chosen, and recorded in every lockfile.
   Two carve-outs: upstream's own `pip_packages` are exact, deliberately modern pins
   (`tox==4.16.0` post-dates every repo here) and are installed unconstrained; and the
   build backend is the environment's own modern setuptools (`--no-build-isolation`),
   because a 2020 setuptools cannot build an editable install at all.
8. **setuptools is installed explicitly.** A conda environment has it; a modern virtual
   environment does not, and sphinx 3.x imports `pkg_resources` at module scope.

None of these touch what the agent sees. Both arms get the same workbench, and the grade
always comes from the official container.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex

# The swebench release whose constants this kit reads. 5.x moved the environment specs
# out of the package and behind a network fetch; 4.0.5 is the last release that carries
# `MAP_REPO_VERSION_TO_SPECS` in-process, which is what makes the dump auditable.
SPEC_SOURCE = "swebench==4.0.5"

# Deviation 1. Keyed by the upstream interpreter, not by repo, so a version added later
# inherits the same rule.
PYTHON_SUBSTITUTIONS = {"3.6": "3.8"}

DJANGO_REQUIREMENTS = "tests/requirements/py3.txt"

# Deviation 8: present in a conda environment, absent from a modern venv, and imported
# at module scope by sphinx 3.x. Capped below the release that drops `pkg_resources`.
BOOTSTRAP_PACKAGES = ("setuptools<81", "wheel")

# Deviation 9: our own patches, applied after upstream's and in the same way — to the
# template checkout only, never to a trial workspace. Each one exists because a
# period-correct source distribution no longer compiles on this architecture, and each
# is the narrowest possible edit with upstream's own later behaviour as its warrant.
# A patch whose text is absent is a no-op, so each is self-limiting to the versions
# that need it.
EXTRA_PATCHES = (
    # sphinx 3.1-3.3 declare typed_ast unconditionally, with the comment "# for
    # py35-37"; 3.4 replaced it with exactly this environment marker. Version 1.4.1 is
    # a C extension that does not build against a current clang, and on Python 3.9 the
    # dependency is dead weight by upstream's own account.
    ("setup.py", "'typed_ast',", "\"typed_ast; python_version < '3.8'\",",
     "sphinx 3.4 added this marker; 1.4.1 does not compile on arm64 and is unused here"),
)

# Deviation 6, as environment rather than as shell.
LOCALE_ENV = {"LANG": "en_US.UTF-8", "LANGUAGE": "en_US:en", "LC_ALL": "en_US.UTF-8"}


def workspace_env(workspace: str) -> dict[str, str]:
    """The environment every command in a trial runs under — the agent's and the
    validation gate's alike, so that what the gate proves is what the agent gets."""
    return {"PYTHONPATH": os.path.abspath(workspace), **LOCALE_ENV}

_SED_RE = re.compile(r"^s(?P<d>.)(?P<pat>.*?)(?P=d)(?P<repl>.*?)(?P=d)$")


@dataclasses.dataclass(frozen=True)
class FilePatch:
    """One upstream `sed -i`, parsed. `guard` carries the `grep -q` of the conditional
    form: the substitution applies only when the guard pattern is present (or absent,
    for the `||` branch)."""
    path: str
    pattern: str
    replacement: str
    guard: str = ""
    guard_present: bool = True

    def applies(self, text: str) -> bool:
        if not self.guard:
            return True
        return (self.guard in text) is self.guard_present

    def apply(self, text: str) -> str:
        if not self.applies(text):
            return text
        # sed substitutes at most once per line without the `g` flag; none of the
        # upstream scripts use flags, so the per-line count of 1 is the faithful read.
        repl = self.replacement.replace("\\", "\\\\")
        return "\n".join(re.sub(self.pattern, repl, line, count=1)
                         for line in text.split("\n"))


@dataclasses.dataclass(frozen=True)
class UpstreamSpec:
    repo: str
    version: str
    python: str
    install: str
    test_cmd: str
    packages: str = ""
    pip_packages: tuple[str, ...] = ()
    pre_install: tuple[str, ...] = ()
    eval_commands: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class NativeRecipe:
    """How to build this environment here, and what it is missing."""
    env_key: str
    repo: str
    version: str
    python: str
    python_upstream: str
    pip_packages: tuple[str, ...]
    requirements_file: str
    extras: tuple[str, ...]
    patches: tuple[FilePatch, ...]
    unreachable: tuple[str, ...]
    runner: str            # "django" (tests/runtests.py) or "pytest"
    uninstall: str         # distribution to unlink after the editable install


def _parse_sed(cmd: str) -> FilePatch:
    """`sed -i 's/A/B/' FILE` -> a FilePatch. Anything else is an error, by design."""
    parts = shlex.split(cmd)
    if len(parts) != 4 or parts[0] != "sed" or parts[1] != "-i":
        raise ValueError(f"unrecognised upstream pre_install command: {cmd!r}")
    m = _SED_RE.match(parts[2])
    if not m:
        raise ValueError(f"only plain s/// substitutions are supported: {cmd!r}")
    repl = m.group("repl")
    if re.search(r"(?<!\\)&", repl):
        # `&` in a sed replacement means the whole match; no upstream recipe uses it,
        # and guessing would corrupt a pin.
        raise ValueError(f"sed back-reference '&' not supported: {cmd!r}")
    return FilePatch(path=parts[3], pattern=m.group("pat"), replacement=repl)


def _parse_conditional(cmd: str) -> tuple[FilePatch, FilePatch]:
    """`grep -q 'P' F && sed -i 'S1' F || sed -i 'S2' F` -> the two guarded patches.

    Upstream uses this where a pin's exact spelling differs between two versions that
    share a recipe: tighten the bound if the loose form is there, otherwise cap it.
    """
    head, _, rest = cmd.partition("&&")
    yes, _, no = rest.partition("||")
    g = shlex.split(head.strip())
    if len(g) != 4 or g[0] != "grep" or g[1] != "-q":
        raise ValueError(f"unrecognised guard in pre_install command: {cmd!r}")
    a, b = _parse_sed(yes.strip()), _parse_sed(no.strip())
    return (dataclasses.replace(a, guard=g[2], guard_present=True),
            dataclasses.replace(b, guard=g[2], guard_present=False))


def translate_pre_install(cmds: tuple[str, ...]) -> tuple[tuple[FilePatch, ...],
                                                          tuple[str, ...]]:
    """Split upstream's pre-install shell into patches we apply and steps we cannot."""
    patches: list[FilePatch] = []
    unreachable: list[str] = []
    for cmd in cmds:
        c = cmd.strip()
        if c.startswith("apt-get") or " apt-get " in c:
            unreachable.append(c)          # deviation 3
        elif c.startswith("grep "):
            patches.extend(_parse_conditional(c))
        else:
            patches.append(_parse_sed(c))
    return tuple(patches), tuple(unreachable)


def native(spec: UpstreamSpec) -> NativeRecipe:
    """The arm64 recipe for one upstream spec."""
    patches, unreachable = translate_pre_install(spec.pre_install)
    patches = patches + tuple(FilePatch(path=path, pattern=re.escape(pat),
                                        replacement=repl)
                              for path, pat, repl, _reason in EXTRA_PATCHES)
    django = spec.repo == "django/django"
    return NativeRecipe(
        env_key=f"{spec.repo.replace('/', '__')}-{spec.version}",
        repo=spec.repo,
        version=spec.version,
        python=PYTHON_SUBSTITUTIONS.get(spec.python, spec.python),
        python_upstream=spec.python,
        pip_packages=spec.pip_packages,
        # Upstream's `packages: "requirements.txt"` means "install the repo's own test
        # requirements"; for django that file is at a fixed path.
        requirements_file=DJANGO_REQUIREMENTS if spec.packages == "requirements.txt" else "",
        extras=("test",) if "[test]" in spec.install else (),
        patches=patches,
        unreachable=unreachable,
        # django's runner is its own script; everything else upstream drives through tox
        # is plain pytest underneath (deviation 5).
        runner="django" if django else "pytest",
        uninstall="Django" if django else "Sphinx",
    )


def load_upstream(path: str) -> dict[str, UpstreamSpec]:
    """Read `envs/upstream_specs.json` -> {env_key: UpstreamSpec}."""
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    out: dict[str, UpstreamSpec] = {}
    for repo, versions in blob["specs"].items():
        for version, spec in versions.items():
            key = f"{repo.replace('/', '__')}-{version}"
            out[key] = UpstreamSpec(
                repo=repo, version=version, python=str(spec["python"]),
                install=spec["install"], test_cmd=spec["test_cmd"],
                packages=spec.get("packages", ""),
                pip_packages=tuple(spec.get("pip_packages", ())),
                pre_install=tuple(spec.get("pre_install", ())),
                eval_commands=tuple(spec.get("eval_commands", ())),
            )
    return out


def recipes(root: str) -> dict[str, NativeRecipe]:
    path = os.path.join(root, "envs", "upstream_specs.json")
    return {k: native(v) for k, v in load_upstream(path).items()}


_DJANGO_ID = re.compile(r"^(\S+)\s+\((.+?)\)\s*$")
_DOTTED = re.compile(r"^[\w.]+$")


def django_label(test_id: str) -> str:
    """`test_x (a.b.C)` -> `a.b.C.test_x`, the label `tests/runtests.py` takes, or `""`
    for an entry that is not a test name at all.

    django's own output names a test the other way round, and that is the spelling
    SWE-bench records. It also prints each test's docstring on the next line at
    `--verbosity 2`, and upstream's log parser captured some of those as entries in
    their own right — `PASS_TO_PASS` for django__django-12143 contains
    "{% get_admin_log %} works without specifying a user". The official grader never
    notices, because it matches these strings against the log of a whole-suite run;
    running tests by label does notice, because `runtests.py` tries to import the
    sentence as a module. Such an entry is dropped, and the drop is counted rather than
    hidden: the test it documents is in the list too, immediately before it.
    """
    m = _DJANGO_ID.match(test_id)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    return test_id if _DOTTED.match(test_id) else ""


def runnable(recipe: NativeRecipe, tests: tuple[str, ...]) -> tuple[str, ...]:
    """The entries of `tests` this workbench can actually run by name (see
    `django_label`). pytest node ids always survive; some django entries do not."""
    if recipe.runner != "django":
        return tests
    return tuple(t for t in tests if django_label(t))


def test_argv(recipe: NativeRecipe, python: str, tests: tuple[str, ...]) -> list[str]:
    """The command that runs exactly `tests` in this environment, from the workspace root.

    Used by the workbench validation gate only. The agent is never told a test command;
    finding out how to run the suite is part of the task, and handing one arm a command
    the other had to discover would be handing it the answer.
    """
    if recipe.runner == "django":
        return [python, "tests/runtests.py", "--verbosity", "2",
                "--settings=test_sqlite", "--parallel", "1",
                *[label for label in (django_label(t) for t in tests) if label]]
    # `-rA` is upstream's own flag (its pre_install seds it into tox.ini). Nothing else
    # is added beyond disabling the cache plugin, which would otherwise write a
    # `.pytest_cache` into a workspace that is supposed to end the trial pristine except
    # for the fix. In particular not `--no-header`: the date-pinned environments install
    # period-correct pytest, and that flag did not exist before pytest 6.
    return [python, "-m", "pytest", "-rA", "-p", "no:cacheprovider", *tests]
