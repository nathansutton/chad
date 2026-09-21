"""A trial's working directory, and the one command that decides whether it passed.

`materialize()` copies an exercise into a fresh directory with everything that could leak
the answer left behind, and writes `run-tests.sh` — the exact command the verifier runs.
The model is told to run that same script, so what it sees while iterating and what it is
scored on cannot drift apart.

`verify()` restores every harness-owned file from upstream first (the tests, the script,
the build files), then runs the script. A model that "fixed" a test, or edited the script
to `exit 0`, is scored on the real ones.

Upstream ships most tests switched off (Exercism's one-test-at-a-time teaching flow):
`xtest` in JavaScript, `@Disabled` in Java, `#[ignore]` in Rust, a preprocessor guard in
C++. Every one is switched on here, the same way Aider's harness does it, so a pass means
the whole suite.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from catalog import DATA, Task

SCRIPT = "run-tests.sh"
VERIFY_TIMEOUT_S = 600

# Never copied into a workspace. `.meta` holds the reference solution, `.approaches` and
# `.articles` are prose walkthroughs of it, `.docs` is folded into the prompt (minus its
# hints file). The Gradle wrapper goes too: Java is built with javac, see `_script`.
_LEFT_BEHIND = frozenset({".meta", ".approaches", ".articles", ".docs", "gradle", "gradlew",
                          "gradlew.bat", "build.gradle", "node_modules", "target", "build"})

JARS = os.path.join(DATA, "jars")
NODE_MODULES = os.path.join(DATA, "node", "node_modules")

_JS_SKIPS = re.compile(r"\b(?:xtest|xit|test\.skip|it\.skip)\(")
_JS_SKIPPED_SUITES = re.compile(r"\b(?:xdescribe|describe\.skip)\(")
_JAVA_DISABLED = re.compile(r"^[ \t]*@Disabled(?:\([^)\n]*\))?[ \t]*\n", re.M)


@dataclass(frozen=True)
class Verdict:
    passed: bool
    timed_out: bool
    seconds: float
    tail: str                    # last lines of the test output, for the trial record


def _script(task: Task) -> str:
    """The test command for `task`, as a POSIX sh script."""
    body = {
        "python": f"exec python3 -m pytest -q {' '.join(_tests_of(task, '.py'))}",
        "go": "exec go test ./...",
        "rust": "exec cargo test --offline -- --include-ignored",
        "javascript": f"exec ./node_modules/.bin/jest --ci {' '.join(_tests_of(task, '.js'))}",
        # javac and the JUnit console launcher rather than the exercise's Gradle wrapper:
        # the wrapper pins a Gradle that refuses current JDKs, downloads 130 MB on first
        # use, and parks a daemon in RAM the model needs.
        "java": (
            "rm -rf build/classes && mkdir -p build/classes\n"
            f"javac -nowarn -d build/classes -cp '{JARS}/*' $(find src -name '*.java') || exit 1\n"
            f"exec java -jar '{JARS}/junit-platform-console-standalone.jar' execute "
            f"--class-path \"build/classes:{JARS}/assertj-core.jar:{JARS}/byte-buddy.jar\" "
            "--scan-class-path --fail-if-no-tests --disable-banner --details=summary"),
        # The exercise's CMakeLists runs the test binary as part of the build.
        "cpp": ("cmake -S . -B build -DEXERCISM_RUN_ALL_TESTS=1 >/dev/null || exit 1\n"
                "exec cmake --build build"),
    }[task.language]
    return f"#!/bin/sh\n# Runs the full test suite. Owned by the benchmark: edits are discarded.\n{body}\n"


def _tests_of(task: Task, suffix: str) -> list[str]:
    return [t for t in task.tests if t.endswith(suffix)]


def _enable_all_tests(language: str, text: str) -> str:
    if language == "javascript":
        return _JS_SKIPPED_SUITES.sub("describe(", _JS_SKIPS.sub("test(", text))
    if language == "java":
        return _JAVA_DISABLED.sub("", text)
    return text


def _owned_files(task: Task) -> list[str]:
    """Everything the harness restores before verifying: tests, their helpers, and any
    top-level build file that is not a solution file (go.mod, CMakeLists.txt, …)."""
    owned = list(task.tests)
    for name in sorted(os.listdir(task.source)):
        path = os.path.join(task.source, name)
        if (os.path.isfile(path) and name not in _LEFT_BEHIND
                and name not in task.solution and name not in owned):
            owned.append(name)
    if task.language == "cpp":
        owned += [os.path.join("test", n)
                  for n in sorted(os.listdir(os.path.join(task.source, "test")))]
    return owned


def _install_owned(task: Task, ws: str) -> None:
    for rel in _owned_files(task):
        dest = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(os.path.join(task.source, rel), encoding="utf-8") as f:
            text = f.read()
        with open(dest, "w", encoding="utf-8") as f:
            f.write(_enable_all_tests(task.language, text) if rel in task.tests else text)
    script = os.path.join(ws, SCRIPT)
    with open(script, "w", encoding="utf-8") as f:
        f.write(_script(task))
    os.chmod(script, 0o755)
    if task.language == "javascript":
        link = os.path.join(ws, "node_modules")
        if os.path.islink(link) or os.path.exists(link):
            if os.path.islink(link):
                os.unlink(link)
            else:
                shutil.rmtree(link)
        os.symlink(NODE_MODULES, link)


def materialize(task: Task, ws: str) -> None:
    """Build a fresh workspace for `task` at `ws` (removed first if it exists)."""
    shutil.rmtree(ws, ignore_errors=True)
    shutil.copytree(task.source, ws, ignore=lambda _dir, names: [n for n in names
                                                                 if n in _LEFT_BEHIND])
    _install_owned(task, ws)


def apply_gold(task: Task, ws: str) -> None:
    """Overwrite the stub with upstream's reference solution (gold gate only).

    Example files pair with solution files by extension, with two track conventions:
    Java keeps its reference classes under `.meta/src/reference/java/`, and Rust may ship
    a `Cargo-example.toml` carrying the dependencies the reference needs."""
    for rel in task.example:
        name = os.path.basename(rel)
        if "/reference/java/" in rel:
            dest = os.path.join("src", "main", "java", name)
        elif name == "Cargo-example.toml":
            continue                       # handled below, listed or not
        else:
            ext = os.path.splitext(name)[1]
            matches = [s for s in task.solution if s.endswith(ext)]
            if not matches:
                raise ValueError(f"{task.name}: no solution file for example {rel}")
            dest = matches[0]
        shutil.copyfile(os.path.join(task.source, rel), os.path.join(ws, dest))
    # Not always listed under `example`, but the reference will not link without it.
    gold_manifest = os.path.join(task.source, ".meta", "Cargo-example.toml")
    if os.path.exists(gold_manifest):
        shutil.copyfile(gold_manifest, os.path.join(ws, "Cargo.toml"))


def trial_env() -> dict[str, str]:
    """The environment tests run in: this interpreter's directory first on PATH, so
    `python3 -m pytest` resolves to the environment that has pytest installed."""
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["CARGO_NET_OFFLINE"] = "true"
    return env


def verify(task: Task, ws: str, timeout_s: int = VERIFY_TIMEOUT_S) -> Verdict:
    """Restore the harness-owned files, run the suite, and report. Exit status 0 is the
    only pass: no output parsing, so a test runner's wording cannot move a score."""
    _install_owned(task, ws)
    t0 = time.time()
    try:
        out = subprocess.run(["sh", SCRIPT], cwd=ws, env=trial_env(), capture_output=True,
                             text=True, errors="replace", timeout=timeout_s, check=False,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return Verdict(False, True, time.time() - t0, "")
    tail = "\n".join((out.stdout + "\n" + out.stderr).strip().splitlines()[-15:])
    return Verdict(out.returncode == 0, False, time.time() - t0, tail)
