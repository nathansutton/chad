"""One-time setup: everything a trial needs from the network, fetched before any trial.

    uv run python benchmarks/polyglot/prepare.py

Trials run offline — a test suite that reaches the network mid-run is a flaky benchmark —
so the four things that do need it are pulled here, each pinned:

* the exercises, at `catalog.UPSTREAM_COMMIT`;
* three test jars for Java (JUnit console launcher, AssertJ and its one dependency),
  checked against the SHA-256 recorded below;
* jest and babel for JavaScript, installed once from the committed lockfile and
  symlinked into every workspace;
* the few crates the Rust exercises depend on, into cargo's own cache.

Also reports which toolchains are missing, so the first sign of an absent compiler is
here and not 40 tasks into a run as a wall of false failures.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog  # noqa: E402
import workspace  # noqa: E402

_MAVEN = "https://repo1.maven.org/maven2/"
# local name -> (path under Maven Central, sha256)
JAR_PINS = {
    "junit-platform-console-standalone.jar": (
        "org/junit/platform/junit-platform-console-standalone/1.10.0/"
        "junit-platform-console-standalone-1.10.0.jar",
        "8192d377bdb53969273ca3805af0eef3b864690f7818300c0d39ac7758da39a0"),
    "assertj-core.jar": (
        "org/assertj/assertj-core/3.25.1/assertj-core-3.25.1.jar",
        "620da4e4de03108c2f412ec17a7ed44b0112dc65e181bd0732dc96fa3b65356f"),
    "byte-buddy.jar": (
        "net/bytebuddy/byte-buddy/1.14.11/byte-buddy-1.14.11.jar",
        "62ae28187ed2b062813da6a9d567bfee733c341582699b62dd980230729a0313"),
}

TOOLCHAINS = {
    "python": ["python3"], "go": ["go"], "rust": ["cargo"], "javascript": ["node", "npm"],
    "java": ["java", "javac"], "cpp": ["cmake", "c++"],
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_jars() -> None:
    os.makedirs(workspace.JARS, exist_ok=True)
    for name, (path, want) in JAR_PINS.items():
        dest = os.path.join(workspace.JARS, name)
        if os.path.exists(dest) and _sha256(dest) == want:
            continue
        with urllib.request.urlopen(_MAVEN + path) as response, open(dest + ".part", "wb") as f:
            shutil.copyfileobj(response, f)
        got = _sha256(dest + ".part")
        if got != want:
            os.remove(dest + ".part")
            raise SystemExit(f"{name}: sha256 {got} does not match the pin {want}")
        os.replace(dest + ".part", dest)


def install_node() -> None:
    node_dir = os.path.dirname(workspace.NODE_MODULES)
    os.makedirs(node_dir, exist_ok=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copyfile(os.path.join(catalog.ROOT, "node", name), os.path.join(node_dir, name))
    subprocess.run(["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
                   cwd=node_dir, check=True)


def fetch_crates(tasks: list[catalog.Task]) -> None:
    for task in tasks:
        with open(os.path.join(task.source, "Cargo.toml"), encoding="utf-8") as f:
            manifest = f.read()
        gold_manifest = os.path.join(task.source, ".meta", "Cargo-example.toml")
        if os.path.exists(gold_manifest):
            with open(gold_manifest, encoding="utf-8") as f:
                manifest += f.read()
        if "[dependencies]" not in manifest or not manifest.split("[dependencies]")[1].strip():
            continue
        # `cargo fetch` needs a lockfile target; resolve in a scratch copy so the pinned
        # upstream checkout stays byte-identical.
        scratch = os.path.join(catalog.DATA, "cargo-fetch", task.slug)
        workspace.materialize(task, scratch)
        if os.path.exists(gold_manifest):
            shutil.copyfile(gold_manifest, os.path.join(scratch, "Cargo.toml"))
        subprocess.run(["cargo", "fetch"], cwd=scratch, check=True,
                       env={**os.environ, "CARGO_NET_OFFLINE": "false"})
    shutil.rmtree(os.path.join(catalog.DATA, "cargo-fetch"), ignore_errors=True)


def missing_toolchains() -> dict[str, list[str]]:
    return {lang: [t for t in tools if shutil.which(t) is None]
            for lang, tools in TOOLCHAINS.items()
            if any(shutil.which(t) is None for t in tools)}


def main() -> int:
    missing = missing_toolchains()
    for lang, tools in missing.items():
        print(f"MISSING toolchain for {lang}: {', '.join(tools)} — its tasks will be skipped")
    print("fetching exercises …")
    catalog.fetch()
    if "java" not in missing:
        print("fetching test jars …")
        fetch_jars()
    if "javascript" not in missing:
        print("installing jest + babel …")
        install_node()
    if "rust" not in missing:
        print("fetching crates …")
        fetch_crates(catalog.load_tasks(["rust"]))
    counts = {lang: len(catalog.load_tasks([lang])) for lang in catalog.LANGUAGES}
    print(f"ready: {sum(counts.values())} exercises {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
