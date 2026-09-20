#!/usr/bin/env python3
"""Could chad swap ripgrep for tgrep without telling the model?

chad teaches `rg` and nothing else, because routing follows the model's trained prior and
steering text does not move it (docs/design.md). So the only integration shape worth
testing is a transparent one: rewrite `rg ...` to `tgrep ...` inside the bash tool and
leave the prompt alone. That is only safe if the rewrite returns the same answer.

This replays every single-segment `rg` invocation chad's model actually emitted, in the
repo the session ran in, under both binaries, and sorts the outcome into:

    byte-identical      safe
    reordered           same match set, different file order — safe to read, but any
                        `| head -20` downstream now truncates a different set
    match set differs   a wrong answer, silently
    errored             tgrep rejected a command rg accepted

It also probes the two hazards a replay of past commands cannot reach on its own: the
64 MiB default size cap, and patterns that collide with tgrep's subcommand names.

Run (after building an index per corpus):
    uv run python benchmarks/tgrep/fidelity.py
"""

import glob
import json
import os
import re
import shlex
import subprocess

CALL = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.S)
SEGMENT = re.compile(r"\|\||&&|[|;&\n]")
SESSIONS = os.path.expanduser("~/.chad/sessions/*/*.json")

# Corpora with a committed index, keyed by the cwd a session recorded. A trace from any
# other directory is skipped rather than guessed at: replaying a command against the
# wrong tree would manufacture divergences that never happened.
CORPORA = {
    "/Users/nate/repo/ky": "/tmp/tgrep-idx/ky",
    "/Users/nate/repo/chad": "/tmp/tgrep-idx/chad",
}
# tgrep's own subcommands. A pattern spelled like one is read as a subcommand unless `--`
# precedes it, which no `rg` command line written by a model contains.
SUBCOMMANDS = ("search", "index", "serve", "status", "count-files", "help")


def run(argv: list[str], cwd: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True,
                              stdin=subprocess.DEVNULL, timeout=30)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, b"", b"timed out")


def rg_segments() -> list[tuple[str, str]]:
    """Distinct (cwd, single-segment rg command) pairs from the recorded sessions."""
    seen: set[tuple[str, str]] = set()
    found = []
    for path in sorted(glob.glob(SESSIONS)):
        if path.endswith("index.json"):
            continue
        try:
            with open(path) as fh:
                trace = json.load(fh)
        except (OSError, ValueError):
            continue
        cwd = trace.get("cwd", "")
        if cwd not in CORPORA or not os.path.isdir(cwd):
            continue
        for message in trace.get("messages") or []:
            if message.get("role") != "assistant":
                continue
            for name, body in CALL.findall(message.get("content") or ""):
                if name != "bash":
                    continue
                for key, value in PARAM.findall(body):
                    if key != "command":
                        continue
                    for segment in SEGMENT.split(value):
                        segment = segment.strip()
                        if segment.startswith("rg ") and (cwd, segment) not in seen:
                            seen.add((cwd, segment))
                            found.append((cwd, segment))
    return found


def replay() -> None:
    pairs = rg_segments()
    print(f"{len(pairs)} distinct rg commands replayed from recorded sessions\n")
    same = reordered = differs = errored = unparsed = 0
    notes = []
    for cwd, segment in pairs:
        try:
            argv = shlex.split(segment)
        except ValueError:
            unparsed += 1
            continue
        got_rg = run(argv, cwd)
        got_tg = run(["tgrep", "--index-path", CORPORA[cwd]] + argv[1:], cwd)
        if got_tg.returncode == 2 and got_rg.returncode != 2:
            errored += 1
            notes.append(("ERROR", segment,
                          got_tg.stderr.decode(errors="replace").strip()[:110]))
        elif got_rg.stdout == got_tg.stdout:
            same += 1
        elif sorted(got_rg.stdout.splitlines()) == sorted(got_tg.stdout.splitlines()):
            reordered += 1
        else:
            differs += 1
            rows_rg = set(got_rg.stdout.splitlines())
            rows_tg = set(got_tg.stdout.splitlines())
            notes.append(("DIFFERS", segment,
                          f"rg-only {len(rows_rg - rows_tg)} / "
                          f"tgrep-only {len(rows_tg - rows_rg)}"))
    total = same + reordered + differs + errored
    print(f"byte-identical    : {same:4}")
    print(f"reordered         : {reordered:4}")
    print(f"match set differs : {differs:4}")
    print(f"tgrep errored     : {errored:4}")
    print(f"unparseable       : {unparsed:4}  (shell quoting shlex cannot split)")
    if total:
        print(f"\nsame match set: {100 * (same + reordered) / total:.1f}%   "
              f"byte-identical: {100 * same / total:.1f}%")
    for kind, segment, note in notes[:10]:
        print(f"  {kind:8} {segment[:76]:76} {note}")


def probe_size_cap(workdir: str) -> None:
    """tgrep skips files over 64 MiB by default; rg has no limit. The match is dropped
    with no warning and exit code 0, which reads as `not present`."""
    root = os.path.join(workdir, "sizecap")
    os.makedirs(root, exist_ok=True)
    subprocess.run(["git", "init", "-q", "."], cwd=root, capture_output=True)
    with open(os.path.join(root, "big.log"), "w") as fh:
        fh.write("x" * 70_000_000 + "\nNEEDLE\n")
    with open(os.path.join(root, "small.txt"), "w") as fh:
        fh.write("NEEDLE\n")
    index = os.path.join(workdir, "sizecap-idx")
    subprocess.run(["tgrep", "index", root, "--index-path", index], capture_output=True)
    hits_rg = run(["rg", "-n", "--", "NEEDLE", "."], root).stdout.decode()
    hits_tg = run(["tgrep", "--index-path", index, "-n", "--", "NEEDLE", "."],
                  root).stdout.decode()
    print("\n64 MiB default cap")
    print(f"  rg    found {len(hits_rg.splitlines())} file(s): "
          f"{sorted(h.split(':')[0] for h in hits_rg.splitlines())}")
    print(f"  tgrep found {len(hits_tg.splitlines())} file(s): "
          f"{sorted(h.split(':')[0] for h in hits_tg.splitlines())}")


def probe_subcommand_collision(cwd: str, index: str) -> None:
    """A pattern spelled like a tgrep subcommand is read as one."""
    print("\npattern colliding with a tgrep subcommand (one file, `-n <pattern>`)")
    for pattern in SUBCOMMANDS:
        if pattern == "serve":
            print(f"  {pattern!r:14} not run: `tgrep -n serve <file>` starts a server "
                  "and never returns")
            continue
        n_rg = len(run(["rg", "-n", pattern, "src/chad/tools.py"], cwd).stdout.splitlines())
        bare = run(["tgrep", "--index-path", index, "-n", pattern,
                    "src/chad/tools.py"], cwd)
        n_bare = len(bare.stdout.splitlines())
        n_dash = len(run(["tgrep", "--index-path", index, "-n", "--", pattern,
                          "src/chad/tools.py"], cwd).stdout.splitlines())
        flag = "  <- wrong" if n_bare != n_rg else ""
        print(f"  {pattern!r:14} rg={n_rg:4}  tgrep={n_bare:4}  "
              f"tgrep --={n_dash:4}{flag}")


def probe_staleness(workdir: str) -> None:
    """An on-disk index does not see the agent's own edit. chad edits constantly."""
    root = os.path.join(workdir, "stale")
    os.makedirs(root, exist_ok=True)
    subprocess.run(["git", "init", "-q", "."], cwd=root, capture_output=True)
    with open(os.path.join(root, "mod.py"), "w") as fh:
        fh.write("def alpha():\n    pass\n")
    index = os.path.join(workdir, "stale-idx")
    subprocess.run(["tgrep", "index", root, "--index-path", index], capture_output=True)
    with open(os.path.join(root, "mod.py"), "w") as fh:
        fh.write("def alpha():\n    return FRESHLY_WRITTEN\n")
    hits_rg = run(["rg", "-n", "--", "FRESHLY_WRITTEN", "."], root).stdout.decode().strip()
    hits_tg = run(["tgrep", "--index-path", index, "-n", "--", "FRESHLY_WRITTEN", "."],
                  root).stdout.decode().strip()
    print("\ngrep immediately after an edit, on-disk index, no daemon")
    print(f"  rg    -> {hits_rg or '(no match)'}")
    print(f"  tgrep -> {hits_tg or '(no match)'}")


def main() -> None:
    workdir = os.environ.get("TMPDIR", "/tmp").rstrip("/") + "/tgrep-fidelity"
    os.makedirs(workdir, exist_ok=True)
    replay()
    probe_size_cap(workdir)
    probe_staleness(workdir)
    chad = "/Users/nate/repo/chad"
    if os.path.isdir(chad) and os.path.isdir(CORPORA[chad]):
        probe_subcommand_collision(chad, CORPORA[chad])


main()
