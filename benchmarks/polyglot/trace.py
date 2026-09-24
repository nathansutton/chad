"""Read one trial's ATIF trajectory as a step table: where the tokens and the seconds went.

    python benchmarks/polyglot/trace.py _runs/baseline/trajectories/go/book-store.rep1.json
    python benchmarks/polyglot/trace.py <file> --step 4        # that step's full text
    python benchmarks/polyglot/trace.py _runs/baseline         # the whole run in one profile
    python benchmarks/polyglot/trace.py <dir> --counts > x.jsonl   # per-trial tool counts, and
    python benchmarks/polyglot/trace.py x.jsonl                # the tool mix back from them

`hit` is the share of the step's prompt served from the KV cache (a `*` marks a step that
re-prefilled after compaction); `t/s` is decode speed. `acc` is the share of drafted tokens accepted; `a/rT` and `a/rA` are tokens accepted per
verify round while reasoning and while acting (after `</think>`) — the per-round yield is
what decides whether a wide verify paid, and the two phases accept very differently.
`--step` adds the histogram behind them: per proposed width, how many rounds accepted
exactly 0, 1, 2, … tokens.

Works on a trial that is still running — the runner rewrites the file as it goes, and the
last row is then the step being generated (`finish` reads `...`). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import JsonValue, is_array, is_number, is_object, is_text  # noqa: E402


def _obj(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if is_object(value) else {}


def _num(value: JsonValue) -> float:
    return float(value) if is_number(value) else 0.0


def _text(value: JsonValue) -> str:
    return value if is_text(value) else ""


def _items(value: JsonValue) -> Sequence[JsonValue]:
    return value if is_array(value) else ()


def _one_line(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def _action(step: Mapping[str, JsonValue]) -> str:
    """What the step DID: its tool calls, else its visible message, else its reasoning."""
    calls = _items(step.get("tool_calls"))
    if calls:
        parts = []
        for call in calls:
            args = _obj(_obj(call).get("arguments"))
            head = next((_text(v) for v in args.values() if is_text(v)), "")
            parts.append(f"{_text(_obj(call).get('function_name'))}: {head}")
        return " | ".join(parts)
    return _text(step.get("message")) or "(think) " + _text(step.get("reasoning_content"))


def _yield(extra: Mapping[str, JsonValue], phase: str) -> str:
    """Tokens accepted per verify round in one phase, or `-` when it drafted nothing."""
    block = _obj(_obj(extra.get("draft")).get(phase))
    return f"{_num(block.get('accepted_per_round')):.2f}" if _num(block.get("rounds")) else "-"


def _histogram(extra: Mapping[str, JsonValue]) -> str:
    lines = []
    for phase in ("think", "act"):
        block = _obj(_obj(extra.get("draft")).get(phase))
        if not _num(block.get("rounds")):
            continue
        lines.append(f"{phase}: {int(_num(block.get('rounds')))} rounds, "
                     f"{_num(block.get('accepted_per_round')):.2f} accepted/round, "
                     f"first token missed in {_num(block.get('first_token_miss')):.0%}")
        for width, row in _obj(block.get("by_width")).items():
            counts = " ".join(f"{int(_num(n)):>5}" for n in _items(row))
            lines.append(f"  k={width:>2} accepted 0..k: {counts}")
    return "\n".join(lines)


def agent_steps(doc: JsonValue) -> list[Mapping[str, JsonValue]]:
    return [_obj(s) for s in _items(_obj(doc).get("steps"))
            if _obj(s).get("source") == "agent"]


def table(doc: JsonValue, width: int) -> str:
    rows = [f"{'step':>4} {'gen':>6} {'think':>6} {'gen_s':>6} {'t/s':>5} {'pre_s':>6} {'ctx':>6} "
            f"{'hit':>5} {'acc':>4} {'a/rT':>5} {'a/rA':>5} {'finish':<11} action"]
    for step in agent_steps(doc):
        metrics = _obj(step.get("metrics"))
        extra = _obj(metrics.get("extra"))
        live = _obj(step.get("extra")).get("in_flight") is True
        gen = _obj(step.get("extra")).get("generated_tokens") if live \
            else metrics.get("completion_tokens")
        proposed = _num(extra.get("draft_proposed"))
        accept = f"{_num(extra.get('draft_accepted')) / proposed:.2f}" if proposed else "-"
        prompt = _num(metrics.get("prompt_tokens"))
        hit = f"{_num(metrics.get('cached_tokens')) / prompt:.2f}" if prompt else "-"
        hit += "*" if extra.get("compacted") is True else ""
        rows.append(
            f"{int(_num(step.get('step_id'))):>4} {int(_num(gen)):>6} "
            f"{int(_num(extra.get('think_tokens'))):>6} {_num(extra.get('gen_s')):>6.0f} "
            f"{_num(extra.get('tok_per_s')):>5.1f} {_num(extra.get('prefill_s')):>6.1f} "
            f"{int(prompt):>6} {hit:>5} {accept:>4} {_yield(extra, 'think'):>5} {_yield(extra, 'act'):>5} "
            f"{'...' if live else _text(extra.get('finish')):<11} "
            f"{_one_line(_action(step), width)}")
    totals = _obj(_obj(_obj(doc).get("final_metrics")).get("extra"))
    if totals:
        rows.append(f"total: think {int(_num(totals.get('total_think_tokens')))} tok, "
                    f"decode {_num(totals.get('total_gen_s')):.0f} s, "
                    f"prefill {_num(totals.get('total_prefill_s')):.0f} s, "
                    f"{_num(totals.get('tok_per_s')):.1f} tok/s, "
                    f"cache hit {_num(totals.get('cache_hit_rate')):.0%}, "
                    f"{int(_num(totals.get('compactions')))} compactions")
        for phase in ("think", "act"):
            block = _obj(totals.get(f"draft_{phase}"))
            if block:
                rows.append(f"draft {phase}: accept {_num(block.get('accept_rate')):.0%}, "
                            f"{_num(block.get('accepted_per_round')):.2f} accepted/round over "
                            f"{int(_num(block.get('rounds')))} rounds")
    return "\n".join(rows)


def full_step(doc: JsonValue, step_id: int) -> str:
    for step in agent_steps(doc):
        if int(_num(step.get("step_id"))) == step_id:
            seen = [_text(_obj(r).get("content"))
                    for r in _items(_obj(step.get("observation")).get("results"))]
            draft = _histogram(_obj(_obj(step.get("metrics")).get("extra")))
            return "\n\n".join(filter(None, [
                "=== drafting ===\n" + draft if draft else "",
                "=== reasoning ===\n" + _text(step.get("reasoning_content")),
                "=== message ===\n" + _text(step.get("message")),
                "=== tool calls ===\n" + json.dumps(step.get("tool_calls"), indent=1),
                "=== results ===\n" + "\n---\n".join(seen)]))
    raise SystemExit(f"no agent step {step_id} in this trajectory")


# A shell command doing a dedicated tool's job: the measure of which route the model takes
# when both are on offer. Only the command that STARTS a pipeline counts, so `pytest | tail`
# and `ls | grep x` are a test run and a listing, not a read and a search.
_PIPELINES = re.compile(r"&&|\|\||;|\n")
_SHELL_SEARCH = re.compile(r"\s*(grep|egrep|rg|ag|fd|find)\s")
_SHELL_READ = re.compile(r"\s*(cat|head|tail|less|nl|sed\s+-n)\s")


def _routes(command: str) -> set[str]:
    heads = [pipeline.split("|", 1)[0] for pipeline in _PIPELINES.split(command)]
    return ({"bash:search"} if any(_SHELL_SEARCH.match(h) for h in heads) else set()) | \
           ({"bash:read"} if any(_SHELL_READ.match(h) for h in heads) else set())


def load_run(run_dir: str) -> list[tuple[str, JsonValue]]:
    """Every ATIF document under `run_dir`, whatever the layout: this kit's
    `trajectories/<language>/`, or a Harbor jobs tree of `<trial>/agent/trajectory.json`.
    A run's `output/` is skipped: it holds harness logs and the proxy's second reading of
    trials that already have a trajectory, which would count those trials twice."""
    docs: list[tuple[str, JsonValue]] = []
    output = os.path.join(os.path.normpath(run_dir), "output")
    for root, _dirs, files in sorted(os.walk(run_dir)):
        if root == output or root.startswith(output + os.sep):
            continue
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                doc: JsonValue = json.load(f)
            if is_array(_obj(doc).get("steps")):
                docs.append((os.path.relpath(path, run_dir), doc))
    return docs


def tool_counts(doc: JsonValue) -> dict[str, int]:
    """Calls per tool in one trajectory, plus `bash:search` and `bash:read`: the bash calls
    whose command runs a search or a file read, counted inside `bash`, not beside it."""
    counts: dict[str, int] = {}
    for step in agent_steps(doc):
        for call in _items(step.get("tool_calls")):
            name = _text(_obj(call).get("function_name"))
            counts[name] = counts.get(name, 0) + 1
            if name != "bash":
                continue
            command = _text(_obj(_obj(call).get("arguments")).get("command"))
            for route in _routes(command):
                counts[route] = counts.get(route, 0) + 1
    return counts


def profile(docs: Sequence[JsonValue], counts: Sequence[Mapping[str, int]]) -> str:
    """A run pooled: which tools the model reached for, how much of what it asked the
    engine to read came out of the KV cache, and where the seconds went. `counts` is one
    `tool_counts` row per trial, so committed rows profile without their trajectories."""
    tools: dict[str, int] = {}
    for row in counts:
        for name, n in row.items():
            tools[name] = tools.get(name, 0) + n
    prompt = cached = gen = think = steps = 0
    prefill_s = gen_s = 0.0
    for doc in docs:
        for step in agent_steps(doc):
            metrics = _obj(step.get("metrics"))
            extra = _obj(metrics.get("extra"))
            steps += 1
            prompt += int(_num(metrics.get("prompt_tokens")))
            cached += int(_num(metrics.get("cached_tokens")))
            gen += int(_num(metrics.get("completion_tokens")))
            think += int(_num(extra.get("think_tokens")))
            prefill_s += _num(extra.get("prefill_s"))
            gen_s += _num(extra.get("gen_s"))
    calls = sum(n for name, n in tools.items() if ":" not in name)
    rows = [f"{len(counts)} trials, {calls} tool calls" + (f", {steps} model steps" if steps else "")]
    rows += [f"  {name:<14} {n:>6}  {n / calls:>5.1%}"
             for name, n in sorted(tools.items(), key=lambda kv: -kv[1])]
    if prompt and gen:
        rows += [f"prompt tokens requested {prompt}, from cache {cached} ({cached / prompt:.1%}), "
                 f"prefilled {prompt - cached}",
                 f"generated {gen} tokens, {think} of them thinking ({think / gen:.0%})",
                 f"prefill {prefill_s:.0f} s, decode {gen_s:.0f} s"]
    return "\n".join(rows)


def load_counts(path: str) -> list[Mapping[str, int]]:
    rows: list[Mapping[str, int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row: JsonValue = json.loads(line)
            calls = _obj(_obj(row).get("calls"))
            rows.append({name: int(_num(n)) for name, n in calls.items()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trajectory", help="a trajectory, a directory of them, or --counts rows")
    ap.add_argument("--step", type=int, default=0, help="print one step in full")
    ap.add_argument("--width", type=int, default=90, help="action column width")
    ap.add_argument("--counts", action="store_true",
                    help="with a directory: one JSON line of tool-call counts per trial")
    args = ap.parse_args()
    if args.trajectory.endswith(".jsonl"):
        print(profile([], load_counts(args.trajectory)))
        return 0
    if os.path.isdir(args.trajectory):
        run = load_run(args.trajectory)
        if args.counts:
            for path, doc in run:
                print(json.dumps({"trajectory": path, "calls": tool_counts(doc)}, sort_keys=True))
        else:
            print(profile([doc for _, doc in run], [tool_counts(doc) for _, doc in run]))
        return 0
    with open(args.trajectory, encoding="utf-8") as f:
        trajectory: JsonValue = json.load(f)
    print(full_step(trajectory, args.step) if args.step else table(trajectory, args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
