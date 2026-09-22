"""Read trial rows: score a run, pin a discrimination pool, compare two arms by task.

    python benchmarks/polyglot/stats.py score   _runs/baseline/trials.jsonl
    python benchmarks/polyglot/stats.py pool    _runs/baseline/trials.jsonl > _runs/baseline/pool.txt
    python benchmarks/polyglot/stats.py compare _runs/baseline/trials.jsonl _runs/arm-b/trials.jsonl
    python benchmarks/polyglot/stats.py subset --per-language 6 --seed harness-1
    python benchmarks/polyglot/stats.py subset --spread 3 --seed harness-3 --from subsets/harness-36.txt
    python benchmarks/polyglot/stats.py scorecard --ref _runs/h3-chad-llama _runs/h3-*

Stdlib only; nothing here loads a model.

WHY THE COMPARISON IS PAIRED
----------------------------
Two arms' pass RATES differ mostly because tasks differ in hardness, and that variance
is the same in both arms — it carries no information about the change under test and
swamps what does. Pairing by task cancels it: each task contributes the difference
between its own pass rate under B and under A, and the test asks only whether those
differences lean one way. With one rep per arm this is exactly McNemar's exact test on
the discordant pairs; with several it is the sign test on per-task deltas. Either way
significance is a matter of how many tasks FLIPPED, not how many were run.

WHY THERE IS A POOL
-------------------
A task the baseline always passes, or never passes, almost never flips, so a trial spent
on it buys nearly nothing. The pool is the tasks with a baseline pass rate strictly
between 0 and 1 — where a trial's outcome is most uncertain and therefore most
informative — plus, optionally, the never-passed ones (where only an improvement can
show). Pin the pool from a baseline run BEFORE looking at the arm you want to test, run
both arms on it, and the same evidence costs a fraction of the trials. The full set
still has a job the pool cannot do: checking that a change did not break the tasks that
used to be safe.

WHAT A SCORECARD READS
----------------------
Where a harness spends a local engine's time, from each trial's trajectory — chad's own,
or the proxy's for any other arm, which count the same way: the prompt it sends before
the model has done anything (the tax, turn 1), how long that first prefill takes, and on
every later turn the tokens the server's prefix cache could not serve and the wait they
cost. Those are properties of the harness and repeat from rep to rep; pass rates are
mostly the model's, so they appear only as a paired comparison with a reference arm.

WHY THERE IS ALSO A SUBSET
--------------------------
A pool needs a baseline of the very arm being compared, at two reps or more. Comparing
harnesses has no such arm to spare: they run on a GGUF the pool's baseline never saw,
and a llama-server trial costs two or three of an in-process one. `subset` is the
alternative that needs no run at all: a seeded draw of N tasks per language from the
manifest, committed before any arm runs, so the sample cannot have been chosen by its
results. It spends trials on tasks that never flip, which the paired test simply ties.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import JsonValue, is_array, is_number, is_object, is_text  # noqa: E402


@dataclass(frozen=True)
class Trial:
    task: str
    language: str
    passed: bool
    wall_s: float
    gen_tokens: int
    capped: bool


@dataclass(frozen=True)
class Comparison:
    tasks: int                   # tasks both arms ran
    up: int                      # tasks whose pass rate rose under B
    down: int
    p_value: float               # exact two-sided sign test on the non-tied tasks
    rate_a: float                # mean of per-task pass rates, over the shared tasks
    rate_b: float
    token_ratio: float           # B's generated tokens per trial over A's
    wall_ratio: float
    up_tasks: tuple[str, ...]
    down_tasks: tuple[str, ...]


def _number(row: Mapping[str, JsonValue], key: str) -> float:
    value = row.get(key)
    return float(value) if is_number(value) else 0.0


def load_trials(path: str) -> list[Trial]:
    trials = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row: JsonValue = json.loads(line)
            if not is_object(row):
                raise ValueError(f"{path}: a row is not a JSON object")
            task, language = row.get("task"), row.get("language")
            if not (is_text(task) and is_text(language)):
                raise ValueError(f"{path}: a row has no task/language")
            trials.append(Trial(task, language, row.get("passed") is True, _number(row, "wall_s"),
                                int(_number(row, "gen_tokens")), row.get("capped") is True))
    return trials


def pass_rates(trials: Sequence[Trial]) -> dict[str, float]:
    """Per-task pass rate over however many reps the task has."""
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for t in trials:
        outcomes[t.task].append(t.passed)
    return {task: sum(v) / len(v) for task, v in outcomes.items()}


def sign_test(up: int, down: int) -> float:
    """Exact two-sided binomial p-value for `up` vs `down` under a fair coin. Ties are
    excluded by the caller: a task that did not move says nothing about direction."""
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def flips_needed(alpha: float = 0.05) -> int:
    """The fewest one-directional flips (no counter-flips) that reach `alpha`. Useful as
    a sanity bound before a campaign: a pool with fewer tasks than this cannot produce a
    significant result whatever happens."""
    n = 1
    while sign_test(n, 0) > alpha:
        n += 1
    return n


def compare(a: Sequence[Trial], b: Sequence[Trial]) -> Comparison:
    rates_a, rates_b = pass_rates(a), pass_rates(b)
    shared = sorted(set(rates_a) & set(rates_b))
    if not shared:
        raise ValueError("the two runs share no tasks")
    up = tuple(t for t in shared if rates_b[t] > rates_a[t])
    down = tuple(t for t in shared if rates_b[t] < rates_a[t])

    def mean(trials: Sequence[Trial], field: str) -> float:
        rows = [t for t in trials if t.task in rates_a and t.task in rates_b]
        values = [t.gen_tokens if field == "gen_tokens" else t.wall_s for t in rows]
        return sum(values) / len(values) if values else 0.0

    tokens_a, wall_a = mean(a, "gen_tokens"), mean(a, "wall_s")
    return Comparison(
        tasks=len(shared), up=len(up), down=len(down), p_value=sign_test(len(up), len(down)),
        rate_a=sum(rates_a[t] for t in shared) / len(shared),
        rate_b=sum(rates_b[t] for t in shared) / len(shared),
        token_ratio=mean(b, "gen_tokens") / tokens_a if tokens_a else 0.0,
        wall_ratio=mean(b, "wall_s") / wall_a if wall_a else 0.0,
        up_tasks=up, down_tasks=down)


def pool(trials: Sequence[Trial], include_never: bool = False) -> list[str]:
    """Tasks that still discriminate under this baseline, sorted."""
    rates = pass_rates(trials)
    reps = defaultdict(int)
    for t in trials:
        reps[t.task] += 1
    if min(reps.values(), default=0) < 2:
        raise ValueError("a pool needs at least 2 reps per task: with one rep every task "
                         "has a pass rate of exactly 0 or 1 and nothing is on the frontier")
    return sorted(task for task, r in rates.items() if 0 < r < 1 or (include_never and r == 0))


def subset(names: Sequence[str], per_language: int, seed: str) -> list[str]:
    """`per_language` tasks from each language, by one seeded draw over the sorted names:
    the same manifest and seed give the same list on any machine and any Python."""
    by_language: dict[str, list[str]] = defaultdict(list)
    for name in sorted(names):
        by_language[name.split("/")[0]].append(name)
    rng = random.Random(seed)
    picked = []
    for language in sorted(by_language):
        if len(by_language[language]) < per_language:
            raise ValueError(f"{language} has only {len(by_language[language])} tasks")
        picked += rng.sample(by_language[language], per_language)
    return sorted(picked)


def spread(names: Sequence[str], n: int, seed: str) -> list[str]:
    """`n` tasks in `n` different languages: the languages drawn first, then one task in
    each — the most languages a sample of `n` can cover."""
    by_language: dict[str, list[str]] = defaultdict(list)
    for name in sorted(names):
        by_language[name.split("/")[0]].append(name)
    if n > len(by_language):
        raise ValueError(f"{n} tasks cannot span {len(by_language)} languages")
    rng = random.Random(seed)
    languages = sorted(rng.sample(sorted(by_language), n))
    return sorted(rng.choice(by_language[language]) for language in languages)


def read_task_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


@dataclass(frozen=True)
class Card:
    """One arm's line of the scorecard. Medians are over trials (turn 1) or over every
    later turn of every trial; `reuse` is pooled over later turns."""

    label: str
    trials: int
    passed: int
    capped: int
    tax: float                   # prompt tokens of turn 1
    first_wait_s: float          # prefill seconds of turn 1
    later_uncached: float        # prompt tokens the cache did not serve, per later turn
    later_wait_s: float          # prefill seconds per later turn
    reuse: float                 # cached share of later turns' prompt tokens
    steps: float                 # agent turns per trial
    side: float                  # requests beside the agent loop, per trial
    gen_tokens: float            # generated tokens per trial


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _obj(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if is_object(value) else {}


def _items(value: JsonValue) -> Sequence[JsonValue]:
    return value if is_array(value) else ()


def card(run_dir: str) -> Card:
    """An arm's line, from its rows and the trajectory each row names."""
    rows: list[Mapping[str, JsonValue]] = []
    with open(os.path.join(run_dir, "trials.jsonl"), encoding="utf-8") as f:
        for line in f:
            doc: JsonValue = json.loads(line) if line.strip() else None
            if is_object(doc):
                rows.append(doc)
    tax, first_wait, uncached, waits, steps, side, gen = [], [], [], [], [], [], []
    cached_sum = prompt_sum = 0.0
    for row in rows:
        side.append(_number(row, "side_requests"))
        gen.append(_number(row, "gen_tokens"))
        path = row.get("trajectory")
        if not is_text(path) or not os.path.exists(os.path.join(run_dir, path)):
            continue
        with open(os.path.join(run_dir, path), encoding="utf-8") as f:
            traj: JsonValue = json.load(f)
        turns = [_obj(_obj(s).get("metrics")) for s in _items(_obj(traj).get("steps"))
                 if _obj(s).get("source") == "agent" and "metrics" in _obj(s)
                 and _obj(_obj(s).get("extra")).get("in_flight") is not True]
        steps.append(len(turns))
        if not turns:
            continue
        tax.append(_number(turns[0], "prompt_tokens"))
        first_wait.append(_number(_obj(turns[0].get("extra")), "prefill_s"))
        for turn in turns[1:]:
            prompt, cached = _number(turn, "prompt_tokens"), _number(turn, "cached_tokens")
            uncached.append(prompt - cached)
            waits.append(_number(_obj(turn.get("extra")), "prefill_s"))
            prompt_sum, cached_sum = prompt_sum + prompt, cached_sum + cached
    return Card(os.path.basename(os.path.normpath(run_dir)), len(rows),
                sum(1 for r in rows if r.get("passed") is True),
                sum(1 for r in rows if r.get("capped") is True),
                _median(tax), _median(first_wait), _median(uncached), _median(waits),
                cached_sum / prompt_sum if prompt_sum else 0.0, _median(steps),
                sum(side) / len(side) if side else 0.0, _median(gen))


def scorecard_report(ref: str, runs: Sequence[str]) -> str:
    """The scorecard as a markdown table, each arm paired with `ref` by task."""
    reference = load_trials(os.path.join(ref, "trials.jsonl"))
    header = ("| arm | pass | capped | tax (tok) | turn-1 wait | uncached / later turn "
              "| wait / later turn | cache reuse | turns | side req | gen tok "
              f"| vs {os.path.basename(os.path.normpath(ref))}: up / down (p) |")
    lines = [header, "|" + "---|" * 12]
    for run_dir in runs:
        c = card(run_dir)
        if os.path.normpath(run_dir) == os.path.normpath(ref):
            paired = "reference"
        else:
            cmp = compare(reference, load_trials(os.path.join(run_dir, "trials.jsonl")))
            paired = f"{cmp.up} / {cmp.down} ({cmp.p_value:.2f})"
        lines.append(f"| {c.label} | {c.passed}/{c.trials} | {c.capped} | {c.tax:,.0f} "
                     f"| {c.first_wait_s:.1f} s | {c.later_uncached:,.0f} | {c.later_wait_s:.1f} s "
                     f"| {100 * c.reuse:.0f}% | {c.steps:.0f} | {c.side:.1f} | {c.gen_tokens:,.0f} "
                     f"| {paired} |")
    return "\n".join(lines)


def score_report(trials: Sequence[Trial]) -> str:
    rates = pass_rates(trials)
    by_lang: dict[str, list[float]] = defaultdict(list)
    lang_of = {t.task: t.language for t in trials}
    for task, rate in rates.items():
        by_lang[lang_of[task]].append(rate)
    reps = len(trials) / len(rates) if rates else 0
    lines = [f"{len(rates)} tasks, {len(trials)} trials ({reps:.1f} reps/task)",
             f"pass rate  {100 * sum(rates.values()) / len(rates):5.1f}%   (mean of per-task rates)"]
    for lang in sorted(by_lang):
        v = by_lang[lang]
        lines.append(f"  {lang:11s}{100 * sum(v) / len(v):5.1f}%   n={len(v)}")
    always = sum(1 for r in rates.values() if r == 1)
    never = sum(1 for r in rates.values() if r == 0)
    lines.append(f"always pass {always} | never pass {never} | frontier {len(rates) - always - never}")
    capped = sum(1 for t in trials if t.capped)
    lines.append(f"hit the wall cap: {capped} trials | median wall "
                 f"{sorted(t.wall_s for t in trials)[len(trials) // 2]:.0f} s | "
                 f"total {sum(t.wall_s for t in trials) / 3600:.1f} h")
    return "\n".join(lines)


def compare_report(c: Comparison) -> str:
    verdict = ("B better" if c.up > c.down else "B worse" if c.down > c.up else "no direction")
    lines = [
        f"{c.tasks} shared tasks | A {100 * c.rate_a:.1f}%  B {100 * c.rate_b:.1f}%",
        f"flips: {c.up} up, {c.down} down, {c.tasks - c.up - c.down} tied  ->  {verdict}, "
        f"p = {c.p_value:.3f} (exact sign test, two-sided)",
        f"cost: B uses {c.token_ratio:.2f}x the generated tokens, {c.wall_ratio:.2f}x the wall clock",
    ]
    if c.up_tasks:
        lines.append("up:   " + " ".join(c.up_tasks))
    if c.down_tasks:
        lines.append("down: " + " ".join(c.down_tasks))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("score").add_argument("trials")
    p = sub.add_parser("pool")
    p.add_argument("trials")
    p.add_argument("--include-never", action="store_true",
                   help="also keep tasks the baseline never passed")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    d = sub.add_parser("subset")
    how = d.add_mutually_exclusive_group(required=True)
    how.add_argument("--per-language", type=int, help="this many tasks in every language")
    how.add_argument("--spread", type=int, help="this many tasks, each in a different language")
    d.add_argument("--seed", required=True)
    d.add_argument("--from", dest="source", default="",
                   help="draw from this task list (default: manifest.json)")
    e = sub.add_parser("scorecard")
    e.add_argument("--ref", required=True, help="the run every other one is paired with")
    e.add_argument("runs", nargs="+", help="run directories, one line each")
    args = ap.parse_args()
    if args.command == "score":
        print(score_report(load_trials(args.trials)))
    elif args.command == "pool":
        print("\n".join(pool(load_trials(args.trials), args.include_never)))
    elif args.command == "scorecard":
        print(scorecard_report(args.ref, args.runs))
    elif args.command == "subset":
        from catalog import load_manifest
        names = read_task_list(args.source) if args.source else load_manifest()
        source = f" --from {args.source}" if args.source else ""
        if args.spread:
            picked = spread(names, args.spread, args.seed)
            print(f"# {args.spread} tasks in {args.spread} languages, drawn by `stats.py subset "
                  f"--spread {args.spread} --seed {args.seed}{source}`")
        else:
            picked = subset(names, args.per_language, args.seed)
            print(f"# {args.per_language} per language, drawn from manifest.json by "
                  f"`stats.py subset --per-language {args.per_language} --seed {args.seed}`")
        print("\n".join(picked))
    else:
        print(compare_report(compare(load_trials(args.a), load_trials(args.b))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
