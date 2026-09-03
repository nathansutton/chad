"""The local-fitness scorecard: what a person at the laptop felt, per harness, per turn.

`run.py table` says which arm finished the tasks and how fast. This says what the wait
was while it happened, derived from what the SERVER saw — `_runs/turns.jsonl`, written by
the sampler proxy for every generation request, carrying llama-server's own `timings` —
never from anything a harness reports about itself.

THE COLUMNS
-----------
Every request is first classified as a MAIN turn or a SIDE request. A side request is a
generation call that carries no tool schemas while the same run's other calls do: the
session-title / summary calls that opencode, goose, crush and dsh fire beside the agent
loop. They are real load on the only GPU and count toward the model's busy time and the
prefill seconds, but they are not turns of the agent, and letting them into the per-turn
columns produced two wrong numbers in the first version of this scorecard (a "first
prompt" of 228 tokens for goose, whose real first agent request is 9.6k; a "prefix churn"
that was side requests flipping the hash, not the agent prompt changing).

    tax                 prompt tokens of the first MAIN request (prompt_n + cache_n):
                        system prompt + tool schemas + the task. Paid at least once, and
                        how much of the window is gone before the user has typed a word.
    wait, turn 1        llama-server's own `prompt_ms` for that request: the time the
                        model spent reading before it could produce a token.
    uncached / later    `prompt_n` on main turns 2+ — the part of each turn the prefix
                        cache could not serve. Median, pooled over the arm's turns.
    wait / later turn   `prompt_ms` on main turns 2+, median and p90 (nearest rank).
                        The pause between a tool result landing and the model thinking.
    cache reuse         median over main turns 2+ of cache_n / (cache_n + prompt_n).
    prefill s / task    Σ prompt_ms over EVERY request of the run (side requests
                        included), median across tasks.
    exp. tok/s          generated tokens / wall clock for the whole task.
    tools, sys chars    tool schemas and system-message characters on the first main
                        request, as the harness sent them.
    prefix churn        main chat turns whose system-message or tool-list hash differs
                        from the previous main turn's, over turns compared.
    side requests       side requests across the arm's grid: how many ran CONCURRENTLY
                        with a main turn (their time window overlapped one), and how
                        many the harness abandoned before the server answered (the
                        proxy saw the client disconnect; no `timings` came back). Both
                        are load on the only GPU while the user is waiting.
    after side / turn   (JSON only) median `prompt_n` of a main turn — 3rd or later,
                        since the 2nd is every harness's test-file read — that follows
                        a side request, against one that follows a main turn. In the
                        committed run every side request sits beside turn 1, so there
                        is no sample and no column; an earlier draft that compared
                        turn 2 against later turns was measuring the test-file read.
    model busy          Σ (prompt_ms + predicted_ms) / wall. Above 100% means two
                        requests were in flight at once.
    round trips         main turns per task (median).
    ctx at exit         prompt_n + cache_n + predicted_n of the last main turn.
    pass                the gate. Printed, never ranked.

WHY NOT THE PROXY'S TIME-TO-FIRST-BYTE
--------------------------------------
The proxy stamps the first byte it sees back from llama-server, and llama-server starts
emitting bytes on a long prefill BEFORE the prefill is done (the streamed form carries
an early chunk), so that stamp sat at ~30 s for every request over ~3k prompt tokens
whatever the real read time was. The scorecard prints how many requests had a first
byte earlier than their own `prompt_ms`; the column is kept in the data as
`proxy_ttft_*` and used for nothing.

MLX arms have no server to observe. Their rows come from chad's own prefill trace
(`CHAD_PREFILL_TRACE`, same field definitions) and are marked self-reported; their turn-1
wait is a warm prefix loaded from disk, not a cold prefill, and is marked as such.

Run:
    uv run python benchmarks/matrix/scorecard.py               # -> _runs/scorecard.md + .json
    uv run python benchmarks/matrix/scorecard.py --runs DIR    # another run's artifacts
    uv run python benchmarks/matrix/scorecard.py --legacy      # grid.json + sampler_audit.jsonl
                                                               # only (runs before turns.jsonl)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS_DEFAULT = os.path.join(HERE, "_runs")
MLX_ARMS = ("chad+mlx", "chad+mlx-nodflash")


# -- loading -------------------------------------------------------------------

def _jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def load(runs: str) -> tuple:
    grid = os.path.join(runs, "grid.json")
    rows = json.load(open(grid)) if os.path.exists(grid) else []
    turns = [t for t in _jsonl(os.path.join(runs, "turns.jsonl"))
             if isinstance(t.get("rep"), int) and t["rep"] >= 0]
    audit = _jsonl(os.path.join(runs, "sampler_audit.jsonl"))
    return rows, turns, audit


# -- one request, normalised ---------------------------------------------------

def norm_turn(t: dict) -> dict | None:
    """One proxy row -> {prompt_n, cache_n, predicted_n, prompt_ms, predicted_ms, ...}.
    Prefers llama-server's `timings`; falls back to an OpenAI `usage` block (no ms)."""
    tm = t.get("timings") if isinstance(t.get("timings"), dict) else None
    us = t.get("usage") if isinstance(t.get("usage"), dict) else None
    base = {"ttft": t.get("ttft_s"), "sys_sha": t.get("sys_sha"),
            "tools_sha": t.get("tools_sha"), "shape": t.get("shape"),
            "n_tools": t.get("n_tools"), "sys_chars": t.get("sys_chars"),
            "t": t.get("t_arrive"), "t_end": t.get("t_last_byte"),
            "abandoned": bool(t.get("client_disconnected")), "measured": True}
    if tm and tm.get("prompt_n") is not None:
        return {"prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n") or 0,
                "predicted_n": tm.get("predicted_n") or 0,
                "prompt_ms": tm.get("prompt_ms"), "predicted_ms": tm.get("predicted_ms"),
                **base}
    if us:
        pt = us.get("prompt_tokens", us.get("input_tokens"))
        det = us.get("prompt_tokens_details") or us.get("input_tokens_details") or {}
        cached = det.get("cached_tokens") or 0
        if pt is not None:
            return {"prompt_n": pt - cached, "cache_n": cached,
                    "predicted_n": us.get("completion_tokens", us.get("output_tokens")) or 0,
                    "prompt_ms": None, "predicted_ms": None, **base}
    if t.get("status") == 200:
        # The server accepted it and the harness walked away before it answered (a
        # title call cancelled once the agent turn came back, typically). No numbers
        # came back, but it held a slot for as long as it ran; it is counted, not
        # measured.
        return {"prompt_n": None, "cache_n": None, "predicted_n": None,
                "prompt_ms": None, "predicted_ms": None, **base, "measured": False}
    return None


def norm_trace(r: dict) -> dict:
    """One CHAD_PREFILL_TRACE row -> the same shape. chad's `prompt_tokens` already
    excludes the cached prefix, so it maps onto llama-server's `prompt_n` directly."""
    return {"prompt_n": r.get("prompt_tokens", 0), "cache_n": r.get("cached_tokens", 0),
            "predicted_n": r.get("gen_tokens", 0),
            "prompt_ms": (r.get("prefill_s") or 0) * 1000.0,
            "predicted_ms": (r.get("gen_s") or 0) * 1000.0,
            "ttft": r.get("prefill_s"), "sys_sha": None, "tools_sha": None,
            "shape": "completion", "n_tools": None, "sys_chars": None,
            "t": r.get("seq", 0), "t_end": None, "abandoned": False, "measured": True}


def classify(turns: list) -> None:
    """Mark each turn `side` (a no-tools call beside a tool-bearing agent loop) or not.

    In place, on an ordered list of normalised turns from ONE run. A harness that never
    sends tool schemas (mini-swe-agent, chad's raw completion path) has no side
    requests by this rule — every call is the agent loop."""
    has_tools = any((t.get("n_tools") or 0) >= 1 for t in turns)
    for t in turns:
        t["side"] = bool(has_tools and not (t.get("n_tools") or 0))


# -- per (arm, task, rep) ------------------------------------------------------

def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _p90(xs):
    """Nearest-rank 90th percentile; None when empty."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[max(0, math.ceil(0.9 * len(xs)) - 1)]


def _s(ms):
    return None if ms is None else ms / 1000.0


def _overlaps(a: dict, b: dict) -> bool:
    """Two requests whose time windows at the proxy intersect."""
    if a["t"] is None or b["t"] is None:
        return False
    a_end, b_end = a.get("t_end") or a["t"], b.get("t_end") or b["t"]
    return a["t"] < b_end and b["t"] < a_end


def run_metrics(row: dict, turns: list) -> dict:
    """Derived numbers for one run. `turns` is normalised, ordered and classified."""
    wall = row.get("wall_s") or 0
    gen = row.get("generated") or 0
    side_all = [t for t in turns if t.get("side")]
    main_all = [t for t in turns if not t.get("side")]
    out = {"arm": row["arm"], "task": row["task"], "passed": bool(row.get("passed")),
           "timed_out": bool(row.get("timed_out")), "wall_s": wall,
           "tests": f"{row.get('tests_passed', 0)}/{row.get('tests_total', 0)}",
           "exp_toks": gen / wall if wall else None,
           "round_trips": len(main_all),
           "side_requests": len(side_all),
           "side_concurrent": sum(1 for s in side_all
                                  if any(_overlaps(s, m) for m in main_all)),
           "side_abandoned": sum(1 for s in side_all if s.get("abandoned")),
           "unmeasured": sum(1 for t in turns if not t.get("measured"))}
    # Everything numeric below is over requests the server reported on.
    all_turns = turns
    turns = [t for t in turns if t.get("measured")]
    if not turns:
        return out
    main = [t for t in turns if not t.get("side")]
    # Whole-run model time: side requests occupy the GPU too.
    pm = [t["prompt_ms"] for t in turns if t["prompt_ms"] is not None]
    gm = [t["predicted_ms"] for t in turns if t["predicted_ms"] is not None]
    if pm:
        out["prefill_s"] = sum(pm) / 1000.0
    if pm and gm and (sum(pm) + sum(gm)) > 0:
        out["prefill_share"] = sum(pm) / (sum(pm) + sum(gm))
        out["model_busy"] = ((sum(pm) + sum(gm)) / 1000.0) / wall if wall else None
    # Instrument check: proxy first-byte earlier than the server's own prefill time.
    out["ttft_lt_prefill"] = sum(1 for t in turns if t["ttft"] is not None
                                 and t["prompt_ms"] is not None
                                 and t["ttft"] < t["prompt_ms"] / 1000.0 - 0.5)
    out["ttft_checked"] = sum(1 for t in turns if t["ttft"] is not None
                              and t["prompt_ms"] is not None)
    if not main:
        return out
    first, later = main[0], main[1:]
    out["tax"] = (first["prompt_n"] or 0) + (first["cache_n"] or 0)
    out["wait_first"] = _s(first["prompt_ms"])
    out["proxy_ttft_cold"] = first["ttft"]
    out["proxy_ttft_warm"] = [t["ttft"] for t in later if t["ttft"] is not None]
    out["uncached_later"] = [t["prompt_n"] or 0 for t in later]
    out["wait_later"] = [_s(t["prompt_ms"]) for t in later if t["prompt_ms"] is not None]
    out["reuse"] = [t["cache_n"] / (t["cache_n"] + t["prompt_n"])
                    for t in later if (t["cache_n"] or 0) + (t["prompt_n"] or 0) > 0]
    out["n_tools"] = first.get("n_tools")
    out["sys_chars"] = first.get("sys_chars")
    # What a side request costs the NEXT main turn: uncached tokens on a main turn that
    # follows a side request, against one that follows a main turn. Main turns 3+ only:
    # turn 1 is the tax and turn 2 is every harness's test-file read, and a title call
    # fired right after turn 1 would otherwise be "followed" by a 3k-token read that
    # every arm pays anyway.
    # Walks the unfiltered sequence: an abandoned (unmeasured) side request still
    # precedes the turn after it.
    after_side, after_main = [], []
    late = set(id(t) for t in main[2:])
    for prev, cur in zip(all_turns, all_turns[1:]):
        if cur.get("side") or id(cur) not in late:
            continue
        (after_side if prev.get("side") else after_main).append(cur["prompt_n"] or 0)
    out["uncached_after_side"] = after_side
    out["uncached_after_main"] = after_main
    chat = [t for t in main if t.get("shape") == "chat" and t.get("sys_sha")]
    if len(chat) >= 2:
        out["churn"] = sum(1 for a, b in zip(chat, chat[1:])
                           if a["sys_sha"] != b["sys_sha"] or a["tools_sha"] != b["tools_sha"])
        out["churn_of"] = len(chat) - 1
    last = main[-1]
    out["ctx_exit"] = ((last["prompt_n"] or 0) + (last["cache_n"] or 0)
                       + (last["predicted_n"] or 0))
    return out


def _trace_path(row: dict, runs_dir: str) -> str:
    """The MLX arm's prefill trace: the row's own path (relative to the repo root, or
    absolute from an older run), else the deterministic per-run location."""
    p = row.get("prefill_trace")
    if p:
        for cand in (p, os.path.join(ROOT, p), os.path.join(runs_dir, p)):
            if os.path.exists(cand):
                return cand
    return os.path.join(runs_dir, "traces",
                        f"{row['arm']}-{row['task']}-{row.get('rep', 0)}",
                        "prefill_trace.jsonl")


def per_run(rows: list, turns: list, runs_dir: str) -> list:
    by_key = defaultdict(list)
    for t in turns:
        n = norm_turn(t)
        if n is not None:
            by_key[(t.get("arm"), t.get("task"), t.get("rep"))].append(n)
    out = []
    for row in rows:
        key = (row["arm"], row["task"], row.get("rep", 0))
        if row["arm"] in MLX_ARMS:
            ts = [norm_trace(r) for r in _jsonl(_trace_path(row, runs_dir))]
        else:
            ts = sorted(by_key.get(key, []), key=lambda t: t["t"] or 0)
        classify(ts)
        out.append(run_metrics(row, ts))
    return out


# -- per arm -------------------------------------------------------------------

def _pool(rs, key):
    return [x for r in rs for x in r.get(key, [])]


def aggregate(per: list) -> list:
    arms = []
    for arm in dict.fromkeys(r["arm"] for r in per):
        rs = [r for r in per if r["arm"] == arm]
        churn = sum(r.get("churn", 0) for r in rs if "churn" in r)
        churn_of = sum(r.get("churn_of", 0) for r in rs if "churn" in r)
        wait_later = _pool(rs, "wait_later")
        arms.append({
            "arm": arm, "n": len(rs),
            "passed": sum(1 for r in rs if r["passed"]),
            "timeouts": sum(1 for r in rs if r["timed_out"]),
            "tax": _med([r.get("tax") for r in rs]),
            "wait_first": _med([r.get("wait_first") for r in rs]),
            "uncached_later": _med(_pool(rs, "uncached_later")),
            "wait_later": _med(wait_later), "wait_later_p90": _p90(wait_later),
            "later_n": len(wait_later),
            "reuse": _med(_pool(rs, "reuse")),
            "prefill_s": _med([r.get("prefill_s") for r in rs]),
            "exp_toks": _med([r["exp_toks"] for r in rs]),
            "n_tools": _med([r.get("n_tools") for r in rs]),
            "sys_chars": _med([r.get("sys_chars") for r in rs]),
            "churn": (churn, churn_of) if churn_of else None,
            "side_requests": sum(r.get("side_requests", 0) for r in rs),
            "side_concurrent": sum(r.get("side_concurrent", 0) for r in rs),
            "side_abandoned": sum(r.get("side_abandoned", 0) for r in rs),
            "uncached_after_side": _med(_pool(rs, "uncached_after_side")),
            "uncached_after_main": _med(_pool(rs, "uncached_after_main")),
            "prefill_share": _med([r.get("prefill_share") for r in rs]),
            "model_busy": _med([r.get("model_busy") for r in rs]),
            "round_trips": _med([r["round_trips"] for r in rs if r["round_trips"]]),
            "ctx_exit": _med([r.get("ctx_exit") for r in rs]),
            "proxy_ttft_cold": _med([r.get("proxy_ttft_cold") for r in rs]),
            "proxy_ttft_warm": _med(_pool(rs, "proxy_ttft_warm")),
            "ttft_lt_prefill": sum(r.get("ttft_lt_prefill", 0) for r in rs),
            "ttft_checked": sum(r.get("ttft_checked", 0) for r in rs),
            "self_reported": arm in MLX_ARMS,
        })
    return arms


# -- rendering -----------------------------------------------------------------

def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _num(x, fmt="{:.1f}"):
    return "–" if x is None else fmt.format(x)


def _table(hdr: list, rows: list) -> str:
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    lines = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)) + " |",
             "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    lines += ["| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
              for r in rows]
    return "\n".join(lines)


def _gate(a: dict) -> str:
    return f"{a['passed']}/{a['n']}" + (f" T{a['timeouts']}" if a["timeouts"] else "")


def render(arms: list, per: list, tasks: list) -> str:
    out = ["### Local-fitness scorecard — same weights, same laptop, same tasks\n",
           "#### What it feels like\n"]
    hdr = ["Arm", "tax: turn-1 prompt (tok)", "wait before 1st token, turn 1",
           "uncached tok / later turn (med)", "wait / later turn (med · p90)",
           "cache reuse", "prefill s / task (med)", "exp. tok/s", "pass (gate)"]
    body = []
    for a in arms:
        mark = " *" if a["self_reported"] else ""
        w1 = _num(a["wait_first"], "{:.1f} s") + (" †" if a["self_reported"] else "")
        body.append([a["arm"] + mark, _num(a["tax"], "{:,.0f}"), w1,
                     _num(a["uncached_later"], "{:,.0f}"),
                     f"{_num(a['wait_later'], '{:.1f} s')} · "
                     f"{_num(a['wait_later_p90'], '{:.0f} s')}",
                     _pct(a["reuse"]), _num(a["prefill_s"], "{:.0f}"),
                     _num(a["exp_toks"]), _gate(a)])
    out.append(_table(hdr, body))
    out.append("""
Every column but the last two is llama-server's own accounting, read through the proxy
(`_runs/turns.jsonl`), never a harness's self-report. **tax** = prompt tokens of the
first agent request (`prompt_n + cache_n`: system prompt + tool schemas + task); **wait,
turn 1** = the server's `prompt_ms` for that request; **uncached / later turn** =
`prompt_n` on agent turns 2+, pooled median; **wait / later turn** = `prompt_ms` on those
turns, median and p90; **cache reuse** = `cache_n / (cache_n + prompt_n)` on those turns;
**prefill s / task** = Σ `prompt_ms` over every request of a task, side requests included;
**exp. tok/s** = generated tokens / wall clock. Side requests (title / summary calls with
no tool schemas) are excluded from the per-turn columns and counted in the next table.
The pass column is a gate, not a ranking.""")
    if any(a["self_reported"] for a in arms):
        out.append("`*` in-process arm: the same fields from chad's own prefill trace, "
                   "self-reported — no server saw it. `†` its turn-1 wait is a system-prompt "
                   "prefix restored from disk, not a cold prefill; the chad+llama row is "
                   "the cold number for the same prompt.")

    out.append("\n#### Shape of the harness\n")
    hdr = ["Arm", "tools", "system prompt (chars)", "prefix churn",
           "side requests (concurrent · abandoned)", "round trips / task",
           "model busy", "prefill share", "ctx at exit"]
    body = []
    for a in arms:
        churn = "–" if not a["churn"] else f"{a['churn'][0]}/{a['churn'][1]}"
        side = (f"{a['side_requests']} ({a['side_concurrent']} · {a['side_abandoned']})"
                if a["side_requests"] else "0")
        body.append([a["arm"] + (" *" if a["self_reported"] else ""),
                     _num(a["n_tools"], "{:.0f}"), _num(a["sys_chars"], "{:,.0f}"),
                     churn, side, _num(a["round_trips"], "{:.0f}"),
                     _pct(a["model_busy"]), _pct(a["prefill_share"]),
                     _num(a["ctx_exit"], "{:,.0f}")])
    out.append(_table(hdr, body))
    out.append("""
**tools** / **system prompt** as the harness sent them on its first agent request;
**prefix churn** = agent turns whose system-message or tool-list hash changed since the
previous agent turn, over turns compared (`–`: chad's raw `/completion` path and the MLX
arms carry no messages to hash — their cache-reuse column is the evidence instead);
**side requests** = no-tools calls beside the agent loop (session titles, summaries),
summed over the arm's grid: how many overlapped an agent turn in time, and how many the
harness abandoned before the server answered (no `timings` came back; counted, not
measured); **model busy** = Σ (`prompt_ms` + `predicted_ms`) /
wall over measured requests — above 100% means two were in flight at once; **round
trips** = agent turns per task; **ctx at exit** = tokens in context at the last agent
turn.""")

    checked = sum(a["ttft_checked"] for a in arms if not a["self_reported"])
    early = sum(a["ttft_lt_prefill"] for a in arms if not a["self_reported"])
    if checked:
        out.append(f"\nInstrument check: the proxy's own first-byte stamp came back "
                   f"earlier than the server's `prompt_ms` on {early} of {checked} "
                   f"requests (llama-server streams a first chunk before a long prefill "
                   f"finishes), so no time-to-first-byte column is printed; the server's "
                   f"prefill time is the wait.")

    out.append("\n#### Per task — cache reuse (median, agent turns 2+) · "
               "wait / later turn (median) · experienced tok/s\n")
    names = [a["arm"] for a in arms]
    body = []
    for task in tasks:
        line = [task]
        for arm in names:
            rs = [r for r in per if r["arm"] == arm and r["task"] == task]
            if not rs:
                line.append("–")
                continue
            r = rs[-1]
            cell = (f"{_pct(_med(r.get('reuse', [])))} · "
                    f"{_num(_med(r.get('wait_later', [])), '{:.1f}s')} · "
                    f"{_num(r['exp_toks'])}")
            if not r["passed"]:
                cell += " (T)" if r["timed_out"] else " (x)"
            line.append(cell)
        body.append(line)
    out.append(_table(["Task"] + names, body))
    out.append("`(x)` failed tests, `(T)` timed out — the numbers still describe the "
               "turns that happened.")
    return "\n".join(out) + "\n"


# -- legacy: what the pre-turns.jsonl artifacts can already say -----------------

def legacy(rows: list, audit: list, tasks: list) -> str:
    """Round trips, prefill per round trip, generated per round trip and experienced
    tok/s from grid.json + the sampler audit alone. For runs made before the proxy
    wrote turn records."""
    rt = defaultdict(int)
    for r in audit:
        rt[(r.get("arm"), r.get("task"))] += 1
    out = ["### What the harness×engine run already says about feel (no turn records)\n"]
    hdr = ["Arm", "pass", "wall (sum)", "generated", "prefill", "round trips",
           "prefill / rt", "gen / rt", "exp. tok/s"]
    body = []
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]
        wall = sum(r["wall_s"] for r in rs)
        gen = sum(r["generated"] for r in rs)
        pre = sum(r["prefill"] for r in rs)
        n = sum(rt.get((arm, r["task"]), 0) for r in rs)
        mark = " *" if arm in MLX_ARMS else ""
        body.append([arm + mark, f"{sum(1 for r in rs if r['passed'])}/{len(rs)}",
                     f"{wall:,.0f}s", f"{gen:,}", f"{pre:,}",
                     str(n) if n else "–",
                     f"{pre / n:,.0f}" if n else "–", f"{gen / n:,.0f}" if n else "–",
                     f"{gen / wall:.1f}" if wall else "–"])
    out.append(_table(hdr, body))
    out.append("\n`prefill / rt` is prompt tokens the server evaluated per model call — the "
               "tokens the prefix cache did NOT serve. Same server, same cache, same "
               "tasks; the spread is the harness. `*` self-reported (in-process), no "
               "round-trip count.")
    hdr2 = ["Task"] + [a[0].rstrip(" *") for a in body]
    body2 = []
    for task in tasks:
        line = [task]
        for arm in [a[0].rstrip(" *") for a in body]:
            rs = [r for r in rows if r["arm"] == arm and r["task"] == task]
            if not rs:
                line.append("–")
                continue
            r = rs[-1]
            n = rt.get((arm, task), 0)
            cell = f"{r['generated'] / r['wall_s']:.1f}"
            cell += f" · {r['prefill'] / n:,.0f}/rt" if n else ""
            if not r["passed"]:
                cell += " (T)" if r["timed_out"] else " (x)"
            line.append(cell)
        body2.append(line)
    out.append("\n### Per task — experienced tok/s · prefill tokens per round trip\n")
    out.append(_table(hdr2, body2))
    return "\n".join(out) + "\n"


def build(runs_dir: str) -> tuple:
    """(markdown, json-able dict) for a run directory, or (None, None) without rows."""
    rows, turns, _ = load(runs_dir)
    if not rows:
        return None, None
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    per = per_run(rows, turns, runs_dir)
    arms = aggregate(per)
    return render(arms, per, tasks), {"arms": arms, "per_run": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=RUNS_DEFAULT)
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--out", default=None, help="markdown path (default: <runs>/scorecard.md)")
    a = ap.parse_args(argv)
    rows, turns, audit = load(a.runs)
    if not rows:
        print(f"no grid.json under {a.runs}")
        return 1
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    if a.legacy:
        text = legacy(rows, audit, tasks)
        out = a.out or os.path.join(a.runs, "scorecard-legacy.md")
    else:
        if not turns and not any(r["arm"] in MLX_ARMS for r in rows):
            print("no turns.jsonl — this run predates the proxy's turn records; "
                  "use --legacy", file=sys.stderr)
            return 1
        text, data = build(a.runs)
        out = a.out or os.path.join(a.runs, "scorecard.md")
        with open(os.path.splitext(out)[0] + ".json", "w") as f:
            json.dump(data, f, indent=1)
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
