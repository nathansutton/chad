#!/usr/bin/env python3
"""Plan 035 — across-turn prefill-tax analyzer (offline, pure, loads NO model).

Reads the per-step prefill trace emitted by agent.py when CHAD_PREFILL_TRACE is set
(one JSON row per engine.generate() call == one prefill event) and prints the three
tables the spike needs to make a GO/NO-GO call on prefix-stable compaction:

  A — TTFT curve:        how prefill_s / prompt_tokens grow across a session, and which
                         of the worst turns are compaction-caused.
  B — re-prefill attribution: of all tokens prefilled, what fraction is unavoidable new
                         content (append) vs re-prefill of content the cache had and lost
                         (reprefill-compaction / reprefill-other / warm-reload).
  C — addressable savings: of the re-prefilled tokens, how many a prefix-stable scheme
                         would have kept warm — the ceiling on what plan 036 can save,
                         translated to seconds of wait removed.

Usage:
    uv run python scripts/prefill_tax.py traces/sess.jsonl [more.jsonl ...]
    uv run python scripts/prefill_tax.py --selftest   # run the inline self-checks only

Row schema (see agent.py): {seq, step, prompt_tokens, cached_tokens, prefill_s,
gen_tokens, gen_s, compacted, sync_kind, peak_ctx}. The analyzer RE-DERIVES sync_kind
itself (single source of classification truth, fully testable on synthetic rows) and
warns if a row's recorded sync_kind disagrees — the recorded value is a convenience, the
derivation here is authoritative.
"""
import json
import sys

# --- classification ---------------------------------------------------------

APPEND = "append"
REPREFILL_COMPACTION = "reprefill-compaction"
REPREFILL_OTHER = "reprefill-other"
WARM_RELOAD = "warm-reload"
REPREFILL_KINDS = (REPREFILL_COMPACTION, REPREFILL_OTHER, WARM_RELOAD)


def classify(rows):
    """Walk rows in chronological (file) order and tag each with sync_kind, derived
    purely from this row's cached_tokens vs the prior row's total cache length plus the
    `compacted` flag — exactly the signals agent.py logs, recomputed here so the logic is
    unit-testable without a live model. Returns the same rows with a 'kind' key added.

    prior_total = cached + prompt + generated of the previous step = the cache length the
    next step inherits. If this step's cached prefix (cached_tokens) is >= that, nothing
    was lost (pure append). If it shrank, the step re-prefilled content the cache had:
      - compacted=True            -> the head rewrite caused it (reprefill-compaction)
      - else cached_tokens > 0    -> diverged but salvaged a prefix (warm-reload, cheap)
      - else                      -> cold reset (reprefill-other: think-mismatch / mid-edit)
    """
    prior_total = 0
    out = []
    for r in rows:
        common = r["cached_tokens"]
        if common >= prior_total:
            kind = APPEND
        elif r.get("compacted"):
            kind = REPREFILL_COMPACTION
        elif common > 0:
            kind = WARM_RELOAD
        else:
            kind = REPREFILL_OTHER
        r = dict(r, kind=kind)
        out.append(r)
        prior_total = common + r["prompt_tokens"] + r["gen_tokens"]
    return out


# --- stats helpers (no numpy: keep this pure/portable) ----------------------

def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _median(xs):
    return _pct(xs, 50)


# --- the three tables -------------------------------------------------------

def analyze(rows):
    """Return a dict of the A/B/C numbers from already-classified rows."""
    n = len(rows)
    total_prefill = sum(r["prompt_tokens"] for r in rows)
    prefill_s = [r["prefill_s"] for r in rows]

    # A — TTFT curve
    worst = sorted(rows, key=lambda r: r["prefill_s"], reverse=True)[: max(1, n // 10)]
    worst_compaction = sum(1 for r in worst if r["kind"] == REPREFILL_COMPACTION)

    # B — attribution by kind
    by_kind = {}
    for k in (APPEND, REPREFILL_COMPACTION, REPREFILL_OTHER, WARM_RELOAD):
        g = [r for r in rows if r["kind"] == k]
        toks = sum(r["prompt_tokens"] for r in g)
        by_kind[k] = {
            "turns": len(g),
            "tokens": toks,
            "pct": (100.0 * toks / total_prefill) if total_prefill else 0.0,
            "mean_prefill_s": (sum(x["prefill_s"] for x in g) / len(g)) if g else 0.0,
        }

    # Per-token prefill cost (s/token) from append turns (the clean signal: prefill of
    # genuinely new content, no reset overhead). Fall back to all turns if no appends.
    append_rows = [r for r in rows if r["kind"] == APPEND and r["prompt_tokens"] > 0]
    cost_rows = append_rows or [r for r in rows if r["prompt_tokens"] > 0]
    s_per_tok = (sum(r["prefill_s"] for r in cost_rows)
                 / sum(r["prompt_tokens"] for r in cost_rows)) if cost_rows else 0.0

    # C — addressable savings. On a re-prefill turn the cache re-fed tokens it already
    # had; a prefix-stable scheme would have kept all but the genuinely-new tail warm. We
    # estimate the genuinely-new tail by the typical append turn's prompt_tokens (one
    # turn's worth of fresh content); everything the re-prefill paid beyond that is
    # content it lost only because an earlier token moved — the addressable waste.
    baseline_new = _median([r["prompt_tokens"] for r in append_rows]) if append_rows else 0.0
    addressable = 0
    for r in rows:
        if r["kind"] in REPREFILL_KINDS:
            addressable += max(0, r["prompt_tokens"] - baseline_new)
    addressable = int(addressable)

    return {
        "n": n,
        "total_prefill": total_prefill,
        "median_prefill_s": _median(prefill_s),
        "p90_prefill_s": _pct(prefill_s, 90),
        "worst_n": len(worst),
        "worst_compaction": worst_compaction,
        "by_kind": by_kind,
        "s_per_tok": s_per_tok,
        "baseline_new": baseline_new,
        "addressable": addressable,
        "addressable_pct": (100.0 * addressable / total_prefill) if total_prefill else 0.0,
        "addressable_s": addressable * s_per_tok,
        "compaction_events": sum(1 for r in rows if r["kind"] == REPREFILL_COMPACTION),
    }


def render(rows, a):
    L = []
    L.append(f"prefill-tax analysis · {a['n']} prefill events · "
             f"{a['total_prefill']:,} tokens prefilled total")
    L.append("")
    # A
    L.append("A — TTFT curve")
    L.append(f"  median prefill_s = {a['median_prefill_s']:.3f}s   "
             f"p90 prefill_s = {a['p90_prefill_s']:.3f}s   "
             f"(p90/median = {a['p90_prefill_s'] / a['median_prefill_s']:.1f}x)"
             if a["median_prefill_s"] else "  (no prefill timing)")
    L.append(f"  worst {a['worst_n']} turns: {a['worst_compaction']} are "
             f"compaction-caused ({a['compaction_events']} compaction events total)")
    L.append("")
    # B
    L.append("B — re-prefill attribution")
    L.append(f"  {'cause':<24}{'turns':>6}{'tokens':>12}{'% prefill':>11}{'mean s':>9}")
    for k in (APPEND, REPREFILL_COMPACTION, REPREFILL_OTHER, WARM_RELOAD):
        d = a["by_kind"][k]
        L.append(f"  {k:<24}{d['turns']:>6}{d['tokens']:>12,}"
                 f"{d['pct']:>10.1f}%{d['mean_prefill_s']:>9.2f}")
    pct_sum = sum(a["by_kind"][k]["pct"] for k in a["by_kind"])
    L.append(f"  {'(sum)':<24}{'':>6}{a['total_prefill']:>12,}{pct_sum:>10.1f}%")
    L.append("")
    # C
    L.append("C — addressable savings (ceiling for prefix-stable compaction)")
    L.append(f"  baseline new-content / append turn = {a['baseline_new']:,.0f} tokens")
    L.append(f"  prefill cost = {a['s_per_tok'] * 1000:.2f} ms/token "
             f"(from append turns)")
    L.append(f"  addressable_tokens = {a['addressable']:,}  "
             f"= {a['addressable_pct']:.1f}% of total prefill")
    L.append(f"  ≈ {a['addressable_s']:.1f}s of wait removable across the session")
    return "\n".join(L)


# --- IO ---------------------------------------------------------------------

def load(paths):
    rows = []
    for p in paths:
        session = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    session.append(json.loads(line))
        if session:
            rows.append(session)  # one list per file = one capture session
    return rows


def main(argv):
    if "--selftest" in argv:
        return selftest()
    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 2
    sessions = load(paths)
    if not sessions:
        print("no rows in trace(s)")
        return 1
    # Classify each capture INDEPENDENTLY: every trace file is a fresh process with a
    # cold cache, so the prior-step cache length must reset at each session boundary
    # (concatenating then classifying once would mis-tag the first row of file N as a
    # re-prefill against file N-1's tail). Then pool the tagged rows for the A/B/C math.
    rows = []
    for session in sessions:
        rows.extend(classify(session))
    # cross-check recorded sync_kind against our re-derivation
    mism = [r for r in rows if "sync_kind" in r and r["sync_kind"] != r["kind"]]
    if mism:
        print(f"WARNING: {len(mism)} row(s) recorded sync_kind != re-derived kind "
              f"(first: seq={mism[0].get('seq')} {mism[0]['sync_kind']} != {mism[0]['kind']})\n")
    a = analyze(rows)
    print(render(rows, a))
    # Step-4 sanity: if compaction fired but compaction re-prefill is ~0, STOP.
    if a["compaction_events"] == 0:
        print("\nNOTE: zero compaction re-prefills detected — if the session DID compact, "
              "the capture didn't grow enough or tagging is wrong (plan 035 STOP).")
    return 0


# --- self-checks (plan 035 Test plan) ---------------------------------------

def _row(seq, prompt, cached, gen, prefill_s=0.0, compacted=False):
    return {"seq": seq, "step": seq, "prompt_tokens": prompt, "cached_tokens": cached,
            "gen_tokens": gen, "prefill_s": prefill_s, "gen_s": 0.0,
            "compacted": compacted, "peak_ctx": cached + prompt}


def selftest():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    # 1. All-append session: each step extends the prefix (cached == prior total), so no
    #    re-prefill at all and ~0 addressable.
    append_sess = classify([
        _row(1, 100, 0, 20, 0.5),       # first step: append (prior_total 0)
        _row(2, 30, 120, 15, 0.2),      # cached 120 == prior total (100+0+20)
        _row(3, 25, 165, 10, 0.2),      # cached 165 == prior total (30+120+15)
    ])
    kinds = [r["kind"] for r in append_sess]
    check(kinds == [APPEND, APPEND, APPEND], f"all-append tags {kinds}")
    a1 = analyze(append_sess)
    check(a1["by_kind"][REPREFILL_COMPACTION]["tokens"] == 0,
          "all-append: 0 compaction re-prefill tokens")
    check(a1["addressable"] == 0, "all-append: 0 addressable tokens")

    # 2. A compaction spike: step 3 compacts (render shrank) and the cached prefix
    #    collapses to a small warm prefix while a huge suffix re-prefills.
    comp_sess = classify([
        _row(1, 200, 0, 30, 0.8),
        _row(2, 40, 230, 20, 0.3),
        _row(3, 6000, 50, 25, 9.0, compacted=True),   # compaction -> big re-prefill
        _row(4, 35, 6075, 15, 0.3),
    ])
    kinds = [r["kind"] for r in comp_sess]
    check(kinds[2] == REPREFILL_COMPACTION, f"compaction step tagged {kinds[2]}")
    a2 = analyze(comp_sess)
    check(a2["by_kind"][REPREFILL_COMPACTION]["tokens"] == 6000,
          "compaction: 6000 tokens attributed to compaction")
    check(a2["by_kind"][REPREFILL_COMPACTION]["pct"] > 90,
          f"compaction dominates prefill ({a2['by_kind'][REPREFILL_COMPACTION]['pct']:.0f}%)")
    check(a2["addressable"] > 5000,
          f"compaction: addressable ~ the re-prefilled survivors ({a2['addressable']})")
    check(a2["worst_compaction"] == 1, "worst turn is the compaction spike")

    # 3. Cold divergence with no compaction (think-mismatch / mid-edit): cached collapses
    #    to 0 without a compaction flag -> reprefill-other, not compaction.
    div_sess = classify([
        _row(1, 200, 0, 30, 0.8),
        _row(2, 40, 230, 200, 0.3),
        _row(3, 5000, 0, 20, 7.0, compacted=False),  # cold reset, not compaction
    ])
    check(div_sess[2]["kind"] == REPREFILL_OTHER,
          f"cold divergence tagged {div_sess[2]['kind']}")
    a3 = analyze(div_sess)
    check(a3["by_kind"][REPREFILL_COMPACTION]["tokens"] == 0,
          "cold divergence: not attributed to compaction")

    # 4. Warm-reload: diverged (cached < prior total) but salvaged a prefix (cached>0),
    #    no compaction flag.
    warm_sess = classify([
        _row(1, 3000, 0, 30, 4.0),
        _row(2, 2000, 100, 20, 3.0, compacted=False),  # cached 100 < prior 3030, >0
    ])
    check(warm_sess[1]["kind"] == WARM_RELOAD,
          f"warm-reload tagged {warm_sess[1]['kind']}")

    # 5. % columns in B sum to ~100 on a mixed session.
    pct_sum = sum(a2["by_kind"][k]["pct"] for k in a2["by_kind"])
    check(abs(pct_sum - 100.0) < 1e-6, f"B percentages sum to 100 ({pct_sum:.4f})")

    print("\n" + ("ALL SELF-CHECKS PASS" if ok else "SELF-CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
