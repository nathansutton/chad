"""Speculative-decoding benchmark: what the block drafter is worth, and where.

Three things the headline tok/s number does not tell you, all measured here in ONE
process against ONE model load (loading a 12 GB model twice on a 24 GB box is how you
get a thrashing machine and numbers that mean nothing):

  1. **Throughput per arm.** Serial decoding (no drafter) against DFlash2 with its
     per-round width schedule and against a fixed full block. Median AND floor (worst
     prompt) over a prompt set, decode-only tok/s (`stats.gen_s`), so prefill never
     flatters a run. The floor is the number that decides the default width policy:
     a fixed full block pays its ~2.2-step round whatever it commits, so on
     low-acceptance text it can land below serial, where the schedule narrows.
  2. **Exactness.** Greedy speculative output against greedy serial output on the same
     prompt, with the first divergent token. The acceptance rule is exact, but the
     batched verify forward and the serial step run different matmul/attention kernels,
     so logits agree to rounding and a greedy run follows serial until the first
     near-tie. Measured: 4/10 160-token seeds bit-identical, the rest diverging 10–95
     tokens in at the SAME positions with `--fresh` and with cache sharing, and at
     different positions under CHAD_NO_QMM_MMA=1 — rounding's signature. A divergence
     at token 0 on an ordinary prompt, or a quality drop, is what a bug looks like.
  3. **Depth, on real agentic traffic.** The schedule picks a verified width per round.
     Prose seeds make it look easy; a real turn is ~2/3 `<think>` tokens, tool-call XML
     and quoted code, and those regimes accept very differently. The agentic corpus
     replays REAL contexts out of ~/.chad/sessions — a mid-session prompt with its tool
     results and schemas — and reports the width histogram and acceptance rate broken
     out by what the model was writing at the time.

Run it:

    uv run python benchmarks/spec_decode.py                   # both corpora, both presets
    uv run python benchmarks/spec_decode.py --corpus agentic --preset thinking
    uv run python benchmarks/spec_decode.py --tokens 384 --prompts 10

Nothing else may be holding the GPU: the model is ~12.1 GB and the drafter ~1.1 GB, so
a second live session on a 24 GB box pushes the run into swap and the numbers collapse
(measured: a 200-token decode that should take 5 s ran past 900 s).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

SESSION_DIR = os.path.expanduser("~/.chad/sessions")


def render(tok, messages, tools: bool = True) -> list:
    """Render a transcript, coerced to a plain id list.

    `apply_chat_template` hands back a BatchEncoding on this tokenizer, and `len()` of
    that is 2 (its key count) — the same trap `Agent._template_ids` guards.

    `tools` decides whether the tool schemas go in, and it changes what is being
    MEASURED, not just the prompt length. With schemas present the model is in agent
    mode: asked to continue a document it answers with a tool call and ends its turn
    after ~60 tokens, so a "prose continuation" arm silently becomes a short
    tool-call-XML arm (measured: 34% acceptance and generations as short as 58 tokens,
    against 52-66% on real agentic turns). The seed corpus therefore renders WITHOUT
    schemas — it exists to measure sustained prose/code decode — while the agentic
    corpus keeps them, because that is the real prompt shape."""
    from chad.agent import Agent
    from chad.tools import active_schemas
    rendered = tok.apply_chat_template(
        messages, tools=active_schemas() if tools else None,
        add_generation_prompt=True, enable_thinking=True)
    return list(Agent._template_ids(rendered))

# The shipped sampling presets, so an arm is measured in a regime someone actually runs.
PRESETS = {
    "greedy": dict(temp=0.0, top_p=0.0, top_k=0),
    "thinking": dict(temp=1.0, top_p=0.95, top_k=20),
}


# -- corpora -------------------------------------------------------------------

def _windows(path: str, tok, want: int, n: int, stride: int) -> list:
    """`n` token windows of ~`want` tokens each, cut out of a real repo file."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return []
    ids = tok.encode(text, add_special_tokens=False)
    out = []
    for i in range(n):
        start = (i * stride) % max(1, len(ids) - want)
        chunk = ids[start:start + want]
        if len(chunk) >= want // 2:
            out.append(tok.decode(chunk))
    return out


def seed_corpus(tok, root: str, count: int, seed_tokens: int = 512) -> list:
    """Prose and code continuations built out of this repo's own files, so the corpus
    is reproducible from a checkout rather than pasted into the script. Returns
    [(label, messages), ...]."""
    prose_files = ["docs/benchmarks.md", "docs/configuration.md", "README.md",
                   "docs/troubleshooting.md"]
    code_files = ["src/chad/tools.py", "src/chad/repomap.py"]
    items = []
    for f in prose_files:
        for w in _windows(os.path.join(root, f), tok, seed_tokens, 2, 4000):
            items.append(("prose", w))
    for f in code_files:
        for w in _windows(os.path.join(root, f), tok, seed_tokens, 1, 3000):
            items.append(("code", w))
    out = []
    for i, (kind, text) in enumerate(items[:count]):
        # "Keep going until you are told to stop" matters: a turn that ends at 60
        # tokens is mostly schedule ramp-up and reports a throughput the steady state
        # never sees.
        ask = ("Continue this documentation in the same voice. Write at least eight "
               "full paragraphs; do not summarize and do not stop early."
               if kind == "prose" else
               "Continue this module with several more functions in the same style. "
               "Write complete implementations with docstrings; do not stop early.")
        out.append((f"{kind}{i}", [
            {"role": "user", "content": f"{ask}\n\n```\n{text}\n```"}]))
    return out


def agentic_corpus(tok, count: int, min_ctx: int, max_ctx: int) -> list:
    """Real mid-session contexts from ~/.chad/sessions: every transcript prefix that
    ENDS in a tool result (i.e. the model is about to write a real agentic turn), sized
    into [min_ctx, max_ctx] rendered tokens. Newest sessions first, at most one cut per
    session so the corpus is not ten slices of the same conversation."""
    files = sorted(glob.glob(os.path.join(SESSION_DIR, "*.json")),
                   key=os.path.getmtime, reverse=True)
    out = []
    for path in files:
        if len(out) >= count:
            break
        try:
            msgs = json.load(open(path))["messages"]
        except Exception:  # noqa: BLE001 — a corrupt session is skipped, not fatal
            continue
        if not any(m.get("role") == "tool" for m in msgs):
            continue
        # The last prefix ending in a tool result is the deepest real context.
        for cut in range(len(msgs) - 1, 1, -1):
            if msgs[cut - 1].get("role") != "tool":
                continue
            prefix = [m for m in msgs[:cut]
                      if m.get("role") in ("system", "user", "assistant", "tool")]
            try:
                ids = render(tok, prefix)
            except Exception:  # noqa: BLE001 — template rejects an odd turn order
                continue
            if min_ctx <= len(ids) <= max_ctx:
                out.append((os.path.basename(path)[:8], prefix))
                break
    return out


# -- depth recording -----------------------------------------------------------

class Recorder:
    """Per-round (width, accepted) trace for one generation, with the committed-token
    span each round produced so rounds can be attributed to what was being written."""

    def __init__(self):
        self.rounds = []          # (width, accepted, committed_before)
        self.committed = 0

    def install(self):
        """Wrap mlx_dflash.block_policy so every policy handed to the engine records
        what it chose and what came back. Returns an uninstall callable."""
        from chad import mlx_dflash
        orig = mlx_dflash.block_policy
        rec = self

        def patched(max_depth, costs=None):
            pol = orig(max_depth, costs)
            real_depth, real_record = pol.depth, pol.record
            pending = {"k": 0}

            def depth():
                pending["k"] = real_depth()
                return pending["k"]

            def record(proposed, accepted, stopped_early):
                rec.rounds.append((proposed, accepted, rec.committed))
                rec.committed += accepted + 1
                return real_record(proposed, accepted, stopped_early)

            pol.depth, pol.record = depth, record
            return pol

        mlx_dflash.block_policy = patched
        return lambda: setattr(mlx_dflash, "block_policy", orig)


def classify(tok, gen_ids: list, rounds: list) -> dict:
    """Attribute each round to the regime it was decoding: inside a <think> block,
    emitting a tool call, or writing prose/code. The assistant turn opens inside
    <think> (the chat template opens it), so the state machine starts there."""
    buckets: dict = {}
    text_so_far = ""
    for width, accepted, before in rounds:
        span = tok.decode(gen_ids[before:before + accepted + 1]) if gen_ids else ""
        if "</think>" in text_so_far:
            kind = "toolcall" if ("<tool_call>" in text_so_far.rsplit("</think>", 1)[1]
                                  or "<function=" in text_so_far.rsplit("</think>", 1)[1]
                                  ) else "answer"
        else:
            kind = "think"
        b = buckets.setdefault(kind, {"rounds": 0, "proposed": 0, "accepted": 0,
                                      "widths": []})
        b["rounds"] += 1
        b["proposed"] += width
        b["accepted"] += accepted
        b["widths"].append(width)
        text_so_far += span
    return buckets


# -- arms ----------------------------------------------------------------------

def set_arm(eng, arm: str, saved_drafter):
    """Configure the loaded engine for one arm. No reload: the drafter object is kept
    aside and re-attached, so every arm measures the same weights in the same process."""
    if arm == "serial":
        eng._dflash = None
        return
    eng._dflash = saved_drafter
    eng.dflash_num_draft = 7
    eng.dflash_adaptive = (arm == "schedule")
    if arm.startswith("fixed"):
        eng.dflash_num_draft = int(arm[5:])


def run_one(eng, tok, messages, max_tokens, record=False, reset=True,
            tools=True):
    """One generation. Returns (tok_s, stats, gen_ids, rounds).

    `reset` drops the cache first. It is only needed for the FIRST arm on a given
    prompt: the arms that follow feed the identical prompt, so `_sync_to` rewinds the
    previous run's generated tail and re-prefills nothing. That is worth arranging —
    an agentic context is 12–19k tokens, and prefilling it once per arm instead of once
    per prompt would cost more wall-clock than every decode in the run put together.
    Decode throughput is measured from `stats.gen_s`, which excludes prefill either
    way, so the sharing cannot flatter an arm."""
    ids = render(tok, messages, tools=tools)
    if reset:
        eng._reset_cache()
        eng._cached_ids = []
        eng._warm_prefix_ids = None
    rec = Recorder() if record else None
    undo = rec.install() if rec else None
    try:
        _, stats = eng.generate(ids, max_tokens, None, [])
    finally:
        if undo:
            undo()
    tps = stats.generated_tokens / stats.gen_s if stats.gen_s else 0.0
    return tps, stats, list(stats.gen_ids or []), (rec.rounds if rec else [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", choices=["seeds", "agentic", "both"], default="both")
    ap.add_argument("--preset", choices=["greedy", "thinking", "both"], default="both")
    ap.add_argument("--arms", default="serial,schedule,fixed7")
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--min-ctx", type=int, default=2000)
    ap.add_argument("--max-ctx", type=int, default=20000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", default=None, help="write the raw per-run rows here")
    ap.add_argument("--fresh", action="store_true",
                    help="drop the cache before EVERY arm (not just the first per prompt): "
                         "removes cache sharing as a variable when chasing an exactness "
                         "difference, at the cost of one prefill per arm")
    a = ap.parse_args(argv)

    from chad.cli import _pick_model
    from chad.engine import Engine

    model_id = a.model or _pick_model()[0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"loading {model_id} ...", flush=True)
    eng = Engine(model_id=model_id, cache_dir=None)
    load_s = eng.load()
    saved = eng._dflash
    print(f"loaded in {load_s:.1f}s  drafter={'yes' if saved else 'NO'}", flush=True)
    if saved is None:
        print("no drafter: every arm would measure the same serial path", file=sys.stderr)
        return 1
    tok = eng.tok

    corpora = {}
    if a.corpus in ("seeds", "both"):
        corpora["seeds"] = seed_corpus(tok, root, a.prompts)
    if a.corpus in ("agentic", "both"):
        corpora["agentic"] = agentic_corpus(tok, a.prompts, a.min_ctx, a.max_ctx)
    for name, c in corpora.items():
        print(f"corpus {name}: {len(c)} prompts", flush=True)

    presets = ["greedy", "thinking"] if a.preset == "both" else [a.preset]
    arms = a.arms.split(",")

    # Warm the Metal variants once so the first measured round is not a compile.
    print("warming ...", flush=True)
    warm = next(iter(corpora.values()))[0][1]
    for arm in arms:
        set_arm(eng, arm, saved)
        for k, v in PRESETS["greedy"].items():
            setattr(eng, k, v)
        t0 = time.time()
        run_one(eng, tok, warm, 24)
        print(f"  warmed {arm} in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for corpus_name, corpus in corpora.items():
        for label, messages in corpus:
            first = True
            for preset in presets:
                for arm in arms:
                    set_arm(eng, arm, saved)
                    for k, v in PRESETS[preset].items():
                        setattr(eng, k, v)
                    t0 = time.time()
                    tps, stats, gen_ids, rounds = run_one(
                        eng, tok, messages, a.tokens,
                        record=(arm == "schedule"), reset=first or a.fresh,
                        tools=(corpus_name == "agentic"))
                    first = False
                    row = dict(corpus=corpus_name, preset=preset, arm=arm, prompt=label,
                               tok_s=round(tps, 2), tokens=stats.generated_tokens,
                               gen_s=round(stats.gen_s, 2),
                               prompt_tokens=stats.prompt_tokens,
                               cached_tokens=stats.cached_tokens,
                               prefill_s=round(stats.prefill_s, 2),
                               forwards=stats.forwards,
                               proposed=stats.draft_proposed,
                               accepted=stats.draft_accepted,
                               full=stats.generated_tokens >= a.tokens,
                               wall_s=round(time.time() - t0, 1))
                    if rounds:
                        row["buckets"] = classify(tok, gen_ids, rounds)
                        row["widths"] = [w for w, _, _ in rounds]
                    row["gen_ids"] = gen_ids
                    rows.append(row)
                    if a.json:
                        # Flush after every run. This is a ~45 min run on a machine
                        # where one stray SIGTERM (or another model loading) ends it;
                        # a partial JSON is a partial result, a lost JSON is an hour.
                        with open(a.json, "w") as f:
                            json.dump(rows, f)
                    print(f"  {corpus_name:8s} {preset:8s} {arm:9s} {label:10s} "
                          f"{tps:6.1f} tok/s  ({stats.generated_tokens} tok, "
                          f"prefill {stats.prompt_tokens}/{stats.prefill_s:.1f}s)",
                          flush=True)

    report(rows, arms, presets, list(corpora))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f)
        print(f"\nraw rows -> {a.json}")
    return 0


def report(rows, arms, presets, corpora):
    def vals(corpus, preset, arm):
        return [r["tok_s"] for r in rows
                if r["corpus"] == corpus and r["preset"] == preset and r["arm"] == arm]

    # Median AND floor per cell. The median says what an arm is worth on a typical
    # prompt; the floor (worst prompt) says what it costs when the drafter is wrong —
    # a fixed full block pays a ~2.2-step round whatever it commits, so on
    # low-acceptance text it can land BELOW serial, which no median will show.
    w = 76
    print("\n" + "=" * w)
    print("decode throughput — tok/s (decode only, prefill excluded)")
    print("=" * w)
    print(f"{'corpus':10s}{'preset':10s}{'':8s}" + "".join(f"{arm:>13s}" for arm in arms)
          + f"{'speedup':>12s}")
    print("-" * w)
    for corpus in corpora:
        for preset in presets:
            per_arm = [vals(corpus, preset, arm) for arm in arms]
            if not any(per_arm):
                continue
            for stat, fn in (("median", statistics.median), ("floor", min)):
                cells = [fn(v) if v else 0.0 for v in per_arm]
                base = cells[0] if cells and arms[0] == "serial" else 0
                best = max(cells) if cells else 0
                sp = f"{best / base:.2f}x" if base else "-"
                print(f"{corpus:10s}{preset:10s}{stat:8s}"
                      + "".join(f"{c:13.1f}" for c in cells) + f"{sp:>12s}")
    print("-" * w)

    # Acceptance and width, by what was being written.
    print("\n" + "=" * w)
    print("verified width & acceptance by content regime (schedule arm)")
    print("=" * w)
    print(f"{'corpus':10s}{'preset':10s}{'regime':10s}{'rounds':>8s}{'med w':>8s}"
          f"{'mean w':>8s}{'accept':>9s}{'tok/round':>11s}")
    print("-" * w)
    for corpus in corpora:
        for preset in presets:
            agg: dict = {}
            for r in rows:
                if r["corpus"] != corpus or r["preset"] != preset:
                    continue
                for kind, b in (r.get("buckets") or {}).items():
                    t = agg.setdefault(kind, {"rounds": 0, "proposed": 0,
                                              "accepted": 0, "widths": []})
                    t["rounds"] += b["rounds"]
                    t["proposed"] += b["proposed"]
                    t["accepted"] += b["accepted"]
                    t["widths"].extend(b["widths"])
            for kind in ("think", "toolcall", "answer"):
                b = agg.get(kind)
                if not b or not b["rounds"]:
                    continue
                acc = b["accepted"] / b["proposed"] if b["proposed"] else 0.0
                print(f"{corpus:10s}{preset:10s}{kind:10s}{b['rounds']:>8d}"
                      f"{statistics.median(b['widths']):>8.0f}"
                      f"{statistics.mean(b['widths']):>8.1f}"
                      f"{acc:>8.0%}"
                      f"{(b['accepted'] + b['rounds']) / b['rounds']:>11.2f}")
    print("-" * w)

    # Exactness: greedy serial vs greedy speculative, same prompt, same ids.
    print("\n" + "=" * w)
    print("exactness — greedy speculative output vs greedy serial output")
    print("=" * w)
    checked = mismatched = 0
    for r in rows:
        if r["preset"] != "greedy" or r["arm"] == "serial":
            continue
        ref = next((x for x in rows
                    if x["corpus"] == r["corpus"] and x["prompt"] == r["prompt"]
                    and x["preset"] == "greedy" and x["arm"] == "serial"), None)
        if not ref or not ref["gen_ids"]:
            continue
        checked += 1
        n = min(len(ref["gen_ids"]), len(r["gen_ids"]))
        if ref["gen_ids"][:n] != r["gen_ids"][:n]:
            mismatched += 1
            first = next(i for i in range(n)
                         if ref["gen_ids"][i] != r["gen_ids"][i])
            print(f"  MISMATCH {r['corpus']}/{r['prompt']} arm={r['arm']} "
                  f"at token {first}")
    if checked:
        print(f"  {checked - mismatched}/{checked} greedy runs token-identical "
              f"to the serial reference")
    else:
        print("  (no greedy pairs to compare — run with --preset greedy or both)")
    print("=" * w + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
