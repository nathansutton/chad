"""Qwen3.8-class MTP (multi-token-prediction) head: sidecar build + load + forward.

The Qwen3.8 checkpoints ship a 1-layer MTP head — `fc` over
[norm(embedding), norm(hidden)] + one full-attention DecoderLayer + a final
norm, sharing the main model's embeddings and lm_head — that mainstream
loaders (mlx-lm, transformers) STRIP on load. This module rebuilds it as a
first-class mlx module so the engine can self-speculate: draft k tokens with
the cheap head, verify them in one batched main-model forward.

Semantics follow the serving-engine reference implementations: the head's
`hidden` input is the main model's POST-final-norm hidden state (exactly what
mlx-lm's `Qwen3_5TextModel.__call__` returns), and `pre_fc_norm_hidden` is
applied on top — the double normalization is how the head was trained. Draft
chaining feeds the head's own output hidden back as the next step's `hidden`.

The head is attention-only (plain KVCache — natively trimmable), so
speculative rollback needs no recurrent snapshots on the MTP side; only the
main model's hybrid cache needs the engine's snapshot/trim machinery.

Weights are not part of the quantized text-only conversions, so they load
from a SIDECAR safetensors file, resolved in order:
  1. $CHAD_MTP_PATH (explicit file)
  2. <model_dir>/mtp.safetensors (a conversion that kept the head)
  3. ~/.cache/chad/mtp/<model-dir-basename>.safetensors (built locally)
Build one from an original-checkpoint shard holding the `mtp.*` tensors:
  uv run python -m chad.mlx_mtp <shard.safetensors> --model <model_dir>
"""

import math
import os
from typing import Any, Optional

from .diag import log

_SIDECAR_DIR = os.path.expanduser("~/.cache/chad/mtp")


class DepthPolicy:
    """Cost-model draft schedule: choose the depth that maximizes expected
    committed tokens per unit round time.

    A round of depth d costs T(d) = V + d*H (one width-(d+1) verify plus d
    head steps) and commits E[tokens](d) = 1 + sum_{k=1..d} prod_{i<k} p_i,
    where p_i is the estimated acceptance of draft position i GIVEN the
    prefix before it was accepted. Greedy marginal rule: extend to position
    d+1 exactly while

        prod_{i<=d+1} p_i  >  h * (1 + E_d) / (1 + d*h)

    with h = H/V the head step's cost relative to the verify forward. The
    p_i are per-position EMAs seeded with an optimistic, gently decaying
    prior (the first rounds should draft, not stall); a fully accepted round
    also transfers bounded optimism to the first unreached position, so deep
    depths become reachable inside a short window. On hot prose this runs
    to the cap; on cold prompts it collapses to 1 and then to 0 (an adaptive
    skip costs exactly what a serial step costs).

    The pending top-2 logit margin (of the row that produced the pending
    token) additionally caps p_0/p_1: a near-tie next token is exactly where
    the head is about to be wrong, whatever its recent streak says.

    THE COST SURFACE IS NOT CONVEX (measured, M4 Pro / 4-bit 27B): verify
    width pays ~25 ms PER ROW through S=8 (ALU-bound qmv region), jumps at
    S=9-10 where dispatch crosses to the tiled GEMM path, then goes FLAT to
    S=32 (342->356 ms, and the plateau holds at 12k context). A greedy
    marginal rule walks to the cliff, sees one negative marginal, and never
    finds the far side — where a width-32 round at hot-prose acceptance
    prices out ~3x better per token than any depth in the qmv region. So
    depth() is an ARGMAX of expected-tokens-per-time over a candidate set
    that spans both regimes, and the deep candidates open only behind a
    full-accept streak (any reject collapses the tail estimate back to the
    prior, so cold or hard content never sees a deep round).
    """

    MAX_DEPTH = 31
    # Widths 1..6 cover the qmv region up to the fused-SDPA envelope; widths
    # 10/12/16/20/24/32 sample the flat tile plateau. Bounded on purpose:
    # each width is a Metal-compile family the engine warms at load, and the
    # E/T frontier between neighbors is nearly linear.
    CANDIDATES = (0, 1, 2, 3, 4, 5, 9, 11, 15, 19, 23, 31)
    STREAK_GATE = 2      # deep candidates need this many perfect rounds
    RETRY_GATE = 6       # perfect shallow rounds that re-open deep inference
    SHALLOW_MAX = 5      # deepest candidate below the cliff
    TAIL_SPLIT = 5       # positions >= this share the pooled tail estimate

    def __init__(self, max_depth: int, h: float, costs: Optional[list] = None):
        """`costs`, when given, is the measured round cost T(d) for depth
        d = 0..max_depth (any unit): T(d) = V(d+1) + d*H with the verify-width
        ladder as measured on THIS stack. Without it, a default table built
        from the measured M4 ladder shape (per-row to S=8, flat plateau
        after) seeds the choice; observe_cost() then corrects both online
        from real round wall-times, so a wrong seed only shapes the first
        rounds."""
        self.max_depth = max(0, min(max_depth, self.MAX_DEPTH))
        self.h = h
        # Seed cost RATIOS (unit-free). Observed wall-times override per
        # depth; unobserved depths are priced as seed * unit, with the unit
        # anchored at the most-observed depth. Never mix a raw observation
        # into the seed list directly: E/T argmax is scale-invariant only
        # while the whole table shares one scale, and a partially-observed
        # mixed-scale table silently prices the unobserved depths out (the
        # depth-1 lock bug this replaces).
        if costs is not None and len(costs) >= self.max_depth + 1:
            self._seed = [float(c) for c in costs]
        else:
            self._seed = [self._default_cost(d, h)
                          for d in range(self.MAX_DEPTH + 1)]
        self._obs: dict = {}      # depth -> (ema wall, count)
        self.ema = [0.85 * (0.98 ** i) for i in range(self.TAIL_SPLIT)]
        self.alpha = 0.15
        # Pooled tail acceptance for positions >= TAIL_SPLIT. Per-position
        # EMAs starve there — only deep rounds visit those positions, so each
        # gets single-sample evidence and one miss craters it (measured: one
        # rejected plateau round locked the plateau out for the whole turn).
        # Pooling is sound because conditional acceptance is near-flat in
        # position for a trained head on stationary content, and one deep
        # round contributes ~25 real samples instead of one.
        self.tail = 0.80          # pooled estimate (prior until observed)
        self.tail_n = 0           # samples folded in
        self.margin: Optional[float] = None   # pending top-2 logit margin
        self.streak = 0                       # consecutive fully-accepted rounds
        self._skips = 0                       # consecutive adaptive skips

    @staticmethod
    def _default_cost(d: int, h: float) -> float:
        """Round cost prior in serial-step units: measured M4 verify ladder
        (per-row through width 8, cliff at 9, flat plateau after) plus d head
        steps at h each. observe_cost() overwrites these with reality."""
        w = d + 1
        if w <= 2:
            v = 1.0 + 0.08 * (w - 1)
        elif w <= 8:
            v = 1.08 + 0.37 * (w - 2)      # ~25 ms / 66 ms per extra row
        else:
            v = 5.2 + 0.008 * (w - 9)      # the flat tile plateau
        return v + d * h

    def observe_cost(self, depth: int, cost: float) -> None:
        """Fold one measured round wall-time (seconds) into the per-depth
        observations. Round costs are noisy (rollback, detok, host work),
        hence the light EMA."""
        if not (0 <= depth < len(self._seed) and cost > 0):
            return
        ema, n = self._obs.get(depth, (cost, 0))
        self._obs[depth] = (ema + 0.25 * (cost - ema), n + 1)

    def _unit(self) -> float:
        """Seconds per seed-unit, anchored at the most-observed depth (the
        depth the loop actually lives at — a stable, representative sample;
        a one-off inflated round elsewhere cannot skew the whole table)."""
        if not self._obs:
            return 1.0
        d0 = max(self._obs, key=lambda d: self._obs[d][1])
        return self._obs[d0][0] / self._seed[d0]

    def cost(self, depth: int) -> float:
        ob = self._obs.get(depth)
        if ob is not None and ob[1] >= 2:
            return ob[0]
        return self._seed[depth] * self._unit()

    def _tail_p(self) -> float:
        """Acceptance estimate for positions >= TAIL_SPLIT. Real pooled
        observations once any deep round has run; before that, INFERENCE
        from the near positions on a qualifying full-accept streak (capped
        at 0.95 — the chain being ended by the schedule, not the head, is
        evidence about the tail, but never certainty), else the prior."""
        inference = min(0.95, sum(self.ema) / len(self.ema))
        if self.tail_n > 0:
            pooled = min(0.95, self.tail)
            # A long full-accept streak at shallow depth is fresh evidence
            # the content turned hot: let inference override stale pooled
            # pessimism so deep gets re-probed (one bad early probe must not
            # lock the plateau out for the rest of the turn — the probe's
            # ~25 new samples then speak for themselves).
            if self.streak >= self.RETRY_GATE:
                return max(pooled, inference)
            return pooled
        if self.streak >= self.STREAK_GATE:
            return inference
        return self.tail   # prior

    def _position_p(self, i: int, tail_p: float) -> float:
        # No continuous margin cap here: the arena's sigmoid(margin/2) gates
        # were fitted to a different logit scale and measured WRONG on this
        # stack — they capped p0 at ~0.7 while realized first-draft
        # acceptance was 0.988, locking the schedule at depth 1. Margins
        # participate only through the extreme-tie clamp in depth(); ongoing
        # calibration belongs to the EMAs, which are calibrated to realized
        # acceptance by construction.
        return self.ema[i] if i < self.TAIL_SPLIT else tail_p

    def depth(self) -> int:
        cap = self.max_depth
        if cap <= 0:
            return 0
        if self.margin is not None and self.margin < 0.25:
            # The pending token is a near coin-flip: the head's next draft is
            # close to a guess whatever the recent streak says. Probe at most
            # one draft; the EMAs stay in charge of everything else.
            cap = min(cap, 1)
        tail_p = self._tail_p()
        best_d, best_rate = 0, 1.0 / self.cost(0)
        reach, expected = 1.0, 0.0
        limit = min(cap, self.MAX_DEPTH)
        # Deep candidates open on a qualifying streak (first unlock, priced
        # by inference) or once the pooled tail holds real samples.
        gate_shallow = self.streak < self.STREAK_GATE and self.tail_n == 0
        d = 0
        for nxt in range(1, limit + 1):
            reach *= self._position_p(nxt - 1, tail_p)
            expected += reach
            d = nxt
            if d not in self.CANDIDATES:
                continue
            if gate_shallow and d > self.SHALLOW_MAX:
                break
            rate = (1.0 + expected) / self.cost(d)
            # 2% hysteresis toward shallower rounds: at equal throughput the
            # shallow round wastes less on a reject and stays inside the
            # fused-kernel envelope.
            if rate > best_rate * 1.02:
                best_d, best_rate = d, rate
        if best_d == 0:
            # An adaptive skip is free, but the EMAs only observe drafted
            # rounds — a hard stretch would otherwise lock drafting out for
            # the rest of the turn even after the content turns easy. Probe
            # depth 1 every 16th consecutive skip (~one EMA half-life).
            self._skips += 1
            if self._skips >= 16:
                self._skips = 0
                return 1
        else:
            self._skips = 0
        return best_d

    def record(self, proposed: int, accepted: int, stopped_early: bool) -> None:
        """Fold one round's outcome into the per-position EMAs. Positions
        before `accepted` observed a success; the position AT `accepted`
        observed a failure only when the walk actually rejected there (not
        when the round ended on a committed stop token); deeper positions
        observe nothing. A FULLY accepted round transfers bounded optimism
        (toward 0.95, never past it) to the first unreached position."""
        a = self.alpha
        e = self.ema
        for i in range(min(accepted, len(e))):
            e[i] += a * (1.0 - e[i])
        if accepted < proposed and not stopped_early and accepted < len(e):
            e[accepted] += a * (0.0 - e[accepted])
        elif accepted == proposed and proposed > 0 and accepted < len(e):
            if e[accepted] < 0.95:
                e[accepted] += a * (0.95 - e[accepted])
        # Pooled tail: positions TAIL_SPLIT..accepted-1 observed successes;
        # the position AT `accepted` observed the failure when the walk
        # genuinely rejected there. Batch-fold with per-sample weight so one
        # deep round's ~25 samples count like 25 observations, not one.
        ts = self.TAIL_SPLIT
        if proposed > ts:
            succ = max(0, min(accepted, proposed) - ts)
            fail = 1 if (ts <= accepted < proposed and not stopped_early) else 0
            n = succ + fail
            if n:
                obs = succ / n
                w = 1.0 - (1.0 - 0.05) ** n
                self.tail += w * (obs - self.tail)
                self.tail_n += n
        self.streak = self.streak + 1 if (proposed > 0 and accepted == proposed) \
            else 0


def build(args: Any):
    """Construct an (unquantized) MTP head module for a qwen3_5 TextModelArgs."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import DecoderLayer

    class _MTPHead(nn.Module):
        def __init__(self, args):
            super().__init__()
            h, eps = args.hidden_size, args.rms_norm_eps
            self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=eps)
            self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=eps)
            self.fc = nn.Linear(2 * h, h, bias=False)
            # layer_idx chosen so DecoderLayer builds its FULL-attention form
            # ((idx+1) % full_attention_interval == 0); the head has no GDN.
            self.layers = [DecoderLayer(args, args.full_attention_interval - 1)]
            self.norm = nn.RMSNorm(h, eps=eps)

        def __call__(self, embeds, hidden, cache=None, mask=None):
            x = mx.concatenate(
                [self.pre_fc_norm_embedding(embeds),
                 self.pre_fc_norm_hidden(hidden)], axis=-1)
            x = self.fc(x)
            x = self.layers[0](x, mask=mask, cache=cache)
            return self.norm(x)

        def make_cache(self):
            from mlx_lm.models.cache import KVCache
            return KVCache()

    return _MTPHead(args)


def _sidecar_candidates(model_dir: str) -> list:
    out = []
    env = os.environ.get("CHAD_MTP_PATH")
    if env:
        out.append(os.path.expanduser(env))
    out.append(os.path.join(model_dir, "mtp.safetensors"))
    out.append(os.path.join(
        _SIDECAR_DIR, os.path.basename(os.path.normpath(model_dir)) + ".safetensors"))
    return out


def _mtp_layers(model_dir: str) -> int:
    """mtp_num_hidden_layers from the on-disk config (0 when absent/unreadable)."""
    import json
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        return int(tc.get("mtp_num_hidden_layers") or 0)
    except Exception:  # noqa: BLE001
        return 0


def _quant_defaults(model_dir: str) -> tuple:
    """(group_size, bits) the base conversion used, so the head matches."""
    import json
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            q = json.load(f).get("quantization") or {}
        return int(q.get("group_size", 64)), int(q.get("bits", 4))
    except Exception:  # noqa: BLE001 — sane default beats a failed load
        return 64, 4


def load_head(model: Any, model_dir: str) -> Optional[Any]:
    """Load the MTP head for a loaded qwen3_5 model, or None (no sidecar found,
    wrong arch, or any failure — MTP is a pure speed feature, never load-bearing)."""
    try:
        import mlx.core as mx
        import mlx.nn as nn

        args = getattr(getattr(model, "language_model", None), "args", None)
        if args is None or _mtp_layers(model_dir) != 1:
            # mlx-lm's TextModelArgs drops unknown config keys, so the head
            # count must come from the on-disk config, not the parsed args.
            return None
        path = next((p for p in _sidecar_candidates(model_dir)
                     if p and os.path.isfile(p)), None)
        if path is None:
            return None
        weights = mx.load(path)
        islands = {k: v for k, v in weights.items()
                   if k.startswith("precision_islands.")}
        if islands:
            weights = {k: v for k, v in weights.items()
                       if not k.startswith("precision_islands.")}
        meta_gs, meta_bits = _read_meta(path)
        if not meta_bits and "fc.scales" in weights:
            # Foreign sidecar (e.g. an arena head artifact) without build
            # metadata: infer the packing from fc's shapes. fc is (H, 2H):
            # 2H / packed-cols u32 values per word -> bits, 2H / scale-cols
            # -> group size.
            in_dim = 2 * int(args.hidden_size)
            meta_bits = 32 // (in_dim // int(weights["fc.weight"].shape[1]))
            meta_gs = in_dim // int(weights["fc.scales"].shape[1])
        head = build(args)
        if meta_bits:
            nn.quantize(head, group_size=meta_gs, bits=meta_bits)
        head.load_weights(list(weights.items()))
        n_isl = _install_islands(head, islands) if islands else 0
        head.eval()
        mx.eval(head.parameters())
        log.info("MTP head loaded (%s, %s%s)", path,
                 f"{meta_bits}-bit g{meta_gs}" if meta_bits else "unquantized",
                 f", {n_isl} bf16 precision-island rows" if n_isl else "")
        return head
    except Exception as e:  # noqa: BLE001 — never break model load over MTP
        log.warning("MTP head load failed (%s); decoding without MTP", e)
        return None


class DraftReadout:
    """Cheap argmax for the DRAFT chain: a 2-bit shadow of lm_head proposes a
    top-K shortlist, the real 4-bit rows exactly rerank it.

    The chained draft step's cost is dominated by the full-vocab lm_head
    readout (~3.0 ms of the 4.3 ms step on the 27B: 248320 x 5120 at 4-bit is
    ~620 MB of weight traffic per draft). The 2-bit shadow halves that
    stream, argpartition keeps the K best, and the exact rerank over K rows
    costs microseconds — measured 1.33x on the readout with 93% argmax
    agreement on RANDOM logits (worst case; real logits are peaky, so the
    true argmax is in a 2-bit top-32 essentially always). DRAFT-SIDE ONLY:
    a shortlist miss is just a draft the verify rejects; the emitted-token
    distribution is untouched.
    """

    K = 32

    def __init__(self, lm_head: Any, w2: Any, s2: Any, b2: Any):
        self._lm = lm_head
        self._w2, self._s2, self._b2 = w2, s2, b2

    def argmax_token(self, hidden: Any) -> Any:
        """(1, 1, H) hidden -> scalar token id array (lazy)."""
        import mlx.core as mx
        lm = self._lm
        lg2 = mx.quantized_matmul(
            hidden, self._w2, scales=self._s2, biases=self._b2,
            transpose=True, group_size=lm.group_size, bits=2)[0, -1]
        v = lg2.shape[-1]
        idx = mx.argpartition(lg2, v - self.K)[v - self.K:]
        ex = mx.quantized_matmul(
            hidden, lm["weight"][idx], scales=lm["scales"][idx],
            biases=lm["biases"][idx], transpose=True,
            group_size=lm.group_size, bits=lm.bits)[0, -1]
        return idx[mx.argmax(ex)]


def build_draft_readout(model: Any, model_dir: str) -> Optional[Any]:
    """2-bit lm_head shadow for DraftReadout, built once and cached beside
    the MTP sidecars. None when ineligible (tied embeddings, non-4-bit or
    non-affine lm_head, tiny vocab) or on any failure — the shortlist is a
    pure speed feature, never load-bearing."""
    try:
        import mlx.core as mx

        lm = getattr(model, "language_model", model)
        if getattr(lm.args, "tie_word_embeddings", False):
            return None
        head = getattr(lm, "lm_head", None)
        if head is None or not hasattr(head, "group_size"):
            return None
        if getattr(head, "mode", "affine") != "affine" or head.bits != 4:
            return None
        gs = head.group_size
        w4, s4, b4 = head["weight"], head["scales"], head.get("biases")
        if b4 is None:
            return None
        V = w4.shape[0]
        if V < DraftReadout.K * 64:   # shortlist over a tiny vocab is noise
            return None

        os.makedirs(_SIDECAR_DIR, exist_ok=True)
        cache = os.path.join(
            _SIDECAR_DIR,
            os.path.basename(os.path.normpath(model_dir)) + "-draft2b.safetensors")
        if os.path.isfile(cache):
            t = mx.load(cache)
            w2, s2, b2 = t["weight"], t["scales"], t["biases"]
        else:
            # Chunked dequantize -> requantize (a full-vocab dequant is a
            # ~2.5 GB transient; 32k-row blocks keep it to ~340 MB).
            step = 32768
            ws, ss, bs = [], [], []
            for r0 in range(0, V, step):
                r1 = min(V, r0 + step)
                deq = mx.dequantize(
                    w4[r0:r1], scales=s4[r0:r1], biases=b4[r0:r1],
                    group_size=gs, bits=4)
                qw, qs, qb = mx.quantize(deq, group_size=gs, bits=2)
                mx.eval(qw, qs, qb)
                ws.append(qw); ss.append(qs); bs.append(qb)
            w2 = mx.concatenate(ws); s2 = mx.concatenate(ss)
            b2 = mx.concatenate(bs)
            mx.eval(w2, s2, b2)
            mx.save_safetensors(
                cache, {"weight": w2, "scales": s2, "biases": b2},
                metadata={"bits": "2", "group_size": str(gs),
                          "source": "lm_head-4bit-requant"})
        log.info("draft shortlist readout ready (2-bit lm_head shadow, "
                 "top-%d rerank)", DraftReadout.K)
        return DraftReadout(head, w2, s2, b2)
    except Exception as e:  # noqa: BLE001 — never break load over a speedup
        log.warning("draft shortlist build failed (%s); full readout", e)
        return None


def _install_islands(head: Any, islands: dict) -> int:
    """Wrap the head's q/k/v projections with exact-row precision islands.

    A sidecar may carry `precision_islands.{q,k,v}.{weight,indices}`: selected
    output rows of the proposal head's attention projections kept in bf16
    (the rows with the largest fp32 reconstruction error under the packed
    4-bit quantization — a draft-quality recovery, worth ~real acceptance at
    depth). The wrapped projection runs the packed QMV unchanged, computes the
    island rows exactly, and overwrites just those outputs. Head-only: the
    main model never sees these arrays. Returns the number of island rows."""
    import mlx.core as mx
    import mlx.nn as nn

    class _IslandLinear(nn.Module):
        def __init__(self, inner, w, idx):
            super().__init__()
            self.inner = inner
            self.island_w = w
            self.island_idx = idx.astype(mx.int32)

        def __call__(self, x):
            y = self.inner(x)
            w = self.island_w
            if w.dtype != x.dtype:
                # One-time cast to the compute dtype (a per-call astype in the
                # matmul would re-launch the cast kernel every draft step).
                w = w.astype(x.dtype)
                self.island_w = w
            exact = x @ w.T
            idx = self.island_idx.reshape((1,) * (y.ndim - 1) + (-1,))
            return mx.put_along_axis(y, idx, exact, axis=-1)

    attn = head.layers[0].self_attn
    total = 0
    for name in ("q", "k", "v"):
        w = islands.get(f"precision_islands.{name}.weight")
        idx = islands.get(f"precision_islands.{name}.indices")
        if w is None or idx is None:
            continue
        if w.shape[0] != idx.shape[0]:
            raise ValueError(
                f"precision island {name}: weight rows {w.shape[0]} != "
                f"indices {idx.shape[0]}")
        proj = f"{name}_proj"
        setattr(attn, proj, _IslandLinear(getattr(attn, proj), w, idx))
        total += int(w.shape[0])
    return total


def _read_meta(path: str) -> tuple:
    """(group_size, bits) recorded at build time; (0, 0) = unquantized."""
    import json
    try:
        # safetensors header: 8-byte little-endian length + JSON with __metadata__
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        meta = hdr.get("__metadata__") or {}
        return int(meta.get("group_size", 0)), int(meta.get("bits", 0))
    except Exception:  # noqa: BLE001
        return 0, 0


def build_sidecar(shard_path: str, model_dir: str, out_path: Optional[str] = None,
                  bits: Optional[int] = None, group_size: Optional[int] = None) -> str:
    """Extract `mtp.*` tensors from an original-checkpoint shard, quantize them
    to match the base conversion, and save the sidecar. Returns the output path."""
    import json

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    from mlx_lm.models.qwen3_5 import ModelArgs, TextModelArgs
    targs = TextModelArgs.from_dict(
        ModelArgs.from_dict(cfg).text_config)
    gs0, bits0 = _quant_defaults(model_dir)
    gs, b = group_size or gs0, bits or bits0

    raw = mx.load(shard_path)
    mtp = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
    if not mtp:
        raise ValueError(f"no mtp.* tensors in {shard_path}")
    # Original Qwen3.5-family checkpoints store RMSNorm weights ZERO-CENTERED
    # (gamma = 1 + w); mlx's RMSNorm is plain-gamma. mlx-lm shifts the main
    # model's norms at conversion (qwen3_5.TextModel.sanitize) — the head's
    # norms need the same +1.0 or every normalization in it is off by ~1
    # (measured: teacher-forced acceptance 0% unshifted, 73% shifted).
    norm_sfx = ("input_layernorm.weight", "post_attention_layernorm.weight",
                "q_norm.weight", "k_norm.weight")
    mtp = {k: (v + 1.0
               if (k.endswith(norm_sfx)
                   or k in ("norm.weight", "pre_fc_norm_embedding.weight",
                            "pre_fc_norm_hidden.weight"))
               else v)
           for k, v in mtp.items()}
    head = build(targs)
    head.load_weights(list(mtp.items()))
    nn.quantize(head, group_size=gs, bits=b)
    mx.eval(head.parameters())

    if out_path is None:
        os.makedirs(_SIDECAR_DIR, exist_ok=True)
        out_path = os.path.join(
            _SIDECAR_DIR,
            os.path.basename(os.path.normpath(model_dir)) + ".safetensors")
    flat = dict(tree_flatten(head.parameters()))
    mx.save_safetensors(out_path, flat,
                        metadata={"group_size": str(gs), "bits": str(b)})
    return out_path


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse
    ap = argparse.ArgumentParser(
        description="Build a quantized MTP sidecar from an original shard")
    ap.add_argument("shard", help="safetensors shard containing mtp.* (bf16)")
    ap.add_argument("--model", required=True,
                    help="base (quantized) model dir the sidecar pairs with")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    a = ap.parse_args(argv)
    out = build_sidecar(a.shard, a.model, a.out, a.bits, a.group_size)
    print(out)


if __name__ == "__main__":  # pragma: no cover
    main()
