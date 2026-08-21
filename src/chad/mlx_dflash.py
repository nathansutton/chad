"""DFlash2 block-diffusion drafter (Qwen3.8-class): load + forward + target tap.

A DFlash drafter is a small external model (1.92B for Qwen3.8-27B) that proposes a
whole BLOCK of tokens in ONE forward. It reads the target's residual stream at a few
tapped layers (``target_layer_ids``), fuses them with one projection (``fc``) into
per-token context rows, and injects those rows straight into its own KV cache — the
prompt is never re-read by the drafter. Each round it then decodes
``[anchor, MASK, MASK, ...]`` against that context; every mask slot becomes a draft.
DFlash2 adds two trained parts on the same backbone: a 2-tap grouped dynamic
convolution around each sublayer, and a *candidate selector* — the target head's
top-K per slot plus a bilinear score over adjacent-slot pairs, walked from the
verified anchor so the block is one coherent path rather than K independent argmaxes.
The drafter shares the target's embedding and ``lm_head``.

One drafter forward proposes the whole block, and acceptance holds to depth 7 —
where a chained 1-token-at-a-time drafter decays past 3. The verify side is one
batched target forward with exact rejection sampling, so the committed output is
what plain decoding would have produced at the same sampling settings.

Provenance
----------
The model classes are an MLX port of z-lab's DFlash (``dflash/model_mlx.py``, MIT,
Copyright (c) 2026 Z Lab; Chen et al., "DFlash: Block Diffusion for Flash Speculative
Decoding", arXiv:2602.06036) with the DFlash2 components (grouped dynamic conv,
candidate selector) as ported in ARahim3/mlx-dspark (``dflash_model.py``, MIT,
Copyright (c) 2026 ARahim3) from the SGLang reference implementation. Adapted here:
quantized codebooks, a cached quantized sidecar, chad's loader/tap conventions.

  Permission is hereby granted, free of charge, to any person obtaining a copy of
  this software and associated documentation files (the "Software"), to deal in the
  Software without restriction, including without limitation the rights to use,
  copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
  Software, and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions: The above copyright notice and this
  permission notice shall be included in all copies or substantial portions of the
  Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

The drafter ships QUANTIZED, inside the model repo, as ``<model_dir>/dflash/``
(config.json + model.safetensors, 4-bit group-64, ~1.1 GB resident) — one
download, nothing built at runtime. $CHAD_DFLASH_PATH overrides it with a local
dir holding either a built sidecar or an unquantized HF checkpoint (quantized on
load). No bundle and no override means no drafter: decoding runs serial.

Build a sidecar for a new target from its bf16 DFlash checkpoint with
``uv run python -m chad.mlx_dflash <hf_checkpoint_dir> --out <model_dir>/dflash``.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .diag import log

# The drafter sidecar's subdirectory inside a model weights dir.
_BUNDLE = "dflash"

# Residual-stream tap. When an engine arms a dict here, every tapped target layer
# (see install_tap) writes its OUTPUT hidden [B, S, H] under its layer index for the
# duration of the forward. None during plain decode and prefill that no drafter
# watches — the tapped layers then cost one attribute read each.
TAP: Optional[dict] = None


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: Optional[dict] = None
    layer_types: tuple = field(default_factory=tuple)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    # DFlash2 (0 = absent -> plain DFlash behavior)
    selector_rank: int = 0
    selector_top_k: int = 0
    conv_kernel_size: int = 0
    conv_group_size: int = 16
    output_multiplier: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "DFlashConfig":
        rope = cfg.get("rope_parameters") or {}
        dfc = cfg.get("dflash_config") or {}
        get = lambda k, d=None: dfc.get(k, cfg.get(k, d))  # noqa: E731
        n_layers = int(cfg["num_hidden_layers"])
        lt = cfg.get("layer_types")
        if not lt:
            lt = (("sliding_attention" if cfg.get("use_sliding_window")
                   and cfg.get("sliding_window") else "full_attention"),) * n_layers
        scaling = cfg.get("rope_scaling")
        if scaling is None and rope.get("rope_type", "default") != "default":
            scaling = dict(rope)
        return cls(
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=n_layers,
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg.get("head_dim")
                         or cfg["hidden_size"] // cfg["num_attention_heads"]),
            intermediate_size=int(cfg["intermediate_size"]),
            vocab_size=int(cfg["vocab_size"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", rope.get("rope_theta", 1e6))),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 262144)),
            block_size=int(get("block_size")),
            target_layer_ids=tuple(int(i) for i in get("target_layer_ids")),
            num_target_layers=int(cfg.get("num_target_layers", 0)),
            mask_token_id=int(get("mask_token_id", 0)),
            rope_scaling=scaling,
            layer_types=tuple(lt),
            sliding_window=(int(cfg["sliding_window"])
                            if cfg.get("sliding_window") else None),
            final_logit_softcapping=get("final_logit_softcapping"),
            selector_rank=int(get("selector_rank", 0) or 0),
            selector_top_k=int(get("selector_top_k", 0) or 0),
            conv_kernel_size=int(get("conv_kernel_size", 0) or 0),
            conv_group_size=int(get("conv_group_size", 16) or 16),
            output_multiplier=float(get("output_multiplier", 1.0) or 1.0),
        )


def build(config: DFlashConfig):
    """Construct an (unquantized, lazily initialized) drafter module."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_causal_mask
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx_lm.models.qwen3 import MLP
    from mlx_lm.models.rope_utils import initialize_rope

    class DFlashAttention(nn.Module):
        def __init__(self, cfg: DFlashConfig, layer_idx: int):
            super().__init__()
            dim = cfg.hidden_size
            self.n_heads = cfg.num_attention_heads
            self.n_kv_heads = cfg.num_key_value_heads
            self.scale = cfg.head_dim ** -0.5
            self.is_sliding = cfg.layer_types[layer_idx] == "sliding_attention"
            self.sliding_window = cfg.sliding_window if self.is_sliding else None
            self.q_proj = nn.Linear(dim, self.n_heads * cfg.head_dim, bias=False)
            self.k_proj = nn.Linear(dim, self.n_kv_heads * cfg.head_dim, bias=False)
            self.v_proj = nn.Linear(dim, self.n_kv_heads * cfg.head_dim, bias=False)
            self.o_proj = nn.Linear(self.n_heads * cfg.head_dim, dim, bias=False)
            self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

        def append_ctx(self, x_ctx, rope, cache):
            """Project + rope + append context rows into this layer's cache; returns
            the cache's full (keys, values). A sliding layer keeps only the last
            ``window - 1`` rows and advances the offset past the skipped ones."""
            B, S, _ = x_ctx.shape
            if self.is_sliding:
                keep = self.sliding_window - 1
                if keep < S:
                    skip = S - keep
                    x_ctx = x_ctx[:, skip:]
                    S = keep
                    cache.offset += skip
            k = self.k_proj(x_ctx).reshape(B, S, self.n_kv_heads, -1)
            v = self.v_proj(x_ctx).reshape(B, S, self.n_kv_heads, -1)
            k = self.k_norm(k).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            k = rope(k, offset=cache.offset)
            return cache.update_and_fetch(k, v)

        def __call__(self, x, x_ctx, rope, cache):
            B, L, _ = x.shape
            if x_ctx is not None:
                keys, values = self.append_ctx(x_ctx, rope, cache)
                ctx_len = keys.shape[2]
            else:
                # No context this turn yet (nothing prefilled): the block attends
                # itself only. Only legal on an empty cache.
                assert cache.offset == 0, "DFlash: ctx-less block on a non-empty cache"
                keys = values = None
                ctx_len = 0
            q = self.q_proj(x).reshape(B, L, self.n_heads, -1)
            pk = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)
            pv = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)
            q = rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=cache.offset)
            pk = rope(self.k_norm(pk).transpose(0, 2, 1, 3), offset=cache.offset)
            pv = pv.transpose(0, 2, 1, 3)
            if keys is not None:
                keys = mx.concatenate([keys, pk], axis=2)
                values = mx.concatenate([values, pv], axis=2)
            else:
                keys, values = pk, pv
            mask = None
            if self.is_sliding:
                mask = ("causal" if ctx_len + L <= self.sliding_window
                        else create_causal_mask(L, offset=ctx_len,
                                                window_size=self.sliding_window))
            out = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=self.scale, mask=mask)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    class DFlashGroupedConv(nn.Module):
        """DFlash2: grouped dynamic depthwise K-tap convolution across one block.
        ``prepare`` convolves a sublayer's input and returns the coefficient set
        ``finish`` applies to its output — both sets from ONE projection of the
        input. Coefficient = learned per-channel base + content-adaptive per-group
        correction. Block rows only; the zero-padded tap shift IS the block-boundary
        mask (the anchor has no in-block predecessor)."""

        def __init__(self, hidden_size: int, taps: int, group_size: int):
            super().__init__()
            if hidden_size % group_size:
                raise ValueError("conv_group_size must divide hidden_size")
            self.taps = taps
            self.group_size = group_size
            self.num_groups = hidden_size // group_size
            self.base_kernel = mx.concatenate(
                [mx.ones((2, 1, hidden_size)), mx.zeros((2, taps - 1, hidden_size))],
                axis=1)
            self.kernel_projection = nn.Linear(
                hidden_size, 2 * taps * self.num_groups, bias=False)

        def _convolve(self, x, delta, side: int):
            B, L, H = x.shape
            xg = x.reshape(B, L, self.num_groups, self.group_size)
            base = self.base_kernel[side].reshape(
                1, 1, self.taps, self.num_groups, self.group_size).astype(x.dtype)
            coeff = base + delta[..., None]
            out = coeff[:, :, 0] * xg
            for t in range(1, self.taps):
                shifted = mx.pad(xg[:, :-t], ((0, 0), (t, 0), (0, 0), (0, 0)))
                out = out + coeff[:, :, t] * shifted
            return out.reshape(B, L, H)

        def prepare(self, x):
            coeff = self.kernel_projection(x).reshape(
                *x.shape[:-1], 2, self.taps, self.num_groups)
            return self._convolve(x, coeff[..., 0, :, :], 0), coeff[..., 1, :, :]

        def finish(self, y, delta):
            return self._convolve(y, delta, 1)

    class CandidateSelector(nn.Module):
        """DFlash2: scores the K x K transitions between adjacent mask slots, then
        walks them. ``scores[slot, p, c] = unary[slot, c] + <A[pred] * proj(h_slot),
        B[c]>``; slot 0's predecessors are all the verified anchor token, slot s's
        are slot s-1's candidates. Codebooks are embeddings so they quantize."""

        def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
            super().__init__()
            self.top_k = top_k
            self.predecessor_codebook = nn.Embedding(vocab_size, rank)
            self.successor_codebook = nn.Embedding(vocab_size, rank)
            self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

        def lattice(self, candidate_ids, unary_logits, hidden, anchor_id: int):
            k = candidate_ids.shape[1]
            h = self.hidden_projection(hidden)                               # [g, r]
            anchor = mx.full((1, k), anchor_id, dtype=candidate_ids.dtype)
            pred_ids = mx.concatenate([anchor, candidate_ids[:-1]], axis=0)  # [g, K]
            pre = self.predecessor_codebook(pred_ids) * h[:, None, :]        # [g, K, r]
            suc = self.successor_codebook(candidate_ids)                     # [g, K, r]
            bilinear = pre @ suc.transpose(0, 2, 1)                          # [g, K, K]
            return (unary_logits[:, None, :].astype(mx.float32)
                    + bilinear.astype(mx.float32))

        def walk_greedy(self, scores, candidate_ids):
            """Chain-argmax path -> drafted ids [g], fully in-graph."""
            g = scores.shape[0]
            idx = mx.argmax(scores[0, 0])
            picks = [candidate_ids[0, idx]]
            for s in range(1, g):
                idx = mx.argmax(scores[s, idx])
                picks.append(candidate_ids[s, idx])
            return mx.stack(picks)

        def walk_sampled(self, scores, candidate_ids, uniforms, temperature: float):
            """Sampled walk -> (drafted ids [g], q rows [g, K] they were drawn from)."""
            g, k = candidate_ids.shape
            t = max(float(temperature), 1e-5)
            init = mx.softmax(scores[0, 0] / t, axis=-1)
            idx = mx.minimum((uniforms[0] >= mx.cumsum(init)).sum(), k - 1)
            picks, q_rows = [candidate_ids[0, idx]], [init]
            if g > 1:
                trans = mx.softmax(scores[1:] / t, axis=-1)                  # [g-1, K, K]
                maps = mx.minimum(
                    (uniforms[1:, None, None] >= mx.cumsum(trans, axis=-1)).sum(axis=-1),
                    k - 1)
                for s in range(1, g):
                    q_rows.append(trans[s - 1, idx])
                    idx = maps[s - 1, idx]
                    picks.append(candidate_ids[s, idx])
            return mx.stack(picks), mx.stack(q_rows)

    class DFlashDecoderLayer(nn.Module):
        def __init__(self, cfg: DFlashConfig, layer_idx: int):
            super().__init__()
            self.self_attn = DFlashAttention(cfg, layer_idx)
            self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
            self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size,
                                                       eps=cfg.rms_norm_eps)
            if cfg.conv_kernel_size:
                self.attention_conv = DFlashGroupedConv(
                    cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)
                self.mlp_conv = DFlashGroupedConv(
                    cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)
            else:
                self.attention_conv = None
                self.mlp_conv = None

        def __call__(self, x, x_ctx, rope, cache):
            a = self.input_layernorm(x)
            if self.attention_conv is not None:
                a, oc = self.attention_conv.prepare(a)
            attn = self.self_attn(a, x_ctx, rope, cache)
            if self.attention_conv is not None:
                attn = self.attention_conv.finish(attn, oc)
            h = x + attn
            m = self.post_attention_layernorm(h)
            if self.mlp_conv is not None:
                m, oc = self.mlp_conv.prepare(m)
            y = self.mlp(m)
            if self.mlp_conv is not None:
                y = self.mlp_conv.finish(y, oc)
            return h + y

    class DFlashDraftModel(nn.Module):
        def __init__(self, cfg: DFlashConfig):
            super().__init__()
            self.config = cfg
            self.fc = nn.Linear(len(cfg.target_layer_ids) * cfg.hidden_size,
                                cfg.hidden_size, bias=False)
            self.hidden_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.layers = [DFlashDecoderLayer(cfg, i)
                           for i in range(cfg.num_hidden_layers)]
            self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.rope = initialize_rope(
                dims=cfg.head_dim, base=cfg.rope_theta, traditional=False,
                scaling_config=cfg.rope_scaling,
                max_position_embeddings=cfg.max_position_embeddings)
            self.candidate_selector = (
                CandidateSelector(cfg.hidden_size, cfg.vocab_size,
                                  cfg.selector_rank, cfg.selector_top_k)
                if cfg.selector_rank else None)
            # Bound by the engine: the target's embedding module and a logits
            # function over post-norm hidden (handles tied embeddings).
            self._embed = None
            self._logits = None

        def bind(self, embed, logits_fn):
            self._embed = embed
            self._logits = logits_fn
            return self

        def make_cache(self):
            out = []
            for lt in self.config.layer_types:
                if lt == "sliding_attention":
                    out.append(RotatingKVCache(
                        max_size=self.config.sliding_window - 1, keep=0))
                else:
                    out.append(KVCache())
            return out

        def project_ctx(self, fused):
            """Fused target hidden [B, S, taps*H] -> context rows [B, S, H]."""
            return self.hidden_norm(self.fc(fused))

        def append_ctx(self, rows, cache):
            """Append PROJECTED context rows to every layer's cache (no drafting)."""
            for layer, c in zip(self.layers, cache):
                layer.self_attn.append_ctx(rows, self.rope, c)

        def forward_hidden(self, inputs, fused, cache, logits_start: int = 0):
            """Backbone forward -> post-final-norm hidden [B, L - logits_start, H].
            ``fused`` (pending context rows, or None) is appended first."""
            h = self._embed(inputs)
            h_ctx = self.project_ctx(fused) if fused is not None else None
            for layer, c in zip(self.layers, cache):
                h = layer(h, h_ctx, self.rope, c)
            if logits_start:
                h = h[:, logits_start:]
            return self.norm(h)

        def _unary(self, logits):
            lg = logits.astype(mx.float32)
            if self.config.output_multiplier != 1.0:
                lg = lg * self.config.output_multiplier
            cap = self.config.final_logit_softcapping
            if cap:
                lg = mx.tanh(lg / cap) * cap
            return lg

        def select_block(self, block, fused, cache, *, cap: int, anchor_id: int,
                         uniforms=None, temperature: float = 1.0):
            """DFlash2 drafting: backbone forward, target-head top-K per mask slot,
            selector walk from the anchor. Greedy (``uniforms`` None) returns
            ``(draft_ids [cap], None, None)`` fully in-graph; sampled returns
            ``(draft_ids [cap], candidate_ids [cap, K], q_rows [cap, K])``."""
            sel = self.candidate_selector
            hidden = self.forward_hidden(block, fused, cache, logits_start=1)[0][:cap]
            logits = self._logits(hidden)[..., : self.config.vocab_size]
            if sel is None:
                if uniforms is None:
                    return mx.argmax(logits, axis=-1), None, None
                raise NotImplementedError("sampled drafting needs a DFlash2 selector")
            k = sel.top_k
            cand = mx.argpartition(logits, kth=-k, axis=-1)[:, -k:].astype(mx.int32)
            unary = self._unary(mx.take_along_axis(logits, cand, axis=-1))
            scores = sel.lattice(cand, unary, hidden, anchor_id)
            if uniforms is None:
                return sel.walk_greedy(scores, cand), None, None
            ids, q_rows = sel.walk_sampled(scores, cand, uniforms, temperature)
            return ids, cand, q_rows

    return DFlashDraftModel(config)


# -- verified-width schedule ---------------------------------------------------

# Measured round cost T(d) for a DFlash round of verified width d, in serial-step
# units (M4 Pro, the shipped 3-bit 27B, a 57 ms serial step, mlx_qmm_mma on from
# M=5): one drafter forward (~0.2 step, flat in d) plus the verify. Rows 2..4
# pay the stock quantized_matmul ladder; rows 5..8 are FLAT under the MMA kernel
# (~125 ms, 2.2 steps), and the last stock row (width 3, M=4) is dearer than the
# first kernel row, so past width 3 more width is free and the argmax lands on
# the full block whenever acceptance carries it. observe_cost() re-anchors the
# scale online; the SHAPE is what has to be right (a per-row seed priced the
# first extra row too cheaply and collapsed the schedule on medium acceptance).
BLOCK_ROUND_COSTS = (1.0, 1.76, 1.93, 2.30, 2.18, 2.20, 2.19, 2.20)


class WidthPolicy:
    """Cost-model width schedule: choose the verified width that maximizes
    expected committed tokens per unit round time.

    The block is always drafted WHOLE in one drafter forward; this picks how
    many of its proposals to verify. A round of width d+1 costs T(d) (one
    drafter step plus the verify ladder, BLOCK_ROUND_COSTS) and commits
    E[tokens](d) = 1 + sum_{k=1..d} prod_{i<k} p_i, where p_i is the
    estimated acceptance of draft position i GIVEN the prefix before it was
    accepted. The p_i are per-position EMAs seeded with an optimistic,
    gently decaying prior (the first rounds should draft, not stall); a
    fully accepted round also transfers bounded optimism to the first
    unreached position, so the full block becomes reachable inside a short
    window. On hot prose this runs to the cap; where acceptance drops
    (sampled decoding on hard content) it collapses to 1 and then to 0 — an
    adaptive skip costs exactly what a serial step costs, so a bad regime
    degrades to serial decoding rather than below it.

    The pending top-2 logit margin (of the row that produced the pending
    token) additionally caps p_0/p_1: a near-tie next token is exactly where
    the drafter is about to be wrong, whatever its recent streak says.

    depth() is an ARGMAX over every width up to the cap rather than a greedy
    marginal walk: the cost ladder is not convex (rows 6..8 are flat under
    the MMA kernel), and a marginal rule stops at the first negative step
    instead of finding the flat region past it.
    """

    MAX_DEPTH = 7        # the block cap: block_size - 1
    CANDIDATES = tuple(range(8))
    STREAK_GATE = 2      # full-accept rounds before the tail may be inferred
    RETRY_GATE = 6       # perfect rounds that re-open a stale pooled tail
    TAIL_SPLIT = 5       # positions >= this share the pooled tail estimate

    def __init__(self, max_depth: int, costs: list):
        """`costs` is the measured round cost T(d) for width d+1, d = 0..
        MAX_DEPTH (any unit), as measured on THIS stack. observe_cost()
        corrects it online from real round wall-times, so a wrong seed only
        shapes the first rounds."""
        self.max_depth = max(0, min(max_depth, self.MAX_DEPTH))
        # Seed cost RATIOS (unit-free). Observed wall-times override per
        # depth; unobserved depths are priced as seed * unit, with the unit
        # anchored at the most-observed depth. Never mix a raw observation
        # into the seed list directly: E/T argmax is scale-invariant only
        # while the whole table shares one scale, and a partially-observed
        # mixed-scale table silently prices the unobserved depths out.
        self._seed = [float(c) for c in costs]
        self._obs: dict = {}      # depth -> (ema wall, count)
        self.ema = [0.85 * (0.98 ** i) for i in range(self.TAIL_SPLIT)]
        self.alpha = 0.15
        # Pooled tail acceptance for positions >= TAIL_SPLIT. Per-position
        # EMAs starve there — only wide rounds visit those positions, so each
        # gets single-sample evidence and one miss craters it (measured: one
        # rejected wide round locked width out for the whole turn). Pooling is
        # sound because conditional acceptance is near-flat in position for a
        # trained drafter on stationary content.
        self.tail = 0.80          # pooled estimate (prior until observed)
        self.tail_n = 0           # samples folded in
        self.margin: Optional[float] = None   # pending top-2 logit margin
        self.streak = 0                       # consecutive fully-accepted rounds
        self._skips = 0                       # consecutive adaptive skips

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
        observations once any wide round has run; before that, INFERENCE
        from the near positions on a qualifying full-accept streak (capped
        at 0.95 — the block being cut short by the schedule, not by the
        drafter, is evidence about the tail, but never certainty), else the
        prior."""
        inference = min(0.95, sum(self.ema) / len(self.ema))
        if self.tail_n > 0:
            pooled = min(0.95, self.tail)
            # A long full-accept streak at a narrow width is fresh evidence
            # the content turned hot: let inference override stale pooled
            # pessimism so width gets re-probed (one bad early probe must not
            # lock the full block out for the rest of the turn).
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
            # The pending token is a near coin-flip: the drafter's next block
            # is close to a guess whatever the recent streak says. Probe at
            # most one draft; the EMAs stay in charge of everything else.
            cap = min(cap, 1)
        tail_p = self._tail_p()
        best_d, best_rate = 0, 1.0 / self.cost(0)
        reach, expected = 1.0, 0.0
        limit = min(cap, self.MAX_DEPTH)
        for d in range(1, limit + 1):
            reach *= self._position_p(d - 1, tail_p)
            expected += reach
            rate = (1.0 + expected) / self.cost(d)
            # 2% hysteresis toward narrower rounds: at equal throughput the
            # narrow round wastes less on a reject.
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
        # wide round's samples count individually, not as one observation.
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


def block_policy(max_depth: int, costs=None):
    """Fresh per-turn WidthPolicy seeded with the measured round-cost ladder."""
    table = list(costs or BLOCK_ROUND_COSTS)
    table += [table[-1]] * (WidthPolicy.MAX_DEPTH + 1 - len(table))
    return WidthPolicy(max_depth, table)



# -- target tap ---------------------------------------------------------------

def install_tap(model: Any, layer_ids) -> bool:
    """Make the target's tapped DecoderLayers publish their output hidden into
    TAP whenever it is armed. Per-instance class swap to a subclass whose
    ``__call__`` defers to whatever ``DecoderLayer.__call__`` currently is
    (stock or the fastpath patch — resolved at call time), so install order
    against mlx_fastpath does not matter and untapped layers are untouched."""
    try:
        from mlx_lm.models import qwen3_5 as q35
    except ImportError:
        return False
    layers = model.language_model.model.layers
    if max(layer_ids) >= len(layers):
        return False

    class _TappedDecoderLayer(q35.DecoderLayer):
        def __call__(self, x, mask=None, cache=None):
            out = super().__call__(x, mask=mask, cache=cache)
            tap = TAP
            if tap is not None:
                tap[self._dflash_tap] = out
            return out

    for i in layer_ids:
        layer = layers[i]
        if not isinstance(layer, q35.DecoderLayer):
            return False
        layer._dflash_tap = int(i)
        layer.__class__ = _TappedDecoderLayer
    return True


# -- loading ------------------------------------------------------------------

def _target_key(model: Any):
    args = getattr(getattr(model, "language_model", None), "args", None)
    if args is None:
        return None
    return (int(args.hidden_size), int(args.num_hidden_layers), int(args.vocab_size))


def _built_dir(src: str, bits: int, gs: int) -> str:
    """Where a CHAD_DFLASH_PATH checkpoint's locally-built sidecar lands."""
    return os.path.join(
        os.path.expanduser("~/.cache/chad/dflash"),
        f"{os.path.basename(src.rstrip('/'))}-q{bits}g{gs}")


def _quantize(drafter, bits: int, gs: int) -> None:
    import mlx.nn as nn
    # Linears at the draft precision; codebooks (embeddings, rank-256 rows
    # gathered per candidate) at 8-bit — tiny, and the bilinear score is
    # sensitive to them. Norms, rope, base kernels stay full precision.
    nn.quantize(drafter, group_size=gs, bits=bits,
                class_predicate=lambda p, m: isinstance(m, nn.Linear))
    nn.quantize(drafter, group_size=gs, bits=8,
                class_predicate=lambda p, m: isinstance(m, nn.Embedding))


def _remap(weights: dict) -> dict:
    """HF checkpoint names -> module parameter names (codebooks are modules here)."""
    out = {}
    for k, v in weights.items():
        if k.endswith("_codebook"):
            k = k + ".weight"
        out[k] = v
    return out


def _load_sidecar(sdir: str, bits: int, gs: int):
    import mlx.core as mx
    with open(os.path.join(sdir, "config.json")) as f:
        cfg = DFlashConfig.from_dict(json.load(f))
    drafter = build(cfg)
    _quantize(drafter, bits, gs)
    drafter.load_weights(list(mx.load(os.path.join(sdir, "model.safetensors")).items()))
    drafter.eval()
    mx.eval(drafter.parameters())
    return drafter


def build_sidecar(src_dir: str, out_dir: str, bits: int = 4, gs: int = 64) -> str:
    """Quantize an HF DFlash checkpoint dir into a chad sidecar dir. Peak memory
    stays near one layer's bf16 weights: the bf16 file is memory-mapped and each
    layer's quantization is evaluated before the next is touched."""
    import glob

    import mlx.core as mx
    from mlx.utils import tree_flatten

    with open(os.path.join(src_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = DFlashConfig.from_dict(raw)
    drafter = build(cfg)
    weights: dict = {}
    for st in sorted(glob.glob(os.path.join(src_dir, "*.safetensors"))):
        weights.update(mx.load(st))
    weights = _remap(weights)
    drafter.load_weights(list(weights.items()))
    del weights
    _quantize(drafter, bits, gs)
    for layer in drafter.layers:
        mx.eval(layer.parameters())
        mx.clear_cache()
    mx.eval(drafter.parameters())
    os.makedirs(out_dir, exist_ok=True)
    mx.save_safetensors(
        os.path.join(out_dir, "model.safetensors"),
        dict(tree_flatten(drafter.parameters())),
        metadata={"bits": str(bits), "group_size": str(gs), "source": src_dir})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(raw, f)
    return out_dir


def _is_sidecar(d: str) -> bool:
    try:
        import struct
        with open(os.path.join(d, "model.safetensors"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return "bits" in (hdr.get("__metadata__") or {})
    except Exception:  # noqa: BLE001
        return False


def bundle_dir(model_dir: str) -> Optional[str]:
    """The drafter sidecar bundled with the weights at `model_dir`, or None.
    CHAD_DFLASH_PATH wins when set (an explicit dir is an order)."""
    explicit = os.environ.get("CHAD_DFLASH_PATH")
    if explicit:
        return os.path.expanduser(explicit)
    d = os.path.join(model_dir, _BUNDLE)
    return d if os.path.isfile(os.path.join(d, "config.json")) else None


def _complete(d: str) -> bool:
    return (os.path.isfile(os.path.join(d, "config.json"))
            and os.path.isfile(os.path.join(d, "model.safetensors")))


def ensure_bundle(model_dir: str, repo_id: Optional[str]) -> None:
    """Fetch the bundle's weights when the weights dir has its config but not its
    `model.safetensors`. Best-effort and quiet on failure — a missing drafter costs
    speed, never correctness.

    This half-state is reachable and was measured, so it is worth healing rather than
    asserting against: mlx-lm downloads a repo with `allow_patterns=[... ,
    "model*.safetensors", ...]`, and that pattern is anchored at the START of the
    relative path. `dflash/config.json` matches `*.json` (the leading star absorbs the
    directory) but `dflash/model.safetensors` matches nothing — so any path that reaches
    the weights through mlx-lm rather than through `cli._ensure_model`'s pattern-free
    snapshot lands a config-only bundle. `cli._cached_weights_complete` then reports the
    cache complete, so the full download never runs and the drafter stays dark forever.
    The same glob is why the file is named `model.safetensors` under a subdirectory in
    the first place: mlx-lm's LOADER globs `model*.safetensors` at the repo root, so the
    bundle is invisible to it. Downloader and loader read the same pattern differently.
    """
    if not repo_id or os.path.isdir(repo_id):
        return
    d = os.path.join(model_dir, _BUNDLE)
    if _complete(d) or not os.path.isfile(os.path.join(d, "config.json")):
        return
    try:
        from huggingface_hub import hf_hub_download
        log.info("DFlash drafter: fetching the bundled weights from %s "
                 "(the base download's file filter skips them)", repo_id)
        hf_hub_download(repo_id, f"{_BUNDLE}/model.safetensors")
    except Exception as e:  # noqa: BLE001 — offline/gated: decode without the drafter
        log.warning("DFlash drafter: could not fetch %s/%s (%s); decoding without it",
                    repo_id, _BUNDLE, e)


def load_drafter(model: Any, model_dir: str, repo_id: Optional[str] = None,
                 bits: int = 4, gs: int = 64) -> Optional[Any]:
    """Load the DFlash drafter bundled with the target's weights (or the dir
    CHAD_DFLASH_PATH names), bound to the target's embedding/lm_head, with the
    target tap installed. None when no drafter ships for this model or on any
    failure — DFlash is a pure speed feature, never load-bearing.

    `repo_id` is the model's HF repo when it came from the hub, so a bundle whose
    weights the base download filtered out can be completed (see ensure_bundle)."""
    try:
        key = _target_key(model)
        ensure_bundle(model_dir, repo_id)
        sdir = bundle_dir(model_dir)
        if key is None or sdir is None:
            return None
        if not _is_sidecar(sdir):
            # An unquantized HF checkpoint: quantize it once into the cache.
            sdir = build_sidecar(sdir, _built_dir(sdir, bits, gs), bits, gs)
        drafter = _load_sidecar(sdir, bits, gs)
        cfg = drafter.config
        if cfg.hidden_size != key[0] or cfg.vocab_size != key[2] \
                or max(cfg.target_layer_ids) >= key[1]:
            log.warning("DFlash drafter %s does not match the target shape %s; "
                        "decoding without it", sdir, key)
            return None
        lm = model.language_model
        embed = lm.model.embed_tokens
        if getattr(lm.args, "tie_word_embeddings", False):
            drafter.bind(embed, embed.as_linear)
        else:
            drafter.bind(embed, lm.lm_head)
        if not install_tap(model, cfg.target_layer_ids):
            log.warning("DFlash drafter: target tap not installable; decoding "
                        "without it")
            return None
        log.info("DFlash drafter loaded (%s, %d-bit g%d, block %d, selector top-%d, "
                 "taps %s)", sdir, bits, gs, cfg.block_size, cfg.selector_top_k,
                 list(cfg.target_layer_ids))
        return drafter
    except Exception as e:  # noqa: BLE001 — never break model load over DFlash
        log.warning("DFlash drafter load failed (%s); decoding without it", e)
        return None


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse
    ap = argparse.ArgumentParser(
        description="Build a quantized DFlash sidecar from an HF checkpoint dir. "
                    "Write it to <model_dir>/dflash/ to bundle it with the weights.")
    ap.add_argument("src", help="HF checkpoint dir (config.json + *.safetensors)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args(argv)
    out = a.out or _built_dir(a.src, a.bits, a.group_size)
    print(build_sidecar(a.src, out, a.bits, a.group_size))


if __name__ == "__main__":  # pragma: no cover
    main()
