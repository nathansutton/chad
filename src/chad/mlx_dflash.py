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

Versus the checkpoint's own MTP head (mlx_mtp.py): the block comes out of one
drafter forward instead of a k-step chain, and acceptance holds to depth 7 where the
1-layer MTP chain decays past 3. The verify side is identical — one batched target
forward, exact rejection sampling, the same rollback primitives.

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

Weights resolve in order: $CHAD_DFLASH_PATH (a local dir holding the HF checkpoint
or a built sidecar), then ~/.cache/chad/dflash/<repo>-q4g64/ (built on first use),
else the registered HF checkpoint for the loaded target is downloaded (bf16,
~3.8 GB once) and quantized to a 4-bit sidecar (~1.1 GB resident).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .diag import log

_SIDECAR_DIR = os.path.expanduser("~/.cache/chad/dflash")

# (target hidden_size, target num_hidden_layers, vocab_size) -> published drafter.
# `incoai/Qwen3.8-27B-DFlash2` and `z-lab/Qwen3.8-27B-DFlash2` are byte-identical.
_REGISTRY = {
    (5120, 64, 248320): "incoai/Qwen3.8-27B-DFlash2",
}

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

def block_policy(max_depth: int, per_row: float = 0.5, h_block: float = 0.2):
    """Per-round verified-width schedule for the block drafter.

    Same expected-tokens-per-time argmax as mlx_mtp.DepthPolicy (per-position
    acceptance EMAs, pooled tail, full-accept streaks, the near-tie margin
    clamp, online cost observation), with the block drafter's cost shape: the
    block is always drafted WHOLE in one forward, so drafting costs the same
    `h_block` whatever width is verified, and the round cost is that plus the
    verify ladder — seeded at `per_row` serial-steps per extra verify row (the
    M4 Pro / 3-bit measurement: ~33 ms per row on a 56 ms step), corrected by
    observe_cost() from real rounds. Candidates are every width up to the
    block cap (7): there is no plateau past it to jump to, so nothing is gated
    behind a streak. Hot content runs the full block; where acceptance drops
    (sampled decoding, cold prompts) it collapses to 1-2 or a free skip, so a
    bad regime degrades to the serial step rather than below it."""
    from .mlx_mtp import DepthPolicy

    class BlockPolicy(DepthPolicy):
        MAX_DEPTH = 7
        CANDIDATES = tuple(range(8))
        SHALLOW_MAX = 7

    costs = [1.0 + per_row * d + (h_block if d else 0.0)
             for d in range(DepthPolicy.MAX_DEPTH + 1)]
    return BlockPolicy(max_depth, h_block, costs)


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


def _sidecar_dir(repo: str, bits: int, gs: int) -> str:
    return os.path.join(_SIDECAR_DIR,
                        f"{os.path.basename(repo.rstrip('/'))}-q{bits}g{gs}")


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


def repo_for_model_dir(model_dir: str) -> Optional[str]:
    """The registered drafter repo for the model at `model_dir` (config.json
    shape lookup, no weights touched), honoring CHAD_DFLASH_REPO; None when
    no drafter is published for that shape."""
    env = os.environ.get("CHAD_DFLASH_REPO")
    if env:
        return env
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        key = (int(tc["hidden_size"]), int(tc["num_hidden_layers"]),
               int(tc["vocab_size"]))
    except Exception:  # noqa: BLE001
        return None
    return _REGISTRY.get(key)


def sidecar_ready(repo: str, bits: int = 4, gs: int = 64) -> bool:
    """True when the quantized sidecar for `repo` is already built (or an
    explicit CHAD_DFLASH_PATH is set) — i.e. no download is needed."""
    if os.environ.get("CHAD_DFLASH_PATH"):
        return True
    d = _sidecar_dir(repo, bits, gs)
    return (os.path.isfile(os.path.join(d, "model.safetensors"))
            and os.path.isfile(os.path.join(d, "config.json")))


def load_drafter(model: Any, model_dir: str, bits: int = 4, gs: int = 64) -> Optional[Any]:
    """Load (building/downloading on first use) the DFlash drafter matching a
    loaded target, bound to its embedding/lm_head, with the target tap
    installed. None when no drafter is registered for this target or on any
    failure — DFlash is a pure speed feature, never load-bearing."""
    try:
        key = _target_key(model)
        if key is None:
            return None
        repo = os.environ.get("CHAD_DFLASH_REPO") or _REGISTRY.get(key)
        explicit = os.environ.get("CHAD_DFLASH_PATH")
        if not repo and not explicit:
            return None
        if explicit:
            explicit = os.path.expanduser(explicit)
            if _is_sidecar(explicit):
                sdir = explicit
            else:
                sdir = build_sidecar(explicit, _sidecar_dir(explicit, bits, gs),
                                     bits, gs)
        else:
            sdir = _sidecar_dir(repo, bits, gs)
            if not (os.path.isfile(os.path.join(sdir, "model.safetensors"))
                    and os.path.isfile(os.path.join(sdir, "config.json"))):
                from huggingface_hub import snapshot_download
                log.info("DFlash drafter: fetching %s (one-time, bf16 -> %d-bit "
                         "sidecar at %s)", repo, bits, sdir)
                src = snapshot_download(
                    repo, allow_patterns=["config.json", "*.safetensors"])
                build_sidecar(src, sdir, bits, gs)
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
        description="Build a quantized DFlash sidecar from an HF checkpoint dir")
    ap.add_argument("src", help="HF checkpoint dir (config.json + *.safetensors)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args(argv)
    out = a.out or _sidecar_dir(a.src, a.bits, a.group_size)
    print(build_sidecar(a.src, out, a.bits, a.group_size))


if __name__ == "__main__":  # pragma: no cover
    main()
