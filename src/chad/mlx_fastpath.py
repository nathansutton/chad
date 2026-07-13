"""Decode fast-path for the hybrid MoE checkpoint (Ornith 35B class).

Profiling on the 24 GB M4 Pro (plan 075 follow-on, 2026-07-13) showed the 35B's
decode step is dispatch-bound, not bandwidth-bound: ~400 small Metal kernels per
token, each costing ~9 µs of serialized launch/gap latency on top of ~9 ms of
actual weight traffic. This module removes ~150 of those kernels per token with
two bit-exact-per-row weight transforms and per-layer `mx.compile` fast paths:

1. **Expert gate|up concat** — each MoE layer's `switch_mlp.gate_proj` and
   `up_proj` (identical shape/bits) are concatenated along the output axis so
   the two expert gathers become one `gather_qmm`. Row-wise math is unchanged.
2. **GDN in_proj concat** — each GatedDeltaNet's four same-input projections
   (`in_proj_qkv|z|b|a`, all 6-bit) become one `quantized_matmul` + split.
3. **Compiled S==1 layer step** — the whole MoE block (router, top-k, expert
   gathers, shared expert, combine) and the whole GDN forward (conv, delta
   kernel, gated norm) each become ONE compiled call per layer with the
   layernorms and residuals folded in. Cache state is threaded explicitly
   (pure state-in/state-out), so the compiled path composes with the engine's
   snapshot/rewind machinery, which copies cache entries by reference.

Prefill (S>1) keeps the STOCK op graph (only the weight concats apply): the
compiled kernels fuse elementwise chains and change bf16 rounding, and on the
recurrent hybrid a prefill-side rounding change compounds across the whole
transcript. Decode-side, greedy token choices were verified identical to stock
(the residual logit wiggle is the known path-sensitivity of the 2-bit quant;
same acceptance class as the fused-SDPA wheel patch).

Measured (35B q2_down3, M4 Pro 24 GB, stream_generate): decode 67.1→69.6 tok/s
@8k ctx, 53.2→55.6 @32k; prefill 669→723 tok/s @8k (the concats also feed the
S>1 path), flat @32k. Peak memory unchanged (±0.5 GB transient).

Scope: applies only when the loaded model looks exactly like the qwen3_5_moe
hybrid (GDN + attention + SparseMoeBlock with QuantizedSwitchLinear experts).
Anything unexpected → install() is a silent no-op (stock behavior). Opt out
with CHAD_NO_FASTPATH=1.
"""

from typing import Any

from . import config
from .diag import log


def install(model: Any) -> bool:
    """Apply the decode fast-path to a loaded mlx_lm model, in place.

    Returns True if installed, False if skipped (wrong architecture, flag off,
    or any failure — the model is left untouched on failure ONLY if the failure
    happens before the first weight mutation; transforms are ordered so each
    layer's weights are swapped atomically after its fused copies exist).
    """
    if config.flag("CHAD_NO_FASTPATH"):
        return False
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm.models import qwen3_5  # noqa: F401
    except ImportError:
        return False
    try:
        if not _looks_like_hybrid_moe(model):
            return False
        _concat_expert_gate_up(model)
        _concat_gdn_in_projs(model)
        _install_layer_fastpath(model)
        log.info("FASTPATH installed: fused expert/GDN projections + compiled "
                 "S=1 layer step")
        return True
    except Exception as e:  # noqa: BLE001 — perf path must never break loading
        log.warning("mlx fastpath install failed (%s); running stock", e)
        return False


def _looks_like_hybrid_moe(model) -> bool:
    """True only for the exact shape this module was validated on: a qwen3_5
    MoE hybrid with quantized experts and per-layer [GDN|attention] + MoE."""
    from mlx_lm.models.qwen3_5 import GatedDeltaNet
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    layers = getattr(getattr(getattr(model, "language_model", None), "model", None),
                     "layers", None)
    if not layers:
        return False
    saw_gdn = saw_moe = False
    for layer in layers:
        if not isinstance(getattr(layer, "mlp", None), Qwen3NextSparseMoeBlock):
            return False
        sw = layer.mlp.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not isinstance(getattr(sw, name, None), QuantizedSwitchLinear):
                return False
        saw_moe = True
        if layer.is_linear and isinstance(layer.linear_attn, GatedDeltaNet):
            saw_gdn = True
    # gate|up must agree on quant params to concat
    sw0 = layers[0].mlp.switch_mlp
    if (sw0.gate_proj.bits != sw0.up_proj.bits
            or sw0.gate_proj.group_size != sw0.up_proj.group_size):
        return False
    return saw_gdn and saw_moe


def _concat_expert_gate_up(model) -> None:
    """One gather_qmm instead of two: per layer, stack gate|up expert weights
    along the output axis. Per-row quantized math is unchanged (bit-exact); the
    original modules keep placeholder arrays so mlx's module tree stays valid."""
    import mlx.core as mx

    z = mx.zeros((8,), dtype=mx.uint32)
    for layer in model.language_model.model.layers:
        sw = layer.mlp.switch_mlp
        g, u = sw.gate_proj, sw.up_proj
        w = mx.contiguous(mx.concatenate([g.weight, u.weight], axis=1))
        s = mx.contiguous(mx.concatenate([g.scales, u.scales], axis=1))
        b = mx.contiguous(mx.concatenate([g.biases, u.biases], axis=1))
        mx.eval(w, s, b)
        sw._fused_w, sw._fused_s, sw._fused_b = w, s, b
        sw._fused_gs, sw._fused_bits = g.group_size, g.bits
        for m in (g, u):
            m.weight = z
            m.scales = z
            m.biases = z
        mx.clear_cache()
    _patch_switch_glu()


def _patch_switch_glu() -> None:
    """Replace SwitchGLU.__call__ with the fused-gather version (all S)."""
    import mlx.core as mx
    from mlx_lm.models import switch_layers as sl

    stock_call = sl.SwitchGLU.__call__

    def fused_call(self, x, indices):
        if not hasattr(self, "_fused_w"):  # foreign instance: stock path
            return stock_call(self, x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = sl._gather_sort(x, indices)
        gu = mx.gather_qmm(x, self._fused_w, self._fused_s, self._fused_b,
                           rhs_indices=idx, transpose=True,
                           group_size=self._fused_gs, bits=self._fused_bits,
                           sorted_indices=do_sort)
        x_gate, x_up = mx.split(gu, 2, axis=-1)
        h = self.activation(x_up, x_gate)
        y = self.down_proj(h, idx, sorted_indices=do_sort)
        if do_sort:
            y = sl._scatter_unsort(y, inv_order, indices.shape)
        return y.squeeze(-2)

    sl.SwitchGLU.__call__ = fused_call  # type: ignore[method-assign]


def _concat_gdn_in_projs(model) -> None:
    """One quantized_matmul instead of four for the GDN input projections."""
    import mlx.core as mx

    z = mx.zeros((8,), dtype=mx.uint32)
    for layer in model.language_model.model.layers:
        if not layer.is_linear:
            continue
        gd = layer.linear_attn
        mods = [gd.in_proj_qkv, gd.in_proj_z, gd.in_proj_b, gd.in_proj_a]
        if len({(m.bits, m.group_size) for m in mods}) != 1:
            continue  # mixed quant: leave this layer stock
        w = mx.contiguous(mx.concatenate([m.weight for m in mods], axis=0))
        s = mx.contiguous(mx.concatenate([m.scales for m in mods], axis=0))
        b = mx.contiguous(mx.concatenate([m.biases for m in mods], axis=0))
        mx.eval(w, s, b)
        gd._fused_w, gd._fused_s, gd._fused_b = w, s, b
        gd._fused_gs, gd._fused_bits = mods[0].group_size, mods[0].bits
        for m in mods:
            m.weight = z
            m.scales = z
            m.biases = z
        mx.clear_cache()
    _patch_gdn_call()


def _patch_gdn_call() -> None:
    """Stock-graph GDN forward using the fused in_proj (used for S>1; the S==1
    decode path is replaced again by the compiled step in _install_layer_fastpath)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q35

    stock_call = q35.GatedDeltaNet.__call__

    def call(self, inputs, mask=None, cache=None):
        if not hasattr(self, "_fused_w"):
            return stock_call(self, inputs, mask=mask, cache=cache)
        B, S, _ = inputs.shape
        big = mx.quantized_matmul(
            inputs, self._fused_w, scales=self._fused_s, biases=self._fused_b,
            transpose=True, group_size=self._fused_gs, bits=self._fused_bits)
        qkv, z, b, a = mx.split(
            big, [self.conv_dim, self.conv_dim + self.value_dim,
                  self.conv_dim + self.value_dim + self.num_v_heads], axis=-1)
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim),
                                  dtype=inputs.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, state = q35.gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    q35.GatedDeltaNet.__call__ = call  # type: ignore[method-assign]


def _install_layer_fastpath(model) -> None:
    """Per-layer compiled decode step (S==1 only): norms+residuals+block bodies
    fold into one compiled call for the MoE and one for the GDN. Prefill and
    any unexpected cache state fall back to the stock DecoderLayer body."""
    from mlx_lm.models import qwen3_5 as q35

    for layer in model.language_model.model.layers:
        layer._moe_fast = _compile_moe_step(layer)
        if layer.is_linear and hasattr(layer.linear_attn, "_fused_w"):
            layer._gdn_fast = _compile_gdn_step(layer)

    stock_layer_call = q35.DecoderLayer.__call__

    def layer_call(self, x, mask=None, cache=None):
        if mask is None and x.shape[1] == 1 and cache is not None \
                and getattr(self, "_moe_fast", None) is not None:
            if self.is_linear:
                if (getattr(self, "_gdn_fast", None) is not None
                        and cache[0] is not None and cache[1] is not None
                        and cache.lengths is None):
                    h, new_conv, new_rec = self._gdn_fast(x, cache[0], cache[1])
                    cache[0] = new_conv
                    cache[1] = new_rec
                    cache.advance(1)
                    return self._moe_fast(h)
            else:
                r = self.self_attn(self.input_layernorm(x), mask, cache)
                return self._moe_fast(x + r)
        return stock_layer_call(self, x, mask=mask, cache=cache)

    q35.DecoderLayer.__call__ = layer_call  # type: ignore[method-assign]


def _compile_moe_step(layer):
    """post_attention_layernorm + full MoE block + residual as one compiled fn."""
    import mlx.core as mx
    import mlx.nn as nn

    mlp = layer.mlp
    sw = mlp.switch_mlp
    ln_w = layer.post_attention_layernorm.weight
    ln_eps = layer.post_attention_layernorm.eps
    gate = mlp.gate
    fw, fs, fb = sw._fused_w, sw._fused_s, sw._fused_b
    sgs, sbits = sw._fused_gs, sw._fused_bits
    dp = sw.down_proj
    se = mlp.shared_expert
    seg = mlp.shared_expert_gate
    k = mlp.top_k
    norm_topk = mlp.norm_topk_prob

    def qmm(x, m):
        return mx.quantized_matmul(x, m.weight, scales=m.scales, biases=m.biases,
                                   transpose=True, group_size=m.group_size,
                                   bits=m.bits)

    def fwd(h):
        x = mx.fast.rms_norm(h, ln_w, ln_eps)
        gates = mx.softmax(qmm(x, gate), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if norm_topk:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        xe = mx.expand_dims(x, (-2, -3))
        gu = mx.gather_qmm(xe, fw, fs, fb, rhs_indices=inds, transpose=True,
                           group_size=sgs, bits=sbits)
        xg, xu = mx.split(gu, 2, axis=-1)
        hh = nn.silu(xg) * xu
        y = mx.gather_qmm(hh, dp.weight, dp.scales, dp.biases, rhs_indices=inds,
                          transpose=True, group_size=dp.group_size, bits=dp.bits)
        y = (y.squeeze(-2) * scores[..., None]).sum(axis=-2)
        sh = qmm(nn.silu(qmm(x, se.gate_proj)) * qmm(x, se.up_proj), se.down_proj)
        return h + y + mx.sigmoid(qmm(x, seg)) * sh

    return mx.compile(fwd)


def _compile_gdn_step(layer):
    """input_layernorm + full GDN forward + residual as one compiled pure fn
    with explicit (conv_state, recurrent_state) threading."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import gated_delta_update

    gd = layer.linear_attn
    ln_w = layer.input_layernorm.weight
    ln_eps = layer.input_layernorm.eps
    fw, fs, fb = gd._fused_w, gd._fused_s, gd._fused_b
    gs_, bits = gd._fused_gs, gd._fused_bits
    conv_w = gd.conv1d.weight
    n_keep = gd.conv_kernel_size - 1
    A_log, dt_bias = gd.A_log, gd.dt_bias
    norm_w = gd.norm.weight
    op = gd.out_proj
    Hk, Hv = gd.num_k_heads, gd.num_v_heads
    Dk, Dv = gd.head_k_dim, gd.head_v_dim
    key_dim, value_dim, conv_dim = gd.key_dim, gd.value_dim, gd.conv_dim
    eps = gd.layer_norm_epsilon

    def fwd(xin, conv_state, rec_state):
        inputs = mx.fast.rms_norm(xin, ln_w, ln_eps)
        B, S, _ = inputs.shape
        big = mx.quantized_matmul(inputs, fw, scales=fs, biases=fb,
                                  transpose=True, group_size=gs_, bits=bits)
        qkv, z, b, a = mx.split(
            big, [conv_dim, conv_dim + value_dim,
                  conv_dim + value_dim + Hv], axis=-1)
        z = z.reshape(B, S, Hv, Dv)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        new_conv = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(mx.conv1d(conv_input, conv_w, groups=conv_dim))
        q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
            mx.split(conv_out, [key_dim, 2 * key_dim], -1),
            [Hk, Hk, Hv], [Dk, Dk, Dv])]
        inv_scale = Dk ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, new_rec = gated_delta_update(q, k, v, a, b, A_log, dt_bias,
                                          rec_state, None, use_kernel=True)
        # RMSNormGated, matching stock's fp32 _precise_swiglu exactly
        xn = mx.fast.rms_norm(out, norm_w, eps)
        out = (nn.silu(z.astype(mx.float32)) * xn.astype(mx.float32)).astype(xin.dtype)
        out = mx.quantized_matmul(out.reshape(B, S, -1), op.weight,
                                  scales=op.scales, biases=op.biases,
                                  transpose=True, group_size=op.group_size,
                                  bits=op.bits)
        return xin + out, new_conv, new_rec

    return mx.compile(fwd)
