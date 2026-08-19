"""Decode fast-path for the dense qwen3_5 hybrid (Qwen3.8-27B class).

Profiling on a 24 GB M4 Pro showed this family's decode step is dispatch-bound,
not bandwidth-bound: hundreds of small Metal kernels per token, each costing
~9 µs of serialized launch/gap latency on top of the actual weight traffic. This
module removes a large fraction of those kernels with bit-exact-per-row weight
transforms and per-layer `mx.compile` fast paths:

1. **MLP gate|up concat** — each layer's `gate_proj` and `up_proj` (identical
   shape/bits) are concatenated along the output axis so the two matmuls become
   one `quantized_matmul` + split. Row-wise math is unchanged.
2. **GDN in_proj concat** — each GatedDeltaNet's four same-input projections
   (`in_proj_qkv|z|b|a`) become one `quantized_matmul` + split.
3. **Compiled S==1 layer step** — the whole MLP block and the whole GDN forward
   (conv, delta kernel, gated norm) each become ONE compiled call per layer with
   the layernorms and residuals folded in. Cache state is threaded explicitly
   (pure state-in/state-out), so the compiled path composes with the engine's
   snapshot/rewind machinery, which copies cache entries by reference.

Prefill (S>1) keeps the STOCK op graph (only the weight concats apply): the
compiled kernels fuse elementwise chains and change bf16 rounding, and on a
recurrent hybrid a prefill-side rounding change compounds across the whole
transcript. Decode-side, greedy token choices were verified identical to stock.

A sibling MoE path lived here through 1.x, for the Ornith 35B: expert gate|up
concats plus a compiled sparse-MoE block body, and a set of fused MoE Metal
kernels in `mlx_moe_fused.py` worth another 5-7% decode on that geometry. 2.0.0
removed all of it with the model — the shipped checkpoint is dense, so none of it
could ever install.

Scope: applies when the loaded model looks like the qwen3_5 DENSE hybrid (GDN +
attention + quantized swiglu MLP). Anything unexpected → install() is a silent
no-op (stock behavior). Opt out with CHAD_NO_FASTPATH=1.
"""

from typing import Any

from . import config
from .diag import log


def install(model: Any, model_path: Any = None) -> bool:
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
        if _looks_like_hybrid_dense(model):
            _concat_dense_gate_up(model)
            _concat_gdn_in_projs(model)
            _install_layer_fastpath(model)
            log.info("FASTPATH installed (dense hybrid): fused MLP + fused GDN "
                     "projections + S=1 layer step")
            return True
        return False
    except Exception as e:  # noqa: BLE001 — perf path must never break loading
        log.warning("mlx fastpath install failed (%s); running stock", e)
        return False


def _looks_like_hybrid_dense(model) -> bool:
    """True only for the qwen3_5 DENSE hybrid (Qwen3.8-27B class): per-layer
    [GDN|attention] + a plain quantized swiglu MLP, no experts anywhere."""
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import MLP as DenseMLP
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    layers = getattr(getattr(getattr(model, "language_model", None), "model", None),
                     "layers", None)
    if not layers:
        return False
    saw_gdn = saw_attn = False
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, DenseMLP):
            return False
        for name in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(mlp, name, None)
            if not isinstance(m, nn.QuantizedLinear) or hasattr(m, "bias"):
                return False
        if layer.is_linear:
            if not isinstance(layer.linear_attn, GatedDeltaNet):
                return False
            saw_gdn = True
        else:
            saw_attn = True
    # gate|up must agree on quant params to concat
    m0 = layers[0].mlp
    if (m0.gate_proj.bits != m0.up_proj.bits
            or m0.gate_proj.group_size != m0.up_proj.group_size):
        return False
    return saw_gdn and saw_attn


def _concat_dense_gate_up(model) -> None:
    """One quantized_matmul instead of two: per layer, stack the dense MLP's
    gate|up weights along the output axis. Bit-exact per row, same argument as
    the expert concat; originals keep placeholder arrays."""
    import mlx.core as mx

    z = mx.zeros((8,), dtype=mx.uint32)
    for layer in model.language_model.model.layers:
        mlp = layer.mlp
        g, u = mlp.gate_proj, mlp.up_proj
        w = mx.contiguous(mx.concatenate([g.weight, u.weight], axis=0))
        s = mx.contiguous(mx.concatenate([g.scales, u.scales], axis=0))
        b = mx.contiguous(mx.concatenate([g.biases, u.biases], axis=0))
        mx.eval(w, s, b)
        mlp._fused_w, mlp._fused_s, mlp._fused_b = w, s, b
        mlp._fused_gs, mlp._fused_bits = g.group_size, g.bits
        for m in (g, u):
            m.weight = z
            m.scales = z
            m.biases = z
        mx.clear_cache()
    _patch_dense_mlp_call()


def _patch_dense_mlp_call() -> None:
    """Replace the dense MLP __call__ with the fused-concat version (all S).
    The class is shared with qwen3_next models; foreign instances (no _fused_w)
    take the stock path."""
    import mlx.core as mx
    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.models.qwen3_next import swiglu

    if getattr(q35.MLP.__call__, "_chad_fastpath", False):
        return
    stock_call = q35.MLP.__call__

    def fused_call(self, x):
        if not hasattr(self, "_fused_w"):
            return stock_call(self, x)
        gu = _qmm(
            x, self._fused_w, scales=self._fused_s, biases=self._fused_b,
            transpose=True, group_size=self._fused_gs, bits=self._fused_bits)
        g, u = mx.split(gu, 2, axis=-1)
        return self.down_proj(swiglu(g, u))

    fused_call._chad_fastpath = True  # type: ignore[attr-defined]
    q35.MLP.__call__ = fused_call  # type: ignore[method-assign]


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


# Verify-forward input capture (MTP speculative decoding). When an engine
# arms a collector dict here, the NEXT S>1 GDN forwards record, per GDN
# layer, the conv_input activation and the recurrence's inputs + pre-round
# state — all by reference (they are intermediates of the round's graph, held
# one round). A partial draft rejection then rebuilds the hybrid cache by
# re-running the tiny recurrence over the accepted prefix instead of paying
# either the re-feed forward that made speculative decoding a wash on
# recurrent models (the PLD lesson) or the stepped per-position state
# checkpoints that made verify forwards ~3x a plain step. The batched matmuls
# are untouched, so weights are still read once per verify. Armed per-forward
# by engine._generate_mtp; always None during prefill and normal decode.
GDN_COLLECTOR = None

# Quantized-matmul indirection for the fused-concat call sites below. Stock
# mx.quantized_matmul until mlx_qmm_s.install() swaps in its small-batch
# dispatcher (same signature); S=1 decode and prefill route back to the stock
# op inside the dispatcher, so this hook only changes verify-shaped calls.
QMM = None


def _qmm(x, w, **kw):
    import mlx.core as mx
    return (QMM or mx.quantized_matmul)(x, w, **kw)


def _patch_gdn_call() -> None:
    """Stock-graph GDN forward using the fused in_proj (used for S>1; the S==1
    decode path is replaced again by the compiled step in _install_layer_fastpath)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q35

    if getattr(q35.GatedDeltaNet.__call__, "_chad_fastpath", False):
        return
    stock_call = q35.GatedDeltaNet.__call__

    def call(self, inputs, mask=None, cache=None):
        if not hasattr(self, "_fused_w"):
            return stock_call(self, inputs, mask=mask, cache=cache)
        B, S, _ = inputs.shape
        big = _qmm(
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
        coll = GDN_COLLECTOR
        if (coll is not None and cache is not None and S > 1
                and mask is None and cache.lengths is None):
            # Capture-and-replay: run the SAME batched recurrence a plain
            # S>1 forward uses and record, by reference, this round's inputs
            # plus the pre-round state — free, they are intermediates of the
            # graph being built. A partial rejection re-runs the recurrence
            # over the accepted prefix only (engine rollback: ~one tiny lazy
            # kernel per GDN layer); full acceptance costs nothing. The
            # stepped-checkpoint design this replaces serialized S kernel
            # launches and materialized S recurrent states per layer per
            # verify (48 layers x S x ~12.6 MB on the 27B), which ate the
            # entire speculative win.
            coll["conv"].append(conv_input)
            coll["args"].append(
                (q, k, v, a, b, self.A_log, self.dt_bias, state,
                 not self.training))
        out, state = q35.gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    call._chad_fastpath = True  # type: ignore[attr-defined]
    q35.GatedDeltaNet.__call__ = call  # type: ignore[method-assign]


def _install_layer_fastpath(model) -> None:
    """Per-layer compiled decode step (S==1 only): norms+residuals+block bodies
    fold into one compiled call for the MLP and one for the GDN. Prefill and
    any unexpected cache state fall back to the stock DecoderLayer body."""
    from mlx_lm.models import qwen3_5 as q35

    for layer in model.language_model.model.layers:
        if hasattr(layer.mlp, "_fused_w"):
            layer._mlp_fast = _compile_dense_step(layer)
        else:
            continue  # unknown mlp shape: layer stays stock
        if layer.is_linear and hasattr(layer.linear_attn, "_fused_w"):
            layer._gdn_fast = _compile_gdn_step(layer)

    if getattr(q35.DecoderLayer.__call__, "_chad_fastpath", False):
        return
    stock_layer_call = q35.DecoderLayer.__call__

    def layer_call(self, x, mask=None, cache=None):
        # S==1 cannot distinguish decode from a stray 1-token prefill chunk
        # (mask is None for both); such a chunk takes this path and pays one
        # token's worth of the same rounding class decode itself injects into
        # the cache every step — bounded, unlike S>1 compile-fusion drift,
        # which is why prefill proper stays on the stock graph.
        if mask is None and x.shape[1] == 1 and cache is not None \
                and getattr(self, "_mlp_fast", None) is not None:
            if self.is_linear:
                if (getattr(self, "_gdn_fast", None) is not None
                        and cache[0] is not None and cache[1] is not None
                        and cache.lengths is None):
                    h, new_conv, new_rec = self._gdn_fast(x, cache[0], cache[1])
                    out = self._mlp_fast(h)
                    cache[0] = new_conv
                    cache[1] = new_rec
                    cache.advance(1)
                    return out
            else:
                r = self.self_attn(self.input_layernorm(x), mask, cache)
                return self._mlp_fast(x + r)
        # NOTE (plan 138): a compiled S=2..4 verify step was built and
        # measured HERE — round time did not move (dispatch was not the
        # verify cost), and the compiled bodies bypass the QMM hook that
        # mlx_qmm_s routes small-S weight matmuls through, which IS the
        # verify cost under production memory residency. So S>1 stays on the
        # stock graph, where every projection reaches the hook.
        return stock_layer_call(self, x, mask=mask, cache=cache)

    layer_call._chad_fastpath = True  # type: ignore[attr-defined]
    q35.DecoderLayer.__call__ = layer_call  # type: ignore[method-assign]


def _compile_gdn_step(layer):
    import mlx.core as mx
    return mx.compile(_gdn_body(layer))


def _compile_dense_step(layer):
    import mlx.core as mx
    return mx.compile(_dense_mlp_body(layer))


def _dense_mlp_body(layer):
    """post_attention_layernorm + fused swiglu MLP + residual, UNCOMPILED so the
    caller decides the compile region)."""
    import mlx.core as mx
    import mlx.nn as nn

    mlp = layer.mlp
    ln_w = layer.post_attention_layernorm.weight
    ln_eps = layer.post_attention_layernorm.eps
    fw, fs, fb = mlp._fused_w, mlp._fused_s, mlp._fused_b
    gs_, bits = mlp._fused_gs, mlp._fused_bits
    dp = mlp.down_proj

    def fwd(h):
        x = mx.fast.rms_norm(h, ln_w, ln_eps)
        gu = mx.quantized_matmul(x, fw, scales=fs, biases=fb, transpose=True,
                                 group_size=gs_, bits=bits)
        g, u = mx.split(gu, 2, axis=-1)
        y = mx.quantized_matmul(nn.silu(g) * u, dp.weight, scales=dp.scales,
                                biases=dp.biases, transpose=True,
                                group_size=dp.group_size, bits=dp.bits)
        return h + y

    return fwd


def _gdn_body(layer):
    """input_layernorm + full GDN forward + residual, UNCOMPILED, with explicit
    (conv_state, recurrent_state) threading."""
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

    return fwd
