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

Prism's Hadamard-folded ternary packs (`prism_pack`) are the same dense hybrid with
every projection stored in a rotated basis, so each takes a blockwise Hadamard of
its INPUT first. The three transforms above port with one addition each: the
rotation happens once per fused matmul instead of once per projection (gate|up,
qkv|z and — new here, the stock model has no such concat — q|k|v all consume the
same rotated input, and the pack's sign vectors are one per input width), and the
sign vectors are folded into the weights ONCE at install (exact: they are ±1) — the
two layernorms' weights for the residual-width rotations, the rows of `up_proj` for
down_proj's — so at EVERY width a rotation is one kernel, the transform itself, in
the activation dtype. That matters most where it is least visible: a drafted round
is a width-2..8 verify forward through the uncompiled graph, and the pack's own
rotation there was four kernels (two casts, the sign multiply, the transform) on four
rotations a layer. Folding at install rather than inside the compiled bodies also
means the serial step and the verify forward rotate identically, so they write the
KV and GDN state at the same precision. out_proj and o_proj keep their sign multiply
(what feeds them is not linear in anything a sign can fold into). The GDN's two tiny
gate projections (in_proj_b/a) are unrotated bf16 linears in these packs; they fuse
into one plain matmul on the (now signed) norm output, their columns re-signed to
match, and the originals are dropped.

Scope: applies when the loaded model looks like the qwen3_5 DENSE hybrid (GDN +
attention + quantized swiglu MLP), stock-quantized or a Prism pack. Anything
unexpected → install() is a silent no-op (stock behavior). Opt out with
CHAD_NO_FASTPATH=1.
"""

from typing import TYPE_CHECKING, Callable, Optional

from . import config, gguf_pack, mlx_qmm_mma, prism_pack
from .diag import log

if TYPE_CHECKING:  # mlx is imported lazily inside the functions so the module loads on Linux
    import mlx.core as mx
    import mlx.nn as nn


def install(model: "nn.Module", model_path: Optional[str] = None) -> bool:
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
        if _gguf_decline_reason(model) is None:
            _install_gguf(model)
            log.info("FASTPATH installed (GGUF hybrid): S=1 layer step with "
                     "same-format gate|up and qkv|z fused, GDN gates fused")
            return True
        declined = _prism_decline_reason(model)
        if declined is None:
            _concat_prism_gate_up(model)
            _concat_prism_gdn_in_projs(model)
            _fuse_prism_attention(model)
            _fold_prism_signs(model)
            _install_layer_fastpath(model)
            log.info("FASTPATH installed (Prism ternary hybrid): fused MLP + fused "
                     "GDN + fused attention projections, sign vectors folded, one "
                     "one-kernel rotation each, S=1 layer step")
            return True
        if any(prism_pack.is_packed(m) for _, m in model.named_modules()):
            # A Prism pack that runs unfused pays a four-kernel rotation per
            # projection and no compiled step: correct, and far slower. Say so.
            log.warning("mlx fastpath declined this Prism pack (%s); decoding on the "
                        "unfused per-projection path, which is much slower", declined)
        return False
    except Exception as e:  # noqa: BLE001 — perf path must never break loading
        log.warning("mlx fastpath install failed (%s); running stock", e)
        return False


def _looks_like_hybrid_dense(model) -> bool:
    """True only for the qwen3_5 DENSE hybrid (Qwen3.8-27B class): per-layer
    [GDN|attention] + a plain quantized swiglu MLP, no experts anywhere."""
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q35

    if not isinstance(model, q35.Model):
        return False
    layers = model.language_model.model.layers
    if not layers:
        return False
    saw_gdn = saw_attn = False
    first: Optional[list[nn.QuantizedLinear]] = None   # layer 0's gate, up, down
    for layer in layers:
        mlp = layer.mlp
        if not isinstance(mlp, q35.MLP):
            return False
        projs = [m for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
                 if isinstance(m, nn.QuantizedLinear) and not hasattr(m, "bias")]
        if len(projs) != 3:
            return False
        if first is None:
            first = projs
        if layer.is_linear:
            if not isinstance(layer.linear_attn, q35.GatedDeltaNet):
                return False
            saw_gdn = True
        else:
            saw_attn = True
    # gate|up must agree on quant params to concat
    if first is None:
        return False
    gate0, up0 = first[0], first[1]
    if gate0.bits != up0.bits or gate0.group_size != up0.group_size:
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


# The methods this module installed over mlx_lm's classes, so a repeat install
# recognizes its own patch instead of wrapping it a second time.
_mlp_call: Optional[Callable[..., "mx.array"]] = None
_gdn_call: Optional[Callable[..., "mx.array"]] = None
_layer_call: Optional[Callable[..., "mx.array"]] = None


def _patch_dense_mlp_call() -> None:
    """Replace the dense MLP __call__ with the fused-concat version (all S).
    The class is shared with qwen3_next models; foreign instances (no _fused_w)
    take the stock path."""
    global _mlp_call
    import mlx.core as mx
    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.models.qwen3_next import swiglu

    if q35.MLP.__call__ is _mlp_call:
        return
    stock_call = q35.MLP.__call__

    def fused_call(self, x):
        if not hasattr(self, "_fused_w"):
            return stock_call(self, x)
        if hasattr(self, "_fused_signs"):     # Prism: one rotation for gate|up,
            x = prism_pack.rotate(x, self._fused_block)   # signs already in the norm
        # mlx_qmm_mma.qmm: the small-M MMA kernel on verified (shape, width)
        # pairs — speculative verify widths — stock quantized_matmul otherwise.
        gu = mlx_qmm_mma.qmm(x, self._fused_w, self._fused_s, self._fused_b,
                             self._fused_gs, self._fused_bits)
        g, u = mx.split(gu, 2, axis=-1)
        return self.down_proj(swiglu(g, u))

    # SAFETY: mlx_lm's classes are plain Python, so the method is reassignable in
    # place; only the stubs say otherwise.
    q35.MLP.__call__ = fused_call  # type: ignore[method-assign]  # SAFETY: plain class
    _mlp_call = fused_call


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


# Verify-forward input capture (speculative decoding). When an engine
# arms a collector dict here, the NEXT S>1 GDN forwards record, per GDN
# layer, the conv_input activation and the recurrence's inputs + pre-round
# state — all by reference (they are intermediates of the round's graph, held
# one round). A partial draft rejection then rebuilds the hybrid cache by
# re-running the tiny recurrence over the accepted prefix instead of paying
# either the re-feed forward that made speculative decoding a wash on
# recurrent models (the PLD lesson) or the stepped per-position state
# checkpoints that made verify forwards ~3x a plain step. The batched matmuls
# are untouched, so weights are still read once per verify. Armed per-forward
# by engine._generate_spec; always None during prefill and normal decode.
GDN_COLLECTOR = None


def _patch_gdn_call() -> None:
    """Stock-graph GDN forward using the fused in_proj (used for S>1; the S==1
    decode path is replaced again by the compiled step in _install_layer_fastpath)."""
    global _gdn_call
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q35

    if q35.GatedDeltaNet.__call__ is _gdn_call:
        return
    stock_call = q35.GatedDeltaNet.__call__

    def call(self, inputs, mask=None, cache=None):
        gguf = hasattr(self, "_gguf_ba")
        if not hasattr(self, "_fused_w") and not gguf:
            return stock_call(self, inputs, mask=mask, cache=cache)
        B, S, _ = inputs.shape
        if gguf:
            qkv, z = _gguf_qkv_z(self, inputs)
            b, a = mx.split(inputs @ self._gguf_ba.T, 2, axis=-1)
        elif hasattr(self, "_fused_signs"):
            # Prism: qkv|z share one rotation; b|a read the unrotated input. The
            # input arrives signed (folded into the layernorm), which is the
            # rotation's sign multiply already done and is why b|a are re-signed.
            xr = prism_pack.rotate(inputs, self._fused_block)
            big = mlx_qmm_mma.qmm(xr, self._fused_w, self._fused_s, self._fused_b,
                                  self._fused_gs, self._fused_bits)
            qkv, z = mx.split(big, [self.conv_dim], axis=-1)
            b, a = mx.split(inputs @ self._fused_ba.T, 2, axis=-1)
        else:
            big = mlx_qmm_mma.qmm(inputs, self._fused_w, self._fused_s, self._fused_b,
                                  self._fused_gs, self._fused_bits)
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

    # SAFETY: mlx_lm's classes are plain Python, so the method is reassignable in
    # place; only the stubs say otherwise.
    q35.GatedDeltaNet.__call__ = call  # type: ignore[method-assign]  # SAFETY: plain class
    _gdn_call = call


def _install_layer_fastpath(model) -> None:
    """Per-layer compiled decode step (S==1 only): norms+residuals+block bodies
    fold into one compiled call for the MLP and one for the GDN. Prefill and
    any unexpected cache state fall back to the stock DecoderLayer body."""
    for layer in model.language_model.model.layers:
        if hasattr(layer.mlp, "_fused_w"):
            layer._mlp_fast = _compile_dense_step(layer)
        else:
            continue  # unknown mlp shape: layer stays stock
        if layer.is_linear and hasattr(layer.linear_attn, "_fused_w"):
            layer._gdn_fast = _compile_gdn_step(layer)
    _patch_layer_call()


def _patch_layer_call() -> None:
    """Route S==1 decode through each layer's compiled `_mlp_fast`/`_gdn_fast`."""
    global _layer_call
    from mlx_lm.models import qwen3_5 as q35

    if q35.DecoderLayer.__call__ is _layer_call:
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
        # A compiled S=2..4 verify step was built and measured HERE, and
        # round time did not move: dispatch is not what a verify forward
        # costs. S>1 therefore stays on the stock graph — the simpler of two
        # equal-speed options.
        return stock_layer_call(self, x, mask=mask, cache=cache)

    # SAFETY: mlx_lm's classes are plain Python, so the method is reassignable in
    # place; only the stubs say otherwise.
    q35.DecoderLayer.__call__ = layer_call  # type: ignore[method-assign]  # SAFETY: plain class
    _layer_call = layer_call


def _compile_gdn_step(layer):
    import mlx.core as mx
    body = (_prism_gdn_body if hasattr(layer.linear_attn, "_fused_signs")
            else _gdn_body)
    return mx.compile(body(layer))


def _compile_dense_step(layer):
    import mlx.core as mx
    body = _prism_mlp_body if hasattr(layer.mlp, "_fused_signs") else _dense_mlp_body
    return mx.compile(body(layer))


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


# ---------------------------------------------------------------- GGUF checkpoints
#
# A GGUF (gguf_pack) is the same dense hybrid with every projection in llama.cpp
# blocks, and its formats vary per role and per layer: layer 0's gate|up may be IQ1_S
# while its down is IQ2_XS, and qkv and z often differ. Fusion therefore happens only
# between same-format partners, per layer, and a layer whose partners differ runs them
# as two matmuls inside the same compiled step. Stacking block rows is exact, and the
# originals become row views of the fused weight, so fusing costs no memory. The GDN's
# two per-head gate projections are small dense bf16 linears (gguf_pack) and always fuse.


def _gguf_decline_reason(model) -> Optional[str]:
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q35

    if not isinstance(model, q35.Model):
        return "not a qwen3_5 model"
    layers = model.language_model.model.layers
    if not layers:
        return "no layers"
    for layer in layers:
        mlp = layer.mlp
        if not isinstance(mlp, q35.MLP):
            return "a non-dense MLP"
        if not all(gguf_pack.is_gguf_linear(m)
                   for m in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)):
            return "MLP projections not in GGUF blocks"
        if layer.is_linear:
            gd = layer.linear_attn
            if not (gguf_pack.is_gguf_linear(gd.in_proj_qkv)
                    and gguf_pack.is_gguf_linear(gd.in_proj_z)):
                return "GDN projections not in GGUF blocks"
            if not (isinstance(gd.in_proj_b, nn.Linear) and isinstance(gd.in_proj_a, nn.Linear)):
                return "GDN gate projections not dense"
    return None


def _fuse_gguf_pair(owner, first: str, second: str):
    """Fuse owner.<first>|owner.<second> when they share a format; the originals are
    rebound to row views of the fused weight. None when they differ."""
    a, b = owner[first], owner[second]
    if a.qtype != b.qtype or a.in_dims != b.in_dims or "gather" in a or "gather" in b:
        return None
    fused = gguf_pack.fuse_rows([a, b])
    n = a.out_dims
    owner[first] = gguf_pack.row_slice(fused, 0, n)
    owner[second] = gguf_pack.row_slice(fused, n, fused.out_dims)
    return fused


def _gguf_qkv_z(gd, x):
    """The GDN's qkv and z projections: one fused matmul when they share a format."""
    import mlx.core as mx
    if "_gguf_qkvz" in gd:
        qkv, z = mx.split(gd._gguf_qkvz(x), [gd.conv_dim], axis=-1)
        return qkv, z
    return gd.in_proj_qkv(x), gd.in_proj_z(x)


def _install_gguf(model) -> None:
    import mlx.core as mx

    for layer in model.language_model.model.layers:
        mlp = layer.mlp
        gu = _fuse_gguf_pair(mlp, "gate_proj", "up_proj")
        if gu is not None:
            mlp._gguf_gu = gu
        if layer.is_linear:
            gd = layer.linear_attn
            qz = _fuse_gguf_pair(gd, "in_proj_qkv", "in_proj_z")
            if qz is not None:
                gd._gguf_qkvz = qz
            b, a = gd.in_proj_b, gd.in_proj_a
            ba = mx.contiguous(mx.concatenate([b.weight, a.weight], axis=0))
            mx.eval(ba)
            gd._gguf_ba = ba
            layer._gdn_fast = mx.compile(_gguf_gdn_body(layer))
        layer._mlp_fast = mx.compile(_gguf_mlp_body(layer))
    _patch_gdn_call()
    _patch_layer_call()


def _gguf_mlp_body(layer):
    import mlx.core as mx
    import mlx.nn as nn

    mlp = layer.mlp
    ln_w = layer.post_attention_layernorm.weight
    ln_eps = layer.post_attention_layernorm.eps
    gu = mlp["_gguf_gu"] if "_gguf_gu" in mlp else None
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj

    def fwd(h):
        x = mx.fast.rms_norm(h, ln_w, ln_eps)
        if gu is not None:
            g, u = mx.split(gu(x), 2, axis=-1)
        else:
            g, u = gate(x), up(x)
        return h + down(nn.silu(g) * u)

    return fwd


def _gguf_gdn_body(layer):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import gated_delta_update

    gd = layer.linear_attn
    ln_w = layer.input_layernorm.weight
    ln_eps = layer.input_layernorm.eps
    ba = gd._gguf_ba
    conv_w = gd.conv1d.weight
    n_keep = gd.conv_kernel_size - 1
    A_log, dt_bias = gd.A_log, gd.dt_bias
    norm_w = gd.norm.weight
    op = gd.out_proj
    Hk, Hv = gd.num_k_heads, gd.num_v_heads
    Dk, Dv = gd.head_k_dim, gd.head_v_dim
    key_dim, conv_dim = gd.key_dim, gd.conv_dim
    eps = gd.layer_norm_epsilon

    def fwd(xin, conv_state, rec_state):
        inputs = mx.fast.rms_norm(xin, ln_w, ln_eps)
        B, S, _ = inputs.shape
        qkv, z = _gguf_qkv_z(gd, inputs)
        b, a = mx.split(inputs @ ba.T, 2, axis=-1)
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
        xn = mx.fast.rms_norm(out, norm_w, eps)
        out = (nn.silu(z.astype(mx.float32)) * xn.astype(mx.float32)).astype(xin.dtype)
        return xin + op(out.reshape(B, S, -1)), new_conv, new_rec

    return fwd


# ---------------------------------------------------------------- Prism packs


def _prism_decline_reason(model) -> Optional[str]:
    """None for a Prism-packed qwen3_5 DENSE hybrid this module can fuse: every
    projection a rotated Packed, the GDN's gate projections plain linears, and the
    same-input projections sharing one sign vector (what makes them fusable behind
    one rotation). Otherwise the first thing that did not fit, for the log."""
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q35

    if not isinstance(model, q35.Model):
        return "not a qwen3_5 model"
    layers = model.language_model.model.layers
    if not layers:
        return "no layers"

    def rotated(*mods) -> bool:
        return all(prism_pack.is_packed(m) and m.block and not m.embedding
                   and not m.signs_folded for m in mods)

    def shared(*mods) -> bool:
        return all(m.block == mods[0].block and m.signs is mods[0].signs
                   for m in mods[1:])

    saw_gdn = saw_attn = False
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        if not isinstance(mlp, q35.MLP):
            return f"layer {i}: not a dense MLP"
        if not rotated(mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            return f"layer {i}: MLP projections are not rotated packs"
        if not shared(mlp.gate_proj, mlp.up_proj):
            return f"layer {i}: gate_proj and up_proj do not share a sign vector"
        if layer.is_linear:
            gd = layer.linear_attn
            if not isinstance(gd, q35.GatedDeltaNet):
                return f"layer {i}: not a GatedDeltaNet"
            if not rotated(gd.in_proj_qkv, gd.in_proj_z, gd.out_proj):
                return f"layer {i}: GDN projections are not rotated packs"
            if not shared(gd.in_proj_qkv, gd.in_proj_z):
                return f"layer {i}: in_proj_qkv and in_proj_z do not share a sign vector"
            for lin in (gd.in_proj_b, gd.in_proj_a):
                if (not isinstance(lin, nn.Linear) or "bias" in lin
                        or lin.weight.ndim != 2):
                    return f"layer {i}: GDN gate projections are not plain linears"
            saw_gdn = True
        else:
            at = layer.self_attn
            if not rotated(at.q_proj, at.k_proj, at.v_proj, at.o_proj):
                return f"layer {i}: attention projections are not rotated packs"
            if not shared(at.q_proj, at.k_proj, at.v_proj):
                return f"layer {i}: q/k/v do not share a sign vector"
            saw_attn = True
    if not (saw_gdn and saw_attn):
        return "not a GDN + attention hybrid"
    return None


def _looks_like_prism(model) -> bool:
    return _prism_decline_reason(model) is None


def _stack(mods):
    """Concatenate Packed projections along the output axis (bit-exact per row),
    returning the fused arrays; the originals keep placeholder arrays."""
    import mlx.core as mx

    z = mx.zeros((8,), dtype=mx.uint32)
    w = mx.contiguous(mx.concatenate([m.weight for m in mods], axis=0))
    s = mx.contiguous(mx.concatenate([m.scales for m in mods], axis=0))
    b = mx.contiguous(mx.concatenate([m.biases for m in mods], axis=0))
    mx.eval(w, s, b)
    for m in mods:
        m.weight = z
        m.scales = z
        m.biases = z
    mx.clear_cache()
    return w, s, b


def _record_fuse(owner, mods) -> None:
    owner._fused_w, owner._fused_s, owner._fused_b = _stack(mods)
    owner._fused_gs, owner._fused_bits = mods[0].group_size, mods[0].bits
    owner._fused_block, owner._fused_signs = mods[0].block, mods[0].signs


def _concat_prism_gate_up(model) -> None:
    for layer in model.language_model.model.layers:
        mlp = layer.mlp
        _record_fuse(mlp, [mlp.gate_proj, mlp.up_proj])
    _patch_dense_mlp_call()


def _concat_prism_gdn_in_projs(model) -> None:
    """qkv|z behind one rotation; b|a (plain, unrotated) as one bf16 matmul. The
    b|a columns are re-signed with the input sign vector, because the input they
    read arrives signed once `_fold_prism_signs` has run ((x*s) @ (W*s).T == x @ W.T).
    One resident copy: the originals are dropped like every other fused original."""
    import mlx.core as mx

    z = mx.zeros((1,), dtype=mx.bfloat16)
    for layer in model.language_model.model.layers:
        if not layer.is_linear:
            continue
        gd = layer.linear_attn
        _record_fuse(gd, [gd.in_proj_qkv, gd.in_proj_z])
        gd._fused_ba = _signed(mx.contiguous(mx.concatenate(
            [gd.in_proj_b.weight, gd.in_proj_a.weight], axis=0)), gd._fused_signs)
        gd.in_proj_b.weight = z
        gd.in_proj_a.weight = z
    mx.clear_cache()
    _patch_gdn_call()


def _fold_prism_signs(model) -> None:
    """Fold every sign vector that CAN fold into the weights upstream of its
    rotation, once, so the rotation is the one-kernel transform on every path:

    * the residual-width vectors (gate|up, qkv|z, q|k|v) into the layernorm that
      feeds them — `rms_norm(x, w) * s == rms_norm(x, w * s)`, and nothing else reads
      either norm's output;
    * down_proj's into the ROWS of up_proj, which is a sign flip of those rows'
      affine scales and biases (-(s*q + b) == (-s)*q + (-b)), so
      `silu(g) * (u * s_mid)` costs nothing at all.

    All exact: the vectors are ±1. out_proj's and o_proj's stay with their modules
    (a gated per-head norm and an attention output gated by a sigmoid sit in front
    of them, neither linear in anything a per-channel sign could ride on), as does
    lm_head's, which the drafter calls on its own hidden states."""
    import mlx.core as mx

    for layer in model.language_model.model.layers:
        mlp = layer.mlp
        post = layer.post_attention_layernorm
        post.weight = _signed(post.weight, mlp._fused_signs)
        dp = mlp.down_proj
        n = int(mlp._fused_s.shape[0]) // 2
        flip = mx.concatenate([mx.ones((n,), dtype=dp.signs.dtype), dp.signs])[:, None]
        mlp._fused_s = mlp._fused_s * flip.astype(mlp._fused_s.dtype)
        mlp._fused_b = mlp._fused_b * flip.astype(mlp._fused_b.dtype)
        mx.eval(mlp._fused_s, mlp._fused_b)
        dp.signs_folded = True
        mixer = layer.linear_attn if layer.is_linear else layer.self_attn
        pre = layer.input_layernorm
        pre.weight = _signed(pre.weight, mixer._fused_signs)
    mx.clear_cache()


_prism_attention_class = None


def _fuse_prism_attention(model) -> None:
    """q|k|v behind one rotation, per instance: the tapped-layer trick from
    mlx_dflash.install_tap — swap the instance's class for a subclass whose
    __call__ is the stock attention body with the three projections replaced by
    the fused matmul. Everything else (norms, rope, the cache, the SDPA helper
    mlx_qsdpa patches) is the stock code path, resolved at call time."""
    global _prism_attention_class
    import mlx.core as mx
    from mlx_lm.models import qwen3_5 as q35
    from mlx_lm.models import qwen3_next as qn

    if _prism_attention_class is None:
        class _PrismAttention(q35.Attention):
            def __call__(self, x, mask=None, cache=None):
                B, L, _ = x.shape
                xr = prism_pack.rotate(x, self._fused_block)   # signs: in the norm
                big = mlx_qmm_mma.qmm(xr, self._fused_w, self._fused_s, self._fused_b,
                                      self._fused_gs, self._fused_bits)
                q_out, keys, values = mx.split(
                    big, [self._q_cols, self._q_cols + self._kv_cols], axis=-1)
                queries, gate = mx.split(
                    q_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
                gate = gate.reshape(B, L, -1)
                queries = self.q_norm(queries).transpose(0, 2, 1, 3)
                keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)
                                   ).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self.num_key_value_heads, -1
                                        ).transpose(0, 2, 1, 3)
                if cache is not None:
                    queries = self.rope(queries, offset=cache.offset)
                    keys = self.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = self.rope(queries)
                    keys = self.rope(keys)
                output = qn.scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=self.scale, mask=mask)
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output * mx.sigmoid(gate))

        _prism_attention_class = _PrismAttention

    for layer in model.language_model.model.layers:
        if layer.is_linear:
            continue
        at = layer.self_attn
        if isinstance(at, _prism_attention_class):
            continue
        at._q_cols = int(at.q_proj.weight.shape[0])
        at._kv_cols = int(at.k_proj.weight.shape[0])
        _record_fuse(at, [at.q_proj, at.k_proj, at.v_proj])
        at.__class__ = _prism_attention_class


def _signed(weight, signs):
    """weight * signs, exact in weight's dtype (signs are ±1)."""
    import mlx.core as mx

    out = (weight.astype(mx.float32) * signs.astype(mx.float32)).astype(weight.dtype)
    mx.eval(out)
    return out


def _prism_mlp_body(layer):
    """post_attention_layernorm + rotation + fused swiglu MLP + rotation + residual.
    Both sign vectors are already in the weights (`_fold_prism_signs`)."""
    import mlx.core as mx
    import mlx.nn as nn

    rot = prism_pack.rotate
    mlp = layer.mlp
    ln_w = layer.post_attention_layernorm.weight
    ln_eps = layer.post_attention_layernorm.eps
    fw, fs, fb = mlp._fused_w, mlp._fused_s, mlp._fused_b
    gs_, bits = mlp._fused_gs, mlp._fused_bits
    block_in = mlp._fused_block
    dp = mlp.down_proj
    block_mid = dp.block

    def fwd(h):
        x = rot(mx.fast.rms_norm(h, ln_w, ln_eps), block_in)
        gu = mx.quantized_matmul(x, fw, scales=fs, biases=fb, transpose=True,
                                 group_size=gs_, bits=bits)
        g, u = mx.split(gu, 2, axis=-1)
        y = rot(nn.silu(g) * u, block_mid)
        y = mx.quantized_matmul(y, dp.weight, scales=dp.scales, biases=dp.biases,
                                transpose=True, group_size=dp.group_size, bits=dp.bits)
        return h + y

    return fwd


def _prism_gdn_body(layer):
    """input_layernorm (signs folded at install) + rotation + fused qkv|z, the b|a
    gates off the same signed input (their weights re-signed to match) + the stock
    recurrence + gated norm (out_proj's signs riding on the gate) + rotation +
    out_proj + residual, with explicit (conv_state, recurrent_state) threading."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import gated_delta_update

    rot = prism_pack.rotate
    gd = layer.linear_attn
    ln_w = layer.input_layernorm.weight
    ln_eps = layer.input_layernorm.eps
    fw, fs, fb = gd._fused_w, gd._fused_s, gd._fused_b
    gs_, bits = gd._fused_gs, gd._fused_bits
    block_in = gd._fused_block
    w_ba = gd._fused_ba
    conv_w = gd.conv1d.weight
    n_keep = gd.conv_kernel_size - 1
    A_log, dt_bias = gd.A_log, gd.dt_bias
    op = gd.out_proj
    norm_w = gd.norm.weight
    # The gated norm is per head (head_v_dim wide) while out_proj's sign vector
    # spans every head, so these signs cannot fold into the norm weight; they ride
    # on the fp32 elementwise gate below, which compile fuses into one kernel anyway.
    s_out = op.signs.reshape(gd.num_v_heads, gd.head_v_dim).astype(mx.float32)
    mx.eval(s_out)
    block_out = op.block
    Hk, Hv = gd.num_k_heads, gd.num_v_heads
    Dk, Dv = gd.head_k_dim, gd.head_v_dim
    key_dim, conv_dim = gd.key_dim, gd.conv_dim
    eps = gd.layer_norm_epsilon

    def fwd(xin, conv_state, rec_state):
        inputs = mx.fast.rms_norm(xin, ln_w, ln_eps)
        B, S, _ = inputs.shape
        big = mx.quantized_matmul(rot(inputs, block_in), fw, scales=fs, biases=fb,
                                  transpose=True, group_size=gs_, bits=bits)
        qkv, z = mx.split(big, [conv_dim], axis=-1)
        b, a = mx.split(inputs @ w_ba.T, 2, axis=-1)
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
        xn = mx.fast.rms_norm(out, norm_w, eps).astype(mx.float32) * s_out
        out = (nn.silu(z.astype(mx.float32)) * xn).astype(xin.dtype)
        out = rot(out.reshape(B, S, -1), block_out)
        out = mx.quantized_matmul(out, op.weight, scales=op.scales, biases=op.biases,
                                  transpose=True, group_size=op.group_size,
                                  bits=op.bits)
        return xin + out, new_conv, new_rec

    return fwd
