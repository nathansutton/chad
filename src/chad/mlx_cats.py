"""CATS sparse dense-MLP decode for the Qwen3.8-27B dense hybrid (plan 132).

Exact-CATS: per decode token, compute the full 3-bit gate, keep only the MLP
intermediate channels whose |silu(gate)| clears a per-layer calibrated
threshold, then compute up_proj only on those rows (row-gather kernel) and
down_proj only on those columns (scaled-row-accumulate against a TRANSPOSED
4-bit down sidecar). Selection thresholds target keep-r of the 17408 channels;
measured quality holds at r=0.25 on this model (see plan 132 L9/L12).

Opt-in and inert by default: engages only when CHAD_CATS_R is set (e.g. 0.25),
the model directory carries `downT_4bit.safetensors` + `cats_thresholds.json`,
and the checkpoint matches the validated geometry (3-bit gs64 dense MLP,
K % 256 == 0, D % 32 == 0). The transposed sidecar REPLACES down_proj
everywhere — prefill/S>1 run it densely via quantized_matmul(transpose=False)
— so both phases see the same weights and the original down memory is freed
(net ~+0.7 GB for the 4-bit sidecar).

Interplay with mlx_fastpath: fastpath skips its gate|up concat for CATS models
(the sparse path needs gate dense and up gathered separately) and takes this
module's step body for the S=1 layer fast path.

bf16 note: activations are bf16 on this checkpoint; every kernel boundary
casts x/g to fp16 explicitly (metal_kernel signatures follow input dtype and a
half4 reinterpret of bf16 reads garbage silently — plan 132 L13b).
"""
from __future__ import annotations

import json
import os
from typing import Any

from . import config
from .diag import log

_KERNELS: dict[str, Any] = {}
_PARAMS: dict[tuple, Any] = {}

_HEADER = r"""
inline void unpack8_3bit(const device uchar* w, thread float4& lo,
                         thread float4& hi) {
  uchar b0 = w[0], b1 = w[1], b2 = w[2];
  lo = float4(float(b0 & 0x7), float((b0 >> 3) & 0x7),
              float(((b0 >> 6) | (b1 << 2)) & 0x7), float((b1 >> 1) & 0x7));
  hi = float4(float((b1 >> 4) & 0x7),
              float(((b1 >> 7) | (b2 << 1)) & 0x7),
              float((b2 >> 2) & 0x7), float((b2 >> 5) & 0x7));
}
inline void unpack8_4bit(const device uchar* w, thread float4& lo,
                         thread float4& hi) {
  lo = float4(float(w[0] & 0xF), float(w[0] >> 4),
              float(w[1] & 0xF), float(w[1] >> 4));
  hi = float4(float(w[2] & 0xF), float(w[2] >> 4),
              float(w[3] & 0xF), float(w[3] >> 4));
}
inline float silu_f(float v) { return v / (1.0f + exp(-v)); }
"""

# Threshold + stream compaction: pred per channel, simd-ballot prefix for the
# in-simd offset, one atomic add per simdgroup. Unordered across simdgroups
# (consumers are order-independent). Writes clamp at cap; cnt may exceed it,
# consumers take min(cnt, cap).
_SRC_COMPACT = r"""
    uint i = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    const uint n = params[0];
    const uint cap = params[1];
    bool pred = false;
    if (i < n) pred = fabs(silu_f(float(g[i]))) >= thr[0];
    ulong m = (ulong)simd_ballot(pred);
    uint tot = popcount(m);
    uint off = popcount(m & ((ulong(1) << lane) - 1));
    uint base = 0;
    if (lane == 0 && tot > 0)
      base = atomic_fetch_add_explicit(&cnt[0], tot, memory_order_relaxed);
    base = simd_broadcast_first(base);
    if (pred) {
      uint pos = base + off;
      if (pos < cap)
        atomic_store_explicit(&idx[pos], i, memory_order_relaxed);
    }
"""

# 32 3-bit values are exactly 96 bits, so one 12-byte load per lane covers a
# whole unpack unit and always lands inside a single 64-wide quant group (the
# lane's span starts at a multiple of 32) -- one scale/bias fetch per row per
# iteration. The byte offset is lane*12 into a K*3/8 row, so the uint3 load is
# 4-byte aligned for every lane. Needs K % 1024 == 0.
_UNPACK32 = r"""
#define UNPACK32(w0, w1, w2, Q) {                                    \
  ulong a01 = (ulong)(w0) | ((ulong)(w1) << 32);                     \
  ulong a12 = (ulong)(w1) | ((ulong)(w2) << 32);                     \
  Q[0] = float4(float(a01 & 7), float((a01 >> 3) & 7),               \
                float((a01 >> 6) & 7), float((a01 >> 9) & 7));       \
  Q[1] = float4(float((a01 >> 12) & 7), float((a01 >> 15) & 7),      \
                float((a01 >> 18) & 7), float((a01 >> 21) & 7));     \
  Q[2] = float4(float((a01 >> 24) & 7), float((a01 >> 27) & 7),      \
                float((a01 >> 30) & 7), float((a12 >> 1) & 7));      \
  Q[3] = float4(float((a12 >> 4) & 7), float((a12 >> 7) & 7),        \
                float((a12 >> 10) & 7), float((a12 >> 13) & 7));     \
  Q[4] = float4(float((a12 >> 16) & 7), float((a12 >> 19) & 7),      \
                float((a12 >> 22) & 7), float((a12 >> 25) & 7));     \
  Q[5] = float4(float((a12 >> 28) & 7), float((a12 >> 31) & 7),      \
                float((a12 >> 34) & 7), float((a12 >> 37) & 7));     \
  Q[6] = float4(float((a12 >> 40) & 7), float((a12 >> 43) & 7),      \
                float((a12 >> 46) & 7), float((a12 >> 49) & 7));     \
  Q[7] = float4(float((a12 >> 52) & 7), float((a12 >> 55) & 7),      \
                float((a12 >> 58) & 7), float((a12 >> 61) & 7));     \
}
"""

# Row-gather up_proj with silu(gate) folded in: h[j] = silu(g[idx[j]]) *
# dot(dequant3(Wu[idx[j], :]), x). NSIMD simdgroups x RPS rows each.
_SRC_GATHER_U = r"""
    uint tg = threadgroup_position_in_grid.x;
    uint simd = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    const uint K = params[0];
    const uint cap = params[1];
    uint nact = min(cnt[0], cap);
    uint j0 = (tg * NSIMD + simd) * RPS;
    if (j0 >= nact) return;
    const device uchar* wb = (const device uchar*)wu;
    const uint row_bytes = (K * 3) / 8;
    const uint gstride = K / 64;
    float acc[RPS];
    uint rows[RPS];
    #pragma clang loop unroll(full)
    for (int r = 0; r < RPS; ++r) {
      acc[r] = 0.f;
      rows[r] = (j0 + r < nact) ? idx[j0 + r] : 0xffffffff;
    }
    const uint iters = K / 1024;
    for (uint it = 0; it < iters; ++it) {
      uint v0 = it * 1024 + lane * 32;
      uint gg = v0 / 64;
      const device half4* xp = (const device half4*)(x + v0);
      float4 X[8];
      float4 xsv = float4(0.f);
      #pragma clang loop unroll(full)
      for (int t = 0; t < 8; ++t) { X[t] = float4(xp[t]); xsv += X[t]; }
      float xs = xsv.x + xsv.y + xsv.z + xsv.w;
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPS; ++r) {
        uint row = rows[r];
        if (row == 0xffffffff) continue;
        const device uint* wp =
            (const device uint*)(wb + row * row_bytes + (v0 * 3) / 8);
        float4 Q[8];
        UNPACK32(wp[0], wp[1], wp[2], Q)
        float4 d = float4(0.f);
        #pragma clang loop unroll(full)
        for (int t = 0; t < 8; ++t) d = fma(Q[t], X[t], d);
        uint soff = row * gstride + gg;
        acc[r] += float(su[soff]) * (d.x + d.y + d.z + d.w)
                + float(bu[soff]) * xs;
      }
    }
    #pragma clang loop unroll(full)
    for (int r = 0; r < RPS; ++r) {
      float v = simd_sum(acc[r]);
      if (lane == 0 && j0 + r < nact)
        h[j0 + r] = half(v * silu_f(float(g[rows[r]])));
    }
"""

# Scaled row-accumulate against the transposed 4-bit down: y = sum_j h[j] *
# dequant4(WdT[idx[j], :]). RCH row-chunks x (D / CPT) threads, each owning
# CPT output columns; fp32 partials reduced by the second kernel. RCH and CPT
# move together so the chunk count can drop without dropping thread count --
# the partials round trip (2 * RCH * D * 4 bytes) is pure overhead on top of
# the weights, and at RCH=128/CPT=32 it was a quarter of the kernel's traffic.
_SRC_ACC4 = r"""
    uint chunk = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    const uint D = params[0];
    const uint cap = params[1];
    const uint nchunks = params[2];
    uint nact = min(cnt[0], cap);
    const uint row_bytes = D / 2;
    const uint gstride = D / 64;
    const device uchar* wb = (const device uchar*)w;
    uint per = (nact + nchunks - 1) / nchunks;
    uint j0 = chunk * per;
    uint j1 = min(j0 + per, nact);
    uint c0 = t * CPT;
    float4 acc[CPT / 4];
    #pragma clang loop unroll(full)
    for (int q = 0; q < CPT / 4; ++q) acc[q] = float4(0.f);
    for (uint j = j0; j < j1; ++j) {
      uint row = idx[j];
      float hc = float(h[j]);
      const device uchar* wr = wb + row * row_bytes + c0 / 2;
      const device half* sr = scales + row * gstride;
      const device half* br = biases + row * gstride;
      #pragma clang loop unroll(full)
      for (int q8 = 0; q8 < CPT / 8; ++q8) {
        uint gq = (c0 + q8 * 8) / 64;
        float hs = hc * float(sr[gq]);
        float hb = hc * float(br[gq]);
        float4 qlo, qhi;
        unpack8_4bit(wr + q8 * 4, qlo, qhi);
        acc[2*q8]   = fma(float4(hs), qlo, acc[2*q8]   + hb);
        acc[2*q8+1] = fma(float4(hs), qhi, acc[2*q8+1] + hb);
      }
    }
    device float* out = partials + chunk * D + c0;
    #pragma clang loop unroll(full)
    for (int q = 0; q < CPT / 4; ++q) {
      out[q*4+0] = acc[q].x; out[q*4+1] = acc[q].y;
      out[q*4+2] = acc[q].z; out[q*4+3] = acc[q].w;
    }
"""

_SRC_REDUCE = r"""
    uint i = thread_position_in_grid.x;
    const uint D = params[0];
    const uint nchunks = params[2];
    if (i >= D) return;
    float s = 0.0f;
    for (uint c = 0; c < nchunks; ++c) s += partials[c * D + i];
    y[i] = half(s);
"""

# Kernel shape tunables, all measured end-to-end on the 27B (see plan 132).
# RPS/NSIMD: rows per simdgroup and simdgroups per threadgroup in the gather.
# RCH/CPT: row chunks and output columns per thread in the accumulate.
RPS = config.env_int("CHAD_CATS_RPS") or 4
NSIMD = config.env_int("CHAD_CATS_NS") or 1
RCH = config.env_int("CHAD_CATS_RCH") or 32
CPT = config.env_int("CHAD_CATS_CPT") or 8


def _kernels():
    if _KERNELS:
        return _KERNELS
    import mlx.core as mx

    defs = (f"\n#define RPS {RPS}\n#define NSIMD {NSIMD}\n"
            f"#define CPT {CPT}\n")
    head = _HEADER + _UNPACK32 + defs
    _KERNELS["compact"] = mx.fast.metal_kernel(
        name="chad_cats_compact", input_names=["g", "thr", "params"],
        output_names=["idx", "cnt"], header=head, source=_SRC_COMPACT,
        atomic_outputs=True)
    _KERNELS["gather_u"] = mx.fast.metal_kernel(
        name="chad_cats_gather_u",
        input_names=["x", "g", "wu", "su", "bu", "idx", "cnt", "params"],
        output_names=["h"], header=head, source=_SRC_GATHER_U)
    _KERNELS["acc4"] = mx.fast.metal_kernel(
        name="chad_cats_acc4",
        input_names=["h", "w", "scales", "biases", "idx", "cnt", "params"],
        output_names=["partials"], header=head, source=_SRC_ACC4)
    _KERNELS["reduce"] = mx.fast.metal_kernel(
        name="chad_cats_reduce", input_names=["partials", "params"],
        output_names=["y"], header=head, source=_SRC_REDUCE)
    return _KERNELS


def _params(*vals):
    """Cached uint32 param vectors. Building these per call cost ~3 host-side
    array constructions per MLP per token (192 a token on the 27B)."""
    import mlx.core as mx

    key = vals
    p = _PARAMS.get(key)
    if p is None:
        p = mx.array(list(vals), dtype=mx.uint32)
        mx.eval(p)
        _PARAMS[key] = p
    return p


def compact(g, thr, cap):
    """g: fp16 (n,) gate pre-activations; thr: fp32 (1,). -> idx (cap,), cnt (1,)."""
    import mlx.core as mx

    n = g.shape[0]
    return _kernels()["compact"](
        inputs=[g, thr, _params(n, cap)], grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(cap,), (1,)], output_dtypes=[mx.uint32, mx.uint32],
        init_value=0)


def gather_u(x, g, wu, su, bu, idx, cnt, cap, K):
    """h[j] = silu(g[idx[j]]) * (dequant3(Wu) @ x)[idx[j]]; x fp16 (K,)."""
    import mlx.core as mx

    per = RPS * NSIMD
    (h,) = _kernels()["gather_u"](
        inputs=[x, g, wu, su, bu, idx, cnt, _params(K, cap)],
        grid=(((cap + per - 1) // per) * NSIMD * 32, 1, 1),
        threadgroup=(NSIMD * 32, 1, 1),
        output_shapes=[(cap,)], output_dtypes=[mx.float16])
    return h


def row_acc4(h, w, sc, bi, idx, cnt, cap, D):
    """y = sum_j h[j] * dequant4(WdT[idx[j], :]); WdT is (I, D) 4-bit gs64."""
    import mlx.core as mx

    params = _params(D, cap, RCH)
    (p,) = _kernels()["acc4"](
        inputs=[h, w, sc, bi, idx, cnt, params],
        grid=(RCH * (D // CPT), 1, 1), threadgroup=(D // CPT, 1, 1),
        output_shapes=[(RCH * D,)], output_dtypes=[mx.float32])
    (y,) = _kernels()["reduce"](
        inputs=[p, params], grid=(D, 1, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(D,)], output_dtypes=[mx.float16])
    return y


def enabled() -> bool:
    return config.env_float("CHAD_CATS_R") is not None


def install(model, model_path) -> bool:
    """Wire CATS into a loaded dense-hybrid model. Returns True on success.

    Loads the downT 4-bit sidecar + per-layer thresholds, frees the original
    down_proj weights, and patches the dense MLP __call__ (S>1 path). The S=1
    fast path is provided via step_body() and consumed by mlx_fastpath.
    """
    import mlx.core as mx

    r = config.env_float("CHAD_CATS_R")
    if r is None or model_path is None:
        return False
    side = os.path.join(str(model_path), "downT_4bit.safetensors")
    thrf = os.path.join(str(model_path), "cats_thresholds.json")
    if not (os.path.exists(side) and os.path.exists(thrf)):
        log.info("CATS requested but sidecars missing (%s); staying dense",
                 side)
        return False
    thresholds = json.load(open(thrf))["exact"]
    rkey = str(r)
    if rkey not in next(iter(thresholds.values())):
        log.warning("CATS: no thresholds for r=%s; staying dense", rkey)
        return False

    layers = model.language_model.model.layers
    mlp0 = layers[0].mlp
    K = mlp0.gate_proj.scales.shape[1] * 64
    I = mlp0.gate_proj.scales.shape[0]
    # K % 1024: the gather's 32-values-per-lane unit spans a 32-lane simdgroup.
    if (mlp0.gate_proj.bits != 3 or mlp0.gate_proj.group_size != 64
            or K % 1024 or I % 8):
        log.warning("CATS: unsupported MLP geometry; staying dense")
        return False
    sw = mx.load(side)
    D = sw["model.layers.0.mlp.downT.scales"].shape[1] * 64
    if D % (CPT * 8) or D // CPT > 1024:
        log.warning("CATS: down width %d unsupported at CPT=%d", D, CPT)
        return False
    cap = min(I, int(2 * r * I) // 8 * 8)

    z = mx.zeros((8,), dtype=mx.uint32)
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        base = f"model.layers.{i}.mlp.downT"
        dT = (sw[f"{base}.weight"],
              sw[f"{base}.scales"].astype(mx.float16),
              sw[f"{base}.biases"].astype(mx.float16))
        mx.eval(*dT)
        mlp._cats = {
            "dT": dT, "thr": mx.array([thresholds[str(i)][rkey]],
                                      dtype=mx.float32),
            "cap": cap, "K": K, "D": D,
            "su": mlp.up_proj.scales.astype(mx.float16),
            "bu": mlp.up_proj.biases.astype(mx.float16),
        }
        mx.eval(mlp._cats["thr"], mlp._cats["su"], mlp._cats["bu"])
        dp = mlp.down_proj
        dp.weight = z
        dp.scales = z
        dp.biases = z
    mx.clear_cache()
    _patch_mlp_call()
    log.info("CATS installed: r=%s cap=%d, downT 4-bit replaces down_proj",
             rkey, cap)
    return True


def _patch_mlp_call() -> None:
    """S>1/prefill (and any stock fallback): dense g,u + dense transposed down.
    Foreign MLP instances (no _cats) keep the stock path."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5 as q35

    if getattr(q35.MLP.__call__, "_chad_cats", False):
        return
    stock = q35.MLP.__call__

    def call(self, x):
        c = getattr(self, "_cats", None)
        if c is None:
            return stock(self, x)
        h = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        q, s, b = c["dT"]
        return mx.quantized_matmul(
            h, q, scales=s, biases=b, transpose=False, group_size=64, bits=4
        ).astype(x.dtype)

    call._chad_cats = True  # type: ignore[attr-defined]
    q35.MLP.__call__ = call  # type: ignore[method-assign]


def step_body(layer):
    """S=1 decode body: post_attention_layernorm + sparse CATS MLP + residual.

    Uncompiled (custom metal kernels inside); mirrors mlx_fastpath's
    _dense_mlp_body contract: h -> h + mlp(norm(h))."""
    import mlx.core as mx

    mlp = layer.mlp
    c = mlp._cats
    ln_w = layer.post_attention_layernorm.weight
    ln_eps = layer.post_attention_layernorm.eps
    gp, up = mlp.gate_proj, mlp.up_proj
    dq, ds, db = c["dT"]
    thr, cap, K, D = c["thr"], c["cap"], c["K"], c["D"]
    su, bu = c["su"], c["bu"]

    def fwd(h):
        x = mx.fast.rms_norm(h, ln_w, ln_eps)
        g = mx.quantized_matmul(
            x, gp.weight, scales=gp.scales, biases=gp.biases, transpose=True,
            group_size=64, bits=3)[0, 0].astype(mx.float16)
        x16 = x[0, 0].astype(mx.float16)
        idx, cnt = compact(g, thr, cap)
        hact = gather_u(x16, g, up.weight, su, bu, idx, cnt, cap, K)
        y = row_acc4(hact, dq, ds, db, idx, cnt, cap, D)
        return h + y.reshape(1, 1, -1).astype(h.dtype)

    return fwd
