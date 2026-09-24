"""GGUF block dequantizers as MLX Metal device functions.

A GGUF weight row is a run of fixed-size blocks along the input dim K; every format
here is a 256-value super-block except Q8_0 (32 values). The block layouts, codebooks
and decode arithmetic are ported from llama.cpp (``ggml/src/ggml-common.h``,
``ggml-quants.c``, ``ggml-metal.metal``; MIT, see NOTICE).

The unit every kernel is built from is one 32-value chunk of one row:

    static inline void deq32_<fmt>(device const uint8_t* row, uint c, thread float* v)

writes ``row[32c .. 32c+31]`` as float32, ``row`` pointing at the row's first block
byte. 32 is the smallest span every format decodes without partial scale groups: an
IQ sub-block, a K-quant 32-value group, or two 16-value sub-blocks (Q2_K/Q3_K/Q6_K).
Fused matmul kernels call it in their inner loop; :func:`dequantize` is the plain
one-thread-per-chunk expansion the tests hold it to.

Numerics: every value is computed in float32 in the same operation order as the
numpy decoders in ``gguf.quants`` — e.g. Q4_K is ``(d*sc)*q - dmin*m`` with the
product rounded before the subtraction — so float32 output is bit-identical to them.
Contracting ``a*b - c`` into one fma, or reassociating ``(d*sc)*q`` into
``d*(sc*q)``, would change the rounding; today's Metal compiler does neither to this
code even without being told, but fast math permits both, so the header switches them
off for everything that follows it rather than leave exactness to the optimizer.

The codebooks are generated from ``gguf.quants``' packed grids (the same numbers as
ggml-common.h, 2 bits per entry instead of one byte) and emitted as 32-bit words, four
grid bytes per word, so a grid row is one or two word loads. They stay in ``constant``
memory: copying them into threadgroup memory at kernel start, as llama.cpp's kernels do,
measured flat to slightly slower on every format here (M4 Pro).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Union, cast

import numpy as np

if TYPE_CHECKING:  # mlx is imported lazily inside the functions so the module loads on Linux
    import mlx.core as mx


@dataclass(frozen=True)
class GGUFFormat:
    name: str            # ggml type name, lower-cased into the deq32_<name> function
    block_values: int    # weights per block
    block_bytes: int     # bytes per block

    @property
    def deq_fn(self) -> str:
        return f"deq32_{self.name.lower()}"


# Keyed by the ggml type id as stored in the GGUF tensor header.
FORMATS: dict[int, GGUFFormat] = {
    8: GGUFFormat("Q8_0", 32, 34),
    10: GGUFFormat("Q2_K", 256, 84),
    11: GGUFFormat("Q3_K", 256, 110),
    12: GGUFFormat("Q4_K", 256, 144),
    13: GGUFFormat("Q5_K", 256, 176),
    14: GGUFFormat("Q6_K", 256, 210),
    16: GGUFFormat("IQ2_XXS", 256, 66),
    17: GGUFFormat("IQ2_XS", 256, 74),
    18: GGUFFormat("IQ3_XXS", 256, 98),
    19: GGUFFormat("IQ1_S", 256, 50),
    20: GGUFFormat("IQ4_NL", 32, 18),
    21: GGUFFormat("IQ3_S", 256, 110),
    22: GGUFFormat("IQ2_S", 256, 82),
    23: GGUFFormat("IQ4_XS", 256, 136),
    29: GGUFFormat("IQ1_M", 256, 56),
}

# Every block size is even, so each block (and each 2-byte field in it) sits on a
# 2-byte boundary of a contiguous row: ushort loads are aligned, wider fields are
# assembled from them.
_DEVICE_FNS = r"""
static inline float gg_f16(device const uint8_t* p) {
    return float(as_type<half>(*(device const ushort*)p));
}
static inline uint gg_u16(device const uint8_t* p) { return *(device const ushort*)p; }
static inline uint gg_u32(device const uint8_t* p) { return gg_u16(p) | (gg_u16(p + 2) << 16); }

static inline void deq32_q8_0(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 34u * c;
    const float d = gg_f16(b);
    device const char* q = (device const char*)(b + 2);
    for (int i = 0; i < 32; ++i) v[i] = float(q[i]) * d;
}

// Q2_K: scales[16] | qs[64] | d | dmin. Chunk j reads bit pair (j&3) of qs half j>>2.
static inline void deq32_q2_k(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 84u * (c >> 3);
    const uint j = c & 7u;
    const float d = gg_f16(b + 80), dmin = gg_f16(b + 82);
    device const uint8_t* q = b + 16 + (j >> 2) * 32;
    const uint sh = 2u * (j & 3u);
    for (int h = 0; h < 2; ++h) {
        const uint sc = b[2 * j + h];
        const float dl = d * float(sc & 15u);
        const float ml = dmin * float(sc >> 4);
        for (int i = 0; i < 16; ++i) v[16 * h + i] = dl * float((q[16 * h + i] >> sh) & 3u) - ml;
    }
}

// Q3_K: hmask[32] | qs[64] | scales[12] (16 packed 6-bit, offset 32) | d. The high bit
// is set when the 3-bit value is NON-negative: q = low2 - 4 * !hbit.
static inline void deq32_q3_k(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 110u * (c >> 3);
    const uint j = c & 7u;
    const float d = gg_f16(b + 108);
    device const uint8_t* q = b + 32 + (j >> 2) * 32;
    const uint sh = 2u * (j & 3u);
    for (int h = 0; h < 2; ++h) {
        const uint s = 2 * j + h;
        const uint lo = (b[96 + (s & 7u)] >> (4u * (s >> 3))) & 15u;
        const uint hi = (b[104 + (s & 3u)] >> (2u * (s >> 2))) & 3u;
        const float dl = d * float(int(lo | (hi << 4)) - 32);
        for (int i = 0; i < 16; ++i) {
            const int n = 16 * h + i;
            const int ql = int((q[n] >> sh) & 3u);
            const int hb = int((b[n] >> j) & 1u);
            v[n] = dl * float(ql - ((hb ^ 1) << 2));
        }
    }
}

// Q4_K/Q5_K share the 12-byte block of eight 6-bit (scale, min) pairs.
static inline void gg_scale_min_k4(device const uint8_t* s, uint j, thread uint& sc, thread uint& m) {
    if (j < 4) {
        sc = s[j] & 63u;
        m = s[j + 4] & 63u;
    } else {
        sc = (s[j + 4] & 15u) | ((s[j - 4] >> 2) & 48u);
        m = (s[j + 4] >> 4) | ((s[j] >> 2) & 48u);
    }
}

// Q4_K: d | dmin | scales[12] | qs[128]; chunk j is nibble (j&1) of qs[32*(j>>1) ..].
static inline void deq32_q4_k(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 144u * (c >> 3);
    const uint j = c & 7u;
    uint sc, m;
    gg_scale_min_k4(b + 4, j, sc, m);
    const float dl = gg_f16(b) * float(sc);
    const float ml = gg_f16(b + 2) * float(m);
    device const uint8_t* q = b + 16 + (j >> 1) * 32;
    const uint sh = 4u * (j & 1u);
    for (int i = 0; i < 32; ++i) v[i] = dl * float((q[i] >> sh) & 15u) - ml;
}

// Q5_K: d | dmin | scales[12] | qh[32] | qs[128]; the fifth bit of chunk j is bit j of qh.
static inline void deq32_q5_k(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 176u * (c >> 3);
    const uint j = c & 7u;
    uint sc, m;
    gg_scale_min_k4(b + 4, j, sc, m);
    const float dl = gg_f16(b) * float(sc);
    const float ml = gg_f16(b + 2) * float(m);
    device const uint8_t* qh = b + 16;
    device const uint8_t* q = b + 48 + (j >> 1) * 32;
    const uint sh = 4u * (j & 1u);
    for (int i = 0; i < 32; ++i) {
        const uint x = ((q[i] >> sh) & 15u) | (((qh[i] >> j) & 1u) << 4);
        v[i] = dl * float(x) - ml;
    }
}

// Q6_K: ql[128] | qh[64] | scales[16] (int8) | d. Each 128-value half takes 64 ql
// bytes (both nibbles) and 32 qh bytes (four bit pairs).
static inline void deq32_q6_k(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 210u * (c >> 3);
    const uint j = c & 7u, jj = j & 3u;
    const float d = gg_f16(b + 208);
    device const uint8_t* ql = b + (j >> 2) * 64 + (jj & 1u) * 32;
    device const uint8_t* qh = b + 128 + (j >> 2) * 32;
    device const char* scales = (device const char*)(b + 192);
    const uint lsh = 4u * (jj >> 1), hsh = 2u * jj;
    for (int h = 0; h < 2; ++h) {
        const float dl = d * float(scales[2 * j + h]);
        for (int i = 0; i < 16; ++i) {
            const int n = 16 * h + i;
            const uint x = ((ql[n] >> lsh) & 15u) | (((qh[n] >> hsh) & 3u) << 4);
            v[n] = dl * float(int(x) - 32);
        }
    }
}

// IQ4_XS: d | scales_h (u16, 2 bits per chunk) | scales_l[4] (nibbles) | qs[128];
// a chunk is 16 bytes, low nibbles first, through the non-linear kvalues_iq4nl table.
static inline void deq32_iq4_xs(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 136u * (c >> 3);
    const uint j = c & 7u;
    const uint lo = (b[4 + (j >> 1)] >> (4u * (j & 1u))) & 15u;
    const uint hi = (gg_u16(b + 2) >> (2u * j)) & 3u;
    const float dl = gg_f16(b) * float(int(lo | (hi << 4)) - 32);
    device const uint8_t* q = b + 8 + 16 * j;
    for (int i = 0; i < 16; ++i) {
        v[i] = dl * float(kvalues_iq4nl[q[i] & 15u]);
        v[i + 16] = dl * float(kvalues_iq4nl[q[i] >> 4]);
    }
}

// IQ4_NL: d | qs[16]: one 32-value block per chunk on the same 16-entry table as IQ4_XS.
static inline void deq32_iq4_nl(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 18u * c;
    const float d = gg_f16(b);
    for (int i = 0; i < 16; ++i) {
        v[i] = d * float(kvalues_iq4nl[b[2 + i] & 15u]);
        v[i + 16] = d * float(kvalues_iq4nl[b[2 + i] >> 4]);
    }
}

// Eight grid bytes starting at word 2*idx of an 8-wide codebook, each scaled by db and
// negated where the sign byte has its bit set.
static inline void gg_grid8(constant uint* grid, uint idx, uint signs, float db, thread float* v) {
    const uchar4 g0 = as_type<uchar4>(grid[2 * idx]);
    const uchar4 g1 = as_type<uchar4>(grid[2 * idx + 1]);
    for (int k = 0; k < 4; ++k) {
        const float a = db * float(g0[k]);
        const float e = db * float(g1[k]);
        v[k] = (signs >> k) & 1u ? -a : a;
        v[k + 4] = (signs >> (k + 4)) & 1u ? -e : e;
    }
}

// IQ2_XXS: d | 8 x (u32 grid indices, u32 4x7-bit sign indices + 4-bit scale).
static inline void deq32_iq2_xxs(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 66u * (c >> 3);
    const uint j = c & 7u;
    device const uint8_t* p = b + 2 + 8 * j;
    const uint aux1 = gg_u32(p + 4);
    const float db = gg_f16(b) * (0.5f + float(aux1 >> 28)) * 0.25f;
    for (int l = 0; l < 4; ++l)
        gg_grid8(iq2xxs_grid, p[l], ksigns_iq2xs[(aux1 >> (7 * l)) & 127u], db, v + 8 * l);
}

// IQ2_XS: d | qs[32] (u16: 9-bit grid index, 7-bit sign index) | scales[8] (nibbles,
// one per 16 values).
static inline void deq32_iq2_xs(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 74u * (c >> 3);
    const uint j = c & 7u;
    const float d = gg_f16(b);
    const uint sc = b[66 + j];
    for (int l = 0; l < 4; ++l) {
        const uint q = gg_u16(b + 2 + 8 * j + 2 * l);
        const float db = d * (0.5f + float((sc >> (4 * (l >> 1))) & 15u)) * 0.25f;
        gg_grid8(iq2xs_grid, q & 511u, ksigns_iq2xs[q >> 9], db, v + 8 * l);
    }
}

// IQ2_S: d | qs[32] (grid index low bytes) | signs[32] | qh[8] (2 high index bits each)
// | scales[8].
static inline void deq32_iq2_s(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 82u * (c >> 3);
    const uint j = c & 7u;
    const float d = gg_f16(b);
    const uint sc = b[74 + j];
    const uint qh = b[66 + j];
    for (int l = 0; l < 4; ++l) {
        const uint idx = b[2 + 4 * j + l] | (((qh >> (2 * l)) & 3u) << 8);
        const float db = d * (0.5f + float((sc >> (4 * (l >> 1))) & 15u)) * 0.25f;
        gg_grid8(iq2s_grid, idx, b[34 + 4 * j + l], db, v + 8 * l);
    }
}

// Four grid bytes of a 4-wide codebook, scaled by db, signed by the low 4 bits of signs.
static inline void gg_grid4(constant uint* grid, uint idx, uint signs, float db, thread float* v) {
    const uchar4 g = as_type<uchar4>(grid[idx]);
    for (int k = 0; k < 4; ++k) {
        const float a = db * float(g[k]);
        v[k] = (signs >> k) & 1u ? -a : a;
    }
}

// IQ3_XXS: d | qs[64] (8-bit grid indices, 4 values each) | 8 x u32 (4x7-bit sign
// indices + 4-bit scale).
static inline void deq32_iq3_xxs(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 98u * (c >> 3);
    const uint j = c & 7u;
    device const uint8_t* q = b + 2 + 8 * j;
    const uint aux = gg_u32(b + 66 + 4 * j);
    const float db = gg_f16(b) * (0.5f + float(aux >> 28)) * 0.5f;
    for (int l = 0; l < 4; ++l) {
        const uint s = ksigns_iq2xs[(aux >> (7 * l)) & 127u];
        gg_grid4(iq3xxs_grid, q[2 * l], s, db, v + 8 * l);
        gg_grid4(iq3xxs_grid, q[2 * l + 1], s >> 4, db, v + 8 * l + 4);
    }
}

// IQ3_S: d | qs[64] | qh[8] (9th index bit) | signs[32] | scales[4] (nibbles).
static inline void deq32_iq3_s(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 110u * (c >> 3);
    const uint j = c & 7u;
    device const uint8_t* q = b + 2 + 8 * j;
    const uint qh = b[66 + j];
    const uint sc = (b[106 + (j >> 1)] >> (4u * (j & 1u))) & 15u;
    const float db = gg_f16(b) * float(1u + 2u * sc);
    for (int l = 0; l < 4; ++l) {
        const uint s = b[74 + 4 * j + l];
        gg_grid4(iq3s_grid, q[2 * l] | (((qh >> (2 * l)) & 1u) << 8), s, db, v + 8 * l);
        gg_grid4(iq3s_grid, q[2 * l + 1] | (((qh >> (2 * l + 1)) & 1u) << 8), s >> 4, db,
                 v + 8 * l + 4);
    }
}

// Eight ternary grid values of iq1s_grid, each dl * (g + delta).
static inline void gg_grid1(uint idx, float delta, float dl, thread float* v) {
    const char4 g0 = as_type<char4>(iq1s_grid[2 * idx]);
    const char4 g1 = as_type<char4>(iq1s_grid[2 * idx + 1]);
    for (int k = 0; k < 4; ++k) {
        v[k] = dl * (float(g0[k]) + delta);
        v[k + 4] = dl * (float(g1[k]) + delta);
    }
}

// IQ1_S: d | qs[32] | qh[8] (u16: 4x3 high index bits, 3-bit scale, delta sign).
static inline void deq32_iq1_s(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 50u * (c >> 3);
    const uint j = c & 7u;
    const uint qh = gg_u16(b + 34 + 2 * j);
    const float dl = gg_f16(b) * float(2u * ((qh >> 12) & 7u) + 1u);
    const float delta = qh & 0x8000u ? -IQ1S_DELTA : IQ1S_DELTA;
    for (int l = 0; l < 4; ++l)
        gg_grid1(b[2 + 4 * j + l] | (((qh >> (3 * l)) & 7u) << 8), delta, dl, v + 8 * l);
}

// IQ1_M: qs[32] | qh[16] (per index: 3 high bits + delta sign) | scales[8] (4 x u16:
// twelve 3-bit sub-scales, and the fp16 d spread over their top nibbles).
static inline void deq32_iq1_m(device const uint8_t* row, uint c, thread float* v) {
    device const uint8_t* b = row + 56u * (c >> 3);
    const uint j = c & 7u;
    device const uint8_t* sp = b + 48;
    const uint dbits = (gg_u16(sp) >> 12) | ((gg_u16(sp + 2) >> 8) & 0x00f0u)
                     | ((gg_u16(sp + 4) >> 4) & 0x0f00u) | (gg_u16(sp + 6) & 0xf000u);
    const float d = float(as_type<half>(ushort(dbits)));
    const uint sc = gg_u16(sp + 2 * (j >> 1));
    for (int l = 0; l < 4; ++l) {
        const uint s = (sc >> (6 * (j & 1u) + 3 * (l >> 1))) & 7u;
        const float dl = d * float(2u * s + 1u);
        const uint qh = b[32 + 2 * j + (l >> 1)] >> (4 * (l & 1));
        const float delta = qh & 8u ? -IQ1S_DELTA : IQ1S_DELTA;
        gg_grid1(b[4 * j + l] | ((qh & 7u) << 8), delta, dl, v + 8 * l);
    }
}
"""


def _grid_words(grid: Optional[np.ndarray]) -> str:
    """One gguf.quants IQ codebook as comma-separated uint32 words, four grid bytes each
    in little-endian order (int8 values keep their two's-complement byte)."""
    assert grid is not None, "call the quant class's init_grid() first"
    raw = np.asarray(grid).reshape(-1).astype(np.int8).view(np.uint8)
    words = np.frombuffer(raw.tobytes(), dtype="<u4")
    return ",".join(f"0x{int(w):08x}u" for w in words)


_header: Optional[str] = None


def metal_header() -> str:
    """Codebooks and every ``deq32_*`` device function, for ``metal_kernel(header=...)``."""
    global _header
    if _header is None:
        from gguf import quants

        ksigns = ",".join(str(x) for x in quants.IQ2_XXS.ksigns)
        kvalues = ",".join(str(x) for x in quants.IQ4_NL.kvalues)
        tables = [
            "#pragma clang fp contract(off)",
            "#pragma clang fp reassociate(off)",
            "#define IQ1S_DELTA 0.125f",
            f"constant uchar ksigns_iq2xs[128] = {{{ksigns}}};",
            f"constant char kvalues_iq4nl[16] = {{{kvalues}}};",
        ]
        quants.IQ2_XXS.init_grid()
        quants.IQ2_XS.init_grid()
        quants.IQ2_S.init_grid()
        quants.IQ3_XXS.init_grid()
        quants.IQ3_S.init_grid()
        quants.IQ1_S.init_grid()
        for name, grid in (("iq2xxs_grid", quants.IQ2_XXS.grid), ("iq2xs_grid", quants.IQ2_XS.grid),
                           ("iq2s_grid", quants.IQ2_S.grid), ("iq3xxs_grid", quants.IQ3_XXS.grid),
                           ("iq3s_grid", quants.IQ3_S.grid), ("iq1s_grid", quants.IQ1_S.grid)):
            tables.append(f"constant uint {name}[] = {{{_grid_words(grid)}}};")
        _header = "\n".join(tables) + "\n" + _DEVICE_FNS
    return _header


_SRC = r"""
    const uint c = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (c >= (uint)(KD / 32)) return;
    float v[32];
    DEQ(w + (size_t)r * ROWB, c, v);
    device T* o = out + (size_t)r * KD + 32 * c;
    for (int i = 0; i < 32; ++i) o[i] = static_cast<T>(v[i]);
"""

class _MetalKernel(Protocol):
    def __call__(self, *, inputs: list["mx.array"],
                 template: list[tuple[str, Union[int, "mx.Dtype"]]],
                 grid: tuple[int, int, int], threadgroup: tuple[int, int, int],
                 output_shapes: list[tuple[int, ...]],
                 output_dtypes: list["mx.Dtype"]) -> list["mx.array"]: ...


_kernels: dict[int, _MetalKernel] = {}


def _kernel(qtype: int) -> _MetalKernel:
    k = _kernels.get(qtype)
    if k is None:
        import mlx.core as mx
        fmt = FORMATS[qtype]
        # SAFETY: the stub types metal_kernel's result as a bare object; it is the
        # keyword-called kernel _MetalKernel describes.
        k = cast(_MetalKernel, mx.fast.metal_kernel(
            name=f"chad_gguf_deq_{fmt.name.lower()}",
            input_names=["w"],
            output_names=["out"],
            source=_SRC.replace("DEQ", fmt.deq_fn),
            header=metal_header(),
        ))
        _kernels[qtype] = k
    return k


def row_bytes(qtype: int, k: int) -> int:
    """Bytes in one weight row of ``k`` values."""
    fmt = FORMATS[qtype]
    if k % fmt.block_values:
        raise ValueError(f"{fmt.name}: row length {k} is not a multiple of {fmt.block_values}")
    return k // fmt.block_values * fmt.block_bytes


def dequantize(w: "mx.array", qtype: int, k: int, dtype: "mx.Dtype") -> "mx.array":
    """Expand a uint8 ``[R, row_bytes]`` GGUF weight into ``[R, k]`` of ``dtype``
    (float32 or bfloat16; bfloat16 is the float32 value rounded once)."""
    import mlx.core as mx
    rows = w.shape[0]
    rb = row_bytes(qtype, k)
    if w.dtype != mx.uint8 or w.ndim != 2 or w.shape[1] != rb:
        raise ValueError(f"expected uint8 [R, {rb}] for {FORMATS[qtype].name}, "
                         f"got {w.dtype} {tuple(w.shape)}")
    chunks = k // 32
    (out,) = _kernel(qtype)(
        inputs=[w],
        template=[("T", dtype), ("KD", k), ("ROWB", rb)],
        output_shapes=[(rows, k)], output_dtypes=[dtype],
        grid=(chunks, rows, 1), threadgroup=(min(chunks, 256), 1, 1),
    )
    return out


# ---------------------------------------------------------------- matmul

# Rows of x up to which the fused kernel runs; past it the weight is expanded to
# bf16 once and multiplied by the stock GEMM. Decode (1) and the speculative verify
# widths (up to the drafter's block) live below it, prefill above.
MV_MAX_M = 16
# Output rows per simdgroup and simdgroups per threadgroup of the fused kernel.
_MV_ROWS, _MV_SG = 2, 4
# The expanded weight of a large-M matmul is bounded to this many bytes per piece:
# the vocabulary projection alone would be 2.5 GB of bf16 in one go.
_EXPAND_BYTES = 256 << 20

# One simdgroup owns _MV_ROWS output rows; each lane walks the row's 32-value chunks
# with stride 32, decodes one chunk per row into registers and applies it to every
# row of x, so each weight byte is read and decoded once for all M. The header's
# no-contraction pragmas exist for the dequantizer's bit-exactness; a dot product
# owes nothing to that, so this body turns contraction back on.
_MV_SRC = r"""
    #pragma clang fp contract(fast)
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row0 = (threadgroup_position_in_grid.y * NSG + sg) * NR;
    const uint nch = KD / 32;
    float acc[MM][NR];
    for (uint m = 0; m < MM; ++m)
        for (uint r = 0; r < NR; ++r) acc[m][r] = 0.f;
    for (uint c = lane; c < nch; c += 32) {
        float4 wv[NR][8];
        for (uint r = 0; r < NR; ++r) {
            const uint row = min(row0 + r, (uint)(ND - 1));
            DEQ(w + (size_t)row * ROWB, c, (thread float*)wv[r]);
        }
        for (uint m = 0; m < MM; ++m) {
            const device vec<T, 4>* xp = (const device vec<T, 4>*)(x + (size_t)m * KD + 32 * c);
            for (uint i = 0; i < 8; ++i) {
                const float4 xv = float4(xp[i]);
                for (uint r = 0; r < NR; ++r) acc[m][r] += dot(xv, wv[r][i]);
            }
        }
    }
    for (uint m = 0; m < MM; ++m)
        for (uint r = 0; r < NR; ++r) {
            const float s = simd_sum(acc[m][r]);
            if (lane == 0 && row0 + r < (uint)ND) out[(size_t)m * ND + row0 + r] = static_cast<T>(s);
        }
"""

# Verify widths (2..MV_MAX_M rows of x). Holding a decoded chunk in registers and
# walking every row of x over it spills past M≈2 (measured: 6.7x the M=1 time at M=8,
# worse with more rows per simdgroup), so these widths stage instead: a threadgroup
# decodes a 32-row x 128-value weight tile into threadgroup memory ONCE (one chunk per
# thread), stages the matching x tile beside it, and each simdgroup applies its 8 rows
# with 8x8 simdgroup MMAs to all of x's rows at once (two column tiles cover 16).
_MM_SG, _MM_TK = 4, 128
_MM_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row0 = (threadgroup_position_in_grid.y * NSG + sg) * 8;
    threadgroup half Wt_all[NSG * 8 * 128];
    threadgroup half* Wt = Wt_all + sg * 8 * 128;
    simdgroup_matrix<float, 8, 8> acc[MT];
    for (uint t = 0; t < MT; ++t) acc[t] = simdgroup_matrix<float, 8, 8>(0);
    const uint drow = lane >> 2, dchunk = lane & 3;
    const device uint8_t* wrow = w + (size_t)min(row0 + drow, (uint)(ND - 1)) * ROWB;
    for (uint k0 = 0; k0 < KD; k0 += 128) {
        float4 v[8];
        DEQ(wrow, k0 / 32 + dchunk, (thread float*)v);
        threadgroup half4* dst = (threadgroup half4*)(Wt + drow * 128 + dchunk * 32);
        for (uint i = 0; i < 8; ++i) dst[i] = half4(v[i]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 128; kk += 8) {
            simdgroup_matrix<half, 8, 8> A, B;
            simdgroup_load(A, Wt + kk, 128);
            for (uint t = 0; t < MT; ++t) {
                simdgroup_load(B, x + (size_t)t * 8 * KD + k0 + kk, KD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[t], A, B, acc[t]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* O = (threadgroup float*)Wt;     // 8 x 16 floats fit in 2 KB
    for (uint t = 0; t < MT; ++t) simdgroup_store(acc[t], O + t * 8, 16);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 8 * 16; i += 32) {
        const uint r = i >> 4, m = i & 15;
        if (m < MM && row0 + r < (uint)ND) out[(size_t)m * ND + row0 + r] = static_cast<T>(O[r * 16 + m]);
    }
"""

_mm_kernels: dict[int, _MetalKernel] = {}


def _mm_kernel(qtype: int) -> _MetalKernel:
    k = _mm_kernels.get(qtype)
    if k is None:
        import mlx.core as mx
        fmt = FORMATS[qtype]
        # SAFETY: the stub types metal_kernel's result as a bare object; it is the
        # keyword-called kernel _MetalKernel describes.
        k = cast(_MetalKernel, mx.fast.metal_kernel(
            name=f"chad_gguf_mm_{fmt.name.lower()}",
            input_names=["x", "w"],
            output_names=["out"],
            source=_MM_SRC.replace("DEQ", fmt.deq_fn),
            header=metal_header(),
        ))
        _mm_kernels[qtype] = k
    return k


_mv_kernels: dict[int, _MetalKernel] = {}


def _mv_kernel(qtype: int) -> _MetalKernel:
    k = _mv_kernels.get(qtype)
    if k is None:
        import mlx.core as mx
        fmt = FORMATS[qtype]
        # SAFETY: the stub types metal_kernel's result as a bare object; it is the
        # keyword-called kernel _MetalKernel describes.
        k = cast(_MetalKernel, mx.fast.metal_kernel(
            name=f"chad_gguf_mv_{fmt.name.lower()}",
            input_names=["x", "w"],
            output_names=["out"],
            source=_MV_SRC.replace("DEQ", fmt.deq_fn),
            header=metal_header(),
        ))
        _mv_kernels[qtype] = k
    return k


def matmul(x: "mx.array", w: "mx.array", qtype: int, k: int) -> "mx.array":
    """``x @ W.T`` for a GGUF weight ``w`` (uint8 ``[N, row_bytes]`` of ``k``-value
    rows); ``x`` is ``[..., k]`` and the result ``[..., N]`` in x's dtype."""
    import mlx.core as mx
    n = w.shape[0]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if 1 < m <= MV_MAX_M and k % _MM_TK == 0:
        mt = (m + 7) // 8
        xh = x2.astype(mx.float16)
        if m < 8 * mt:
            xh = mx.pad(xh, [(0, 8 * mt - m), (0, 0)])
        (out,) = _mm_kernel(qtype)(
            inputs=[xh, w],
            template=[("T", x.dtype), ("MM", m), ("MT", mt), ("NSG", _MM_SG), ("KD", k),
                      ("ND", n), ("ROWB", row_bytes(qtype, k))],
            output_shapes=[(m, n)], output_dtypes=[x.dtype],
            grid=(32 * _MM_SG, (n + 8 * _MM_SG - 1) // (8 * _MM_SG), 1),
            threadgroup=(32 * _MM_SG, 1, 1),
        )
        return out.reshape(*lead, n)
    if m <= MV_MAX_M:
        rows_per_tg = _MV_ROWS * _MV_SG
        (out,) = _mv_kernel(qtype)(
            inputs=[x2, w],
            template=[("T", x.dtype), ("MM", m), ("NR", _MV_ROWS), ("NSG", _MV_SG),
                      ("KD", k), ("ND", n), ("ROWB", row_bytes(qtype, k))],
            output_shapes=[(m, n)], output_dtypes=[x.dtype],
            grid=(32 * _MV_SG, (n + rows_per_tg - 1) // rows_per_tg, 1),
            threadgroup=(32 * _MV_SG, 1, 1),
        )
        return out.reshape(*lead, n)
    step = max(1, _EXPAND_BYTES // (2 * k))
    parts = [x2 @ dequantize(w[i:i + step], qtype, k, x.dtype).T
             for i in range(0, n, step)]
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
    return out.reshape(*lead, n)
