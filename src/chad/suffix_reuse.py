"""Keep the attention rows of text that survived a mid-transcript edit.

The hybrid cache cannot be rewound, so an edit in the middle of the transcript
(compaction stripping old thinking, trimming old tool results, dropping the oldest
turns) used to cost a re-read of everything after the first changed token. Most of
that text is still there after the edit, only at a new position. This module holds
the model-free half of moving it instead: finding the survivors, and the row
surgery on one attention layer's cache.

A *survivor* is a run of tokens present, in order, in both the cached ids and the
new target ids, past their common prefix. Its attention rows are sliced out of the
old cache, their keys re-rotated from the old positions to the new ones, and
appended in place of a forward pass. The recurrent layers are left alone: they carry
on from the state they have, which still summarises the deleted text — the
approximation this mechanism trades for skipping the re-read.

Keys are cached after RoPE, and rotations compose additively in position, so moving
a key from position p to q is one more rotation by q - p over the rotary dims; the
pass-through dims are untouched. The rotation is applied with `mx.fast.rope`
directly from the module's parameters and never by calling the module: a YaRN
module scales the rotary dims by its mscale before rotating, and calling it on a key
it already produced would scale that key twice.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:  # mlx is imported lazily inside the functions so the module loads on Linux
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import KVCache, QuantizedKVCache, _BaseCache
    from mlx_lm.models.rope_utils import YarnRoPE
    from typing_extensions import TypeIs

    AttnCache = Union[KVCache, QuantizedKVCache]
    Rope = Union[nn.RoPE, YarnRoPE]


def is_attention_cache(cache: _BaseCache) -> TypeIs[AttnCache]:
    """True for the two attention cache classes whose rows this module can move."""
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    return isinstance(cache, (KVCache, QuantizedKVCache))


def is_quantized(cache: AttnCache) -> TypeIs[QuantizedKVCache]:
    """True for the quantized attention cache, whose buffers are 3-tuples."""
    from mlx_lm.models.cache import QuantizedKVCache

    return isinstance(cache, QuantizedKVCache)


def is_yarn(rope: Rope) -> TypeIs[YarnRoPE]:
    """True for the YaRN module, which rotates by its own frequencies and scales the
    rotary dims on the way in."""
    from mlx_lm.models.rope_utils import YarnRoPE

    return isinstance(rope, YarnRoPE)


@dataclass(frozen=True)
class Span:
    """One survivor: `size` cached rows starting at `src_lo` in the current cache,
    which belong at `tgt_lo` in the target ids."""
    src_lo: int
    tgt_lo: int
    size: int


@dataclass(frozen=True)
class Plan:
    """How to land the cache on the target ids without re-reading the survivors.

    `common` tokens are shared as a prefix and stay where they are. The target region
    `[common, end)` alternates between gaps (target tokens no span covers, which are
    prefilled) and `spans` (rows moved from the old cache), in target order. Target
    tokens past `end` are the ordinary suffix the caller prefills."""
    common: int
    spans: tuple[Span, ...]
    end: int


def plan(cached_ids: list[int], target_ids: list[int], *, common: int,
         max_spans: int, min_span: int) -> Optional[Plan]:
    """Find the survivors worth moving from `cached_ids` into `target_ids`.

    The common prefix is excluded: the prefix path already serves it in place. Only
    the `max_spans` largest survivors are kept, because every moved span carries rows
    computed under text that is no longer there and the largest few carry nearly all
    of the reuse. A survivor shorter than `min_span` is read rather than moved: below
    that, a forward over it is cheap and exact. None when nothing qualifies, so the
    caller falls back to the rebuild."""
    blocks = difflib.SequenceMatcher(
        a=cached_ids, b=target_ids, autojunk=False).get_matching_blocks()
    kept = [Span(b.a, b.b, b.size) for b in blocks
            if b.a >= common and b.b >= common and b.size >= min_span]
    if not kept or max_spans < 1:
        return None
    kept = sorted(kept, key=lambda s: s.size, reverse=True)[:max_spans]
    spans = tuple(sorted(kept, key=lambda s: s.tgt_lo))
    # Matching blocks never overlap and are monotone in both sequences, so target
    # order is also source order; the splice appends in that order and relies on it.
    for prev, nxt in zip(spans, spans[1:]):
        assert prev.src_lo + prev.size <= nxt.src_lo, (prev, nxt)
        assert prev.tgt_lo + prev.size <= nxt.tgt_lo, (prev, nxt)
    last = spans[-1]
    return Plan(common=common, spans=spans, end=last.tgt_lo + last.size)


def rerotate_keys(keys: mx.array, rope: Rope, delta: int) -> mx.array:
    """Move post-RoPE `keys` by `delta` positions (negative moves them earlier).
    Only the module's first `dims` features rotate; the rest pass through."""
    import mlx.core as mx

    if delta == 0:
        return keys
    # `mx.fast.rope` rotates row t of a sequence by offset + t. Every row here moves
    # by the same delta, so each is presented as a sequence of length one.
    flat = keys.reshape(-1, 1, keys.shape[-1])
    if is_yarn(rope):
        # The YaRN frequencies, without the mscale the module applies on the way in:
        # the cached key already carries it once.
        out = mx.fast.rope(flat, rope.dims, traditional=rope.traditional, base=None,
                           scale=1.0, offset=delta, freqs=rope._freqs)
    else:
        out = mx.fast.rope(flat, rope.dims, traditional=rope.traditional,
                           base=rope.base, scale=rope.scale, offset=delta)
    return out.reshape(keys.shape)


@dataclass(frozen=True)
class Rows:
    """Rows sliced out of one attention layer's cache, along the row axis (-2).

    `keys` and `values` are 1-tuples for a `KVCache` and the `(q, scales, biases)`
    3-tuples for a `QuantizedKVCache`, whose `quant` then holds `(group_size, bits)`.
    Quantization groups run along the last axis, so a row slice keeps every group."""
    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    size: int
    quant: Optional[tuple[int, int]] = None


def _parts(cache: AttnCache) -> tuple[tuple[mx.array, ...], tuple[mx.array, ...],
                                      Optional[tuple[int, int]]]:
    assert cache.keys is not None and cache.values is not None, "empty attention cache"
    if is_quantized(cache):
        return (tuple(cache.keys), tuple(cache.values),
                (cache.group_size, cache.bits))
    return (cache.keys,), (cache.values,), None


def take_rows(cache: AttnCache, lo: int, hi: int) -> Rows:
    """Slice rows `[lo, hi)` out of an attention layer's cache."""
    keys, values, quant = _parts(cache)
    assert 0 <= lo <= hi <= cache.offset, (lo, hi, cache.offset)
    return Rows(keys=tuple(x[..., lo:hi, :] for x in keys),
                values=tuple(x[..., lo:hi, :] for x in values),
                size=hi - lo, quant=quant)


def drop_spare_rows(cache: AttnCache) -> None:
    """Cut the cache's buffers back to its live rows. A trim only lowers `offset`,
    so the next write would land in place, in a buffer moved rows may still share
    (mlx then copies the whole buffer to keep them intact). Past the end of a buffer
    that holds only live rows, mlx-lm grows a fresh one instead."""

    keys, values, _ = _parts(cache)
    off = cache.offset
    assert off >= 1, "no live rows to keep"
    _set_buffers(cache, tuple(x[..., :off, :] for x in keys),
                 tuple(x[..., :off, :] for x in values))


def append_rows(cache: AttnCache, rows: Rows) -> None:
    """Append `rows` after the cache's live rows. The buffer ends exactly at the new
    offset, a length mlx-lm's own growth path accepts: its next write past capacity
    concatenates a fresh step onto the live rows."""

    keys, values, quant = _parts(cache)
    assert quant == rows.quant, (quant, rows.quant)
    off = cache.offset
    new_keys = tuple(_cat(x[..., :off, :], r) for x, r in zip(keys, rows.keys))
    new_values = tuple(_cat(x[..., :off, :], r)
                       for x, r in zip(values, rows.values))
    _set_buffers(cache, new_keys, new_values)
    cache.offset = off + rows.size


def _set_buffers(cache: AttnCache, keys: tuple[mx.array, ...],
                 values: tuple[mx.array, ...]) -> None:
    """Install buffers that end exactly at the live rows, through the cache's own
    `state` setter (a `KVCache` also takes its offset from the buffer length there,
    which is the live count by construction)."""
    if is_quantized(cache):
        cache.state = (keys, values)
    else:
        cache.state = (keys[0], values[0])


def _cat(head: mx.array, tail: mx.array) -> mx.array:
    import mlx.core as mx

    return mx.concatenate([head, tail], axis=-2)


def rerotate_rows(rows: Rows, rope: Rope, delta: int) -> Rows:
    """Rows whose keys have moved `delta` positions; values do not depend on position
    and pass through. A quantized key is dequantized, rotated and requantized — only
    over the groups that hold rotary features when the rotary width is a whole number
    of groups, so the pass-through groups keep their exact bits."""
    import mlx.core as mx

    if delta == 0:
        return rows
    if rows.quant is None:
        return Rows(keys=(rerotate_keys(rows.keys[0], rope, delta),),
                    values=rows.values, size=rows.size)
    group_size, bits = rows.quant
    q, scales, biases = rows.keys
    dims = rope.dims
    if dims % group_size == 0 and dims * bits % 32 == 0:
        qc, sc = dims * bits // 32, dims // group_size
    else:
        qc, sc = q.shape[-1], scales.shape[-1]
    head = mx.dequantize(q[..., :qc], scales[..., :sc], biases[..., :sc],
                         group_size=group_size, bits=bits)
    rq, rs, rb = mx.quantize(rerotate_keys(head, rope, delta),
                             group_size=group_size, bits=bits)
    keys = (mx.concatenate([rq, q[..., qc:]], axis=-1),
            mx.concatenate([rs, scales[..., sc:]], axis=-1),
            mx.concatenate([rb, biases[..., sc:]], axis=-1))
    return Rows(keys=keys, values=rows.values, size=rows.size, quant=rows.quant)
