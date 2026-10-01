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
appended in place of a forward pass.

The recurrent layers cannot be edited that way: their state is one summary of
everything read so far. What can be chosen is *which* state the cache continues
from. The engine holds a few (the live one, a snapshot at the last turn boundary),
each the exact state after reading the old ids up to some position — an *anchor*.
A plan ends at the latest anchor that falls inside a survivor, so the recurrent
layers continue from a state that has just read the very text the target has at
that point, and everything after it is read fresh through every layer. It still
summarises the deleted text too — the approximation this mechanism trades for
skipping the re-read. A plan that ends anywhere else leaves the recurrent layers
past text the target never reaches (a previous answer, say), and the model acts on
what they read: measured, it answers an already-answered question with an
immediate end of turn.

Keys are cached after RoPE, and rotations compose additively in position, so moving
a key from position p to q is one more rotation by q - p over the rotary dims; the
pass-through dims are untouched. The rotation is applied with `mx.fast.rope`
directly from the module's parameters and never by calling the module: a YaRN
module scales the rotary dims by its mscale before rotating, and calling it on a key
it already produced would scale that key twice.
"""
from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Union

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
    prefilled) and `spans` (rows moved from the old cache), in target order. The last
    span ends at `anchor` in the old ids: the recurrent state to continue from is the
    one that had read exactly `anchor` of them. Target tokens past `end` are the
    ordinary suffix the caller prefills."""
    common: int
    spans: tuple[Span, ...]
    end: int
    anchor: int


def plan(cached_ids: list[int], target_ids: list[int], *, common: int,
         anchors: Sequence[int], max_spans: int, min_span: int) -> Optional[Plan]:
    """Find the survivors worth moving from `cached_ids` into `target_ids`, ending at
    one of `anchors` (positions in `cached_ids` the caller holds a recurrent state
    for; see the module docstring).

    The latest anchor that falls inside a survivor wins, and that survivor is cut
    there: it is always moved, whatever its size, because it is what lines the
    recurrent state up with the target. Survivors past it are read, not moved. Of the
    ones before it only the `max_spans - 1` largest are moved, because every moved
    span carries rows computed under text that is no longer there and the largest few
    carry nearly all of the reuse; one shorter than `min_span` is read, since a
    forward over it is cheap. The common prefix is excluded: the prefix path already
    serves it in place. None when no anchor falls inside a survivor, or when matching
    would cost more than the re-read it saves, so the caller falls back to the
    rebuild."""
    if max_spans < 1:
        return None
    blocks = _survivors(cached_ids[common:], target_ids[common:], common)
    if blocks is None:
        return None
    for anchor in sorted(set(anchors), reverse=True):
        hit = next((b for b in blocks if b.src_lo < anchor <= b.src_lo + b.size), None)
        if hit is None:
            continue
        last = Span(hit.src_lo, hit.tgt_lo, anchor - hit.src_lo)
        # Matching blocks never overlap and are monotone in both sequences, so the
        # blocks before `hit` in source order are the ones before it in target order.
        before = [b for b in blocks
                  if b.src_lo + b.size <= hit.src_lo and b.size >= min_span]
        kept = sorted(before, key=lambda s: s.size, reverse=True)[:max_spans - 1]
        spans = tuple(sorted(kept, key=lambda s: s.tgt_lo)) + (last,)
        # The splice appends in target order and relies on it being source order too.
        for prev, nxt in zip(spans, spans[1:]):
            assert prev.src_lo + prev.size <= nxt.src_lo, (prev, nxt)
            assert prev.tgt_lo + prev.size <= nxt.tgt_lo, (prev, nxt)
        return Plan(common=common, spans=spans, end=last.tgt_lo + last.size,
                    anchor=anchor)
    return None


# Survivors are matched as runs of this many consecutive ids, not single ids. A single
# id repeats throughout a transcript (every newline, every indent), and the matcher's
# cost grows with the number of equal pairs across the two sides — measured at about
# 0.4 s per million pairs. A run this long is nearly unique in ordinary text, so the
# same compaction plans in a fraction of the time. A survivor shorter than a run is
# invisible; `min_span` is far above it, and the survivor that holds the anchor is the
# tail of a tool result in practice.
RUN = 8
# Equal pairs beyond which no plan is made and the caller rebuilds: one line repeated
# hundreds of times defeats the runs too, and the matcher would then take longer than
# the re-read it saves.
WORK_BUDGET = 40_000_000


def _survivors(old: list[int], new: list[int], base: int) -> Optional[list[Span]]:
    """The maximal runs of ids shared by `old` and `new`, in order, as spans whose
    positions are offset by `base` (the common prefix the caller cut off). None when
    matching would exceed `WORK_BUDGET`."""
    runs_old = [tuple(old[i:i + RUN]) for i in range(len(old) - RUN + 1)]
    runs_new = [tuple(new[i:i + RUN]) for i in range(len(new) - RUN + 1)]
    if not runs_old or not runs_new:
        return None
    count_new = Counter(runs_new)
    if sum(count_new[r] for r in runs_old) > WORK_BUDGET:
        return None
    out: list[Span] = []
    for m in difflib.SequenceMatcher(a=runs_old, b=runs_new,
                                     autojunk=False).get_matching_blocks():
        if m.size == 0:
            continue
        # `size` equal runs are `size + RUN - 1` equal ids, so neighbouring blocks can
        # share up to `RUN - 1` ids; the later block gives them up, on both sides.
        src_lo, tgt_lo, size = m.a + base, m.b + base, m.size + RUN - 1
        if out:
            prev = out[-1]
            cut = max(prev.src_lo + prev.size - src_lo, prev.tgt_lo + prev.size - tgt_lo, 0)
            src_lo, tgt_lo, size = src_lo + cut, tgt_lo + cut, size - cut
        if size > 0:
            out.append(Span(src_lo, tgt_lo, size))
    return out


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
    """Append `rows` after the cache's live rows. Into the buffer's spare capacity when
    it has enough (a caller that reserved it keeps every append and the reads between
    them writing in place), the way mlx-lm's own write lands. Otherwise the buffer is
    rebuilt to end exactly at the new offset, a length mlx-lm's growth path accepts:
    its next write past capacity concatenates a fresh step onto the live rows."""

    keys, values, quant = _parts(cache)
    assert quant == rows.quant, (quant, rows.quant)
    off = cache.offset
    end = off + rows.size
    if keys[0].shape[-2] >= end:
        for x, r in zip(keys + values, rows.keys + rows.values):
            x[..., off:end, :] = r
        cache.offset = end
        return
    new_keys = tuple(_cat(x[..., :off, :], r) for x, r in zip(keys, rows.keys))
    new_values = tuple(_cat(x[..., :off, :], r)
                       for x, r in zip(values, rows.values))
    _set_buffers(cache, new_keys, new_values)
    cache.offset = end


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
