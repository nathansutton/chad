"""Model-free tests for `chad.suffix_reuse`: the survivor planner, key re-rotation on
both RoPE classes the hybrid model can build, and the row splice on both attention
cache classes."""

import time

import pytest

from chad import suffix_reuse
from chad.suffix_reuse import Span


def _ids(lo, hi):
    return list(range(lo, hi))


def _thirty_trims():
    """A 3000-token prefix, then thirty 1200-token blocks whose first 300 tokens are
    each replaced by 20 fresh ones — the shape of compaction's per-result trims."""
    prefix = _ids(0, 3000)
    cached, target = list(prefix), list(prefix)
    fresh = 10_000_000
    for i in range(30):
        base = 3000 + i * 1200
        cached += _ids(base, base + 1200)
        target += _ids(fresh + i * 20, fresh + i * 20 + 20)
        target += _ids(base + 300, base + 1200)
    return cached, target


def test_drop_oldest_is_one_span():
    prefix, dropped, tail = _ids(0, 3000), _ids(3000, 12000), _ids(12000, 60000)
    p = suffix_reuse.plan(prefix + dropped + tail, prefix + tail, common=3000,
                          max_spans=6, min_span=64)
    assert p is not None
    assert p.spans == (Span(src_lo=12000, tgt_lo=3000, size=48000),)
    assert p.end == 51000
    assert p.common == 3000


def test_many_trims_keep_the_k_largest_in_target_order():
    cached, target = _thirty_trims()
    p = suffix_reuse.plan(cached, target, common=3000, max_spans=30, min_span=64)
    assert p is not None and len(p.spans) == 30
    sizes = sorted((s.size for s in p.spans), reverse=True)

    p6 = suffix_reuse.plan(cached, target, common=3000, max_spans=6, min_span=64)
    assert p6 is not None and len(p6.spans) == 6
    assert sorted((s.size for s in p6.spans), reverse=True) == sizes[:6]
    assert [s.tgt_lo for s in p6.spans] == sorted(s.tgt_lo for s in p6.spans)
    assert p6.end == p6.spans[-1].tgt_lo + p6.spans[-1].size


def test_many_trims_unequal_sizes_pick_the_largest():
    prefix = _ids(0, 100)
    cached, target = list(prefix), list(prefix)
    sizes = [100, 400, 150, 500, 90, 300, 200, 250]
    base, fresh = 1000, 10_000_000
    for i, n in enumerate(sizes):
        cached += _ids(base, base + n + 50)
        target += [fresh + i] + _ids(base + 50, base + 50 + n)
        base += n + 50 + 1000
    p = suffix_reuse.plan(cached, target, common=100, max_spans=3, min_span=64)
    assert p is not None
    assert sorted(s.size for s in p.spans) == [300, 400, 500]
    assert [s.size for s in p.spans] == [400, 500, 300]          # target order
    assert p.end == p.spans[-1].tgt_lo + 300                     # not the largest's end


def test_nothing_to_move_is_none():
    prefix = _ids(0, 3000)
    assert suffix_reuse.plan(prefix + _ids(5000, 9000), prefix + _ids(20000, 21000),
                             common=3000, max_spans=6, min_span=64) is None


def test_spans_below_the_floor_are_dropped():
    prefix = _ids(0, 100)
    cached = prefix + _ids(1000, 1100) + _ids(2000, 2040)
    target = prefix + _ids(2000, 2040) + [9]
    assert suffix_reuse.plan(cached, target, common=100, max_spans=6,
                             min_span=64) is None
    p = suffix_reuse.plan(cached, target, common=100, max_spans=6, min_span=32)
    assert p is not None and p.spans == (Span(src_lo=200, tgt_lo=100, size=40),)


def test_spans_never_overlap_and_are_monotone():
    cached, target = _thirty_trims()
    p = suffix_reuse.plan(cached, target, common=3000, max_spans=64, min_span=64)
    assert p is not None
    for a, b in zip(p.spans, p.spans[1:]):
        assert a.src_lo + a.size <= b.src_lo
        assert a.tgt_lo + a.size <= b.tgt_lo
    for s in p.spans:
        assert s.src_lo >= 3000 and s.tgt_lo >= 3000
        assert cached[s.src_lo:s.src_lo + s.size] == target[s.tgt_lo:s.tgt_lo + s.size]


def test_planning_sixty_thousand_tokens_is_fast():
    cached, target = _thirty_trims()
    tail = _ids(50_000_000, 50_000_000 + 60_000 - len(cached))
    cached, target = cached + tail, target + tail
    t0 = time.perf_counter()
    p = suffix_reuse.plan(cached, target, common=3000, max_spans=6, min_span=64)
    assert time.perf_counter() - t0 < 0.5
    assert p is not None


# -- re-rotation ------------------------------------------------------------------

def _rope(scaling=None):
    from mlx_lm.models.rope_utils import initialize_rope
    return initialize_rope(64, base=1e7, traditional=False, scaling_config=scaling,
                           max_position_embeddings=262144)


def _keys(shape=(1, 4, 8, 256), dtype=None):
    import mlx.core as mx
    mx.random.seed(0)
    k = mx.random.normal(shape)
    return k.astype(dtype) if dtype is not None else k


def test_rerotate_plain_rope_matches_a_fresh_rotation():
    import mlx.core as mx
    rope, k = _rope(), _keys()
    moved = suffix_reuse.rerotate_keys(rope(k, offset=5000), rope, -2000)
    assert mx.allclose(moved, rope(k, offset=3000), rtol=1e-3, atol=1e-3).item()
    assert mx.array_equal(moved[..., 64:], k[..., 64:]).item()


def test_rerotate_yarn_is_not_double_scaled():
    import mlx.core as mx
    rope = _rope({"rope_type": "yarn", "factor": 4.0,
                  "original_max_position_embeddings": 262144})
    assert rope.mscale != 1.0
    k = _keys()
    truth = rope(k, offset=90000)
    moved = suffix_reuse.rerotate_keys(rope(k, offset=120000), rope, -30000)
    # fp32 angles at position 1e5 carry ~1e-3 of rounding either way, so the bound is
    # on the largest error relative to the key's scale rather than per element.
    rel = (mx.abs(moved - truth).max() / mx.abs(truth).max()).item()
    assert rel < 5e-3, rel
    # The trap the helper avoids: calling the module again scales the rotary dims twice.
    twice = rope(rope(k, offset=120000).reshape(-1, 1, 256), offset=-30000)
    twice = twice.reshape(truth.shape)
    rel = (mx.abs(twice - truth).max() / mx.abs(truth).max()).item()
    assert rel > 0.05, rel


def test_rerotate_zero_delta_is_identity():
    import mlx.core as mx
    rope, k = _rope(), _keys()
    rotated = rope(k, offset=777)
    assert mx.array_equal(suffix_reuse.rerotate_keys(rotated, rope, 0), rotated).item()


# -- the row splice ---------------------------------------------------------------

T = 700


def _filled(quantized, keys=None):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, QuantizedKVCache
    cache = QuantizedKVCache(group_size=64, bits=8) if quantized else KVCache()
    k = _keys((1, 4, T, 256), mx.float16) if keys is None else keys
    mx.random.seed(1)
    v = mx.random.normal((1, 4, T, 256)).astype(mx.float16)
    cache.update_and_fetch(k, v)
    mx.eval(cache.state)
    return cache


def _arrays(cache):
    """(keys, values) as tuples of arrays, on either cache class."""
    rows = suffix_reuse.take_rows(cache, 0, cache.offset)
    return rows.keys, rows.values


@pytest.mark.parametrize("quantized", [False, True])
def test_take_then_append_reproduces_the_rows(quantized):
    import mlx.core as mx
    cache = _filled(quantized)
    keys, values = _arrays(cache)
    before_k = [a[..., 300:500, :] for a in keys]
    before_v = [a[..., 300:500, :] for a in values]
    mx.eval(before_k, before_v)
    rows = suffix_reuse.take_rows(cache, 300, 500)
    mx.eval(rows.keys, rows.values)
    cache.trim(400)
    suffix_reuse.append_rows(cache, rows)
    assert cache.offset == 500
    keys, values = _arrays(cache)
    for a, b in zip(keys, before_k):
        assert mx.array_equal(a[..., 300:500, :], b).item()
    for a, b in zip(values, before_v):
        assert mx.array_equal(a[..., 300:500, :], b).item()
    # mlx-lm's own write path accepts the spliced buffer (offset not step-aligned).
    k1 = _keys((1, 4, 1, 256), mx.float16)
    cache.update_and_fetch(k1, k1)
    mx.eval(cache.state)
    assert cache.offset == 501
    assert _arrays(cache)[0][0].shape[-2] == 501


def test_quantized_rerotation_is_close_to_the_truth():
    import mlx.core as mx
    rope = _rope()
    raw = _keys((1, 4, T, 256), mx.float16)
    cache = _filled(True, keys=rope(raw))
    rows = suffix_reuse.rerotate_rows(suffix_reuse.take_rows(cache, 300, 500), rope, -100)
    got = mx.dequantize(*rows.keys, group_size=64, bits=8)
    truth = mx.dequantize(*mx.quantize(rope(raw[..., 300:500, :], offset=200),
                                       group_size=64, bits=8), group_size=64, bits=8)
    assert mx.allclose(got, truth, rtol=3e-2, atol=3e-2).item()
    # The pass-through groups keep their exact bits: only the rotary group moved.
    orig = suffix_reuse.take_rows(cache, 300, 500)
    for a, b, cols in zip(rows.keys, orig.keys, (16, 1, 1)):
        assert mx.array_equal(a[..., cols:], b[..., cols:]).item()


# -- the engine tier on a tiny random-weight hybrid ----------------------------------

@pytest.fixture(scope="module")
def tiny_hybrid():
    """A tiny qwen3_5 built in-process (no weights to load): GatedDeltaNet and
    attention layers with real caches, so the splice meets a real forward."""
    import copy

    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from test_mlx_fastpath import TINY_CFG

    cfg = copy.deepcopy(TINY_CFG)
    cfg["text_config"]["head_dim"] = 64
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(cfg))
    model.eval()
    return model


def _tiny_engine(model, kv_bits):
    from chad.engine import Engine
    eng = object.__new__(Engine)
    eng.model = model
    eng.kv_bits = kv_bits
    eng.cache_dir = None
    eng._warm_prefix_ids = eng._warm_head_ids = None
    eng._is_moe = False
    eng._n_attn_heads = 4
    eng._reset_cache()
    return eng


@pytest.mark.parametrize("kv_bits", [None, 8])
def test_relocation_on_a_real_hybrid_cache(tiny_hybrid, kv_bits):
    """Drop-oldest then replace-in-the-middle on real caches: the ledger, every
    attention layer's offset and buffer, and the untouched recurrent state all agree,
    and the cache keeps accepting forwards afterwards."""
    import mlx.core as mx
    eng = _tiny_engine(tiny_hybrid, kv_bits)
    assert eng._pld_hybrid and not eng._trimmable
    layers = eng._attention_layers()
    assert layers and len(layers) < len(eng._cache)

    def attn_rows():
        return [(c.offset, suffix_reuse.take_rows(c, 0, c.offset).keys[0].shape[-2])
                for c, _ in layers]

    P, D, T = _ids(10, 30), _ids(1000, 1100), _ids(2000, 2150)
    eng._cached_ids = []
    assert eng._prefill(P + D + T) == len(P + D + T)
    eng._cached_ids = P + D + T
    recurrent = [c for layer, c in zip(tiny_hybrid.layers, eng._cache) if layer.is_linear]
    before = [list(c.cache) for c in recurrent]
    old_rows = [suffix_reuse.take_rows(c, 120, 270) for c, _ in layers]
    mx.eval([(r.keys, r.values) for r in old_rows])

    target = P + T + [7]
    common = eng._sync_to(target)
    assert common == len(P + T) and eng._scr_last == (150, 0)
    assert eng._cached_ids == target[:common]
    assert attn_rows() == [(common, common)] * len(layers)
    for c, arrs in zip(recurrent, before):
        assert all(a is b for a, b in zip(c.cache, arrs)), "recurrent state was touched"
    # The moved rows are the old ones re-rotated by -100, values untouched.
    for (c, rope), old in zip(layers, old_rows):
        moved = suffix_reuse.take_rows(c, 20, 170)
        want = suffix_reuse.rerotate_rows(old, rope, -100)
        for a, b in zip(moved.keys + moved.values, want.keys + want.values):
            assert mx.array_equal(a, b).item()

    assert eng._prefill([7]) == 1
    eng._cached_ids = target
    assert attn_rows()[0][0] == len(target)

    # Replace in the middle: the inserted text is read, the tail (and the token
    # decoded after it) moves right by 100.
    edit = _ids(5000, 5100)
    target2 = P + edit + T + [7, 8]
    common = eng._sync_to(target2)
    assert common == len(P + edit + T) + 1 and eng._scr_last == (151, 100)
    assert eng._cached_ids == target2[:common]
    assert [o for o, _ in attn_rows()] == [common] * len(layers)
    assert eng._prefill([8]) == 1
    mx.eval(eng.model(mx.array([[9]], dtype=mx.uint32), cache=eng._cache))
    assert [o for o, _ in attn_rows()] == [common + 2] * len(layers)
