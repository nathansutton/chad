"""Engine glue for the quantized-from-start KV cache (follow-on):
the kv_bits auto-gate, the QuantizedKVCache cache build, per-token byte
accounting, trim, and the turn-boundary rewind — all on the tiny synthetic
qwen3_5 hybrid from test_mlx_fastpath (no downloaded weights)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models import cache as cache_utils  # noqa: E402

from chad.engine import Engine  # noqa: E402
from test_mlx_fastpath import TINY_CFG  # noqa: E402


def build_tiny():
    """test_mlx_fastpath's tiny hybrid, but with head_dim 64 — the smallest
    the QuantizedKVCache supports (its group_size-64 quantization needs the
    last axis divisible by 64)."""
    import copy

    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    cfg = copy.deepcopy(TINY_CFG)
    cfg["text_config"]["head_dim"] = 64
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(cfg))
    nn.quantize(model, group_size=64, bits=4)
    model.eval()
    return model


def _bare_engine(model=None, kv_bits=None):
    eng = object.__new__(Engine)  # bypass __init__ (no weights to load)
    eng.model = model
    eng.draft = None
    eng.kv_bits = kv_bits
    eng._cached_ids = []
    eng._rewind_snap = None
    eng._cache_stack = []
    return eng


# ---- _resolve_kv_bits (pure shape logic, no model) --------------------------

def _resolve(head_dim, n_heads, n_kv_heads, kv_bits, qsdpa_ok=True):
    eng = _bare_engine(kv_bits=kv_bits)
    eng._n_attn_heads = n_heads
    eng._n_kv_heads = n_kv_heads
    eng._head_dim = head_dim
    eng._resolve_kv_bits(qsdpa_ok)
    return eng.kv_bits


def test_kv_bits_auto_defaults_on_for_covered_shapes():
    assert _resolve(256, 16, 2, None) == 8      # 35B: gqa 8
    assert _resolve(256, 16, 4, None) == 8      # 9B: gqa 4
    assert _resolve(128, 16, 2, None) is None   # wrong head_dim
    assert _resolve(256, 16, 8, None) is None   # gqa 2
    assert _resolve(256, 16, 0, None) is None   # shape unknown
    assert _resolve(256, 16, 2, None, qsdpa_ok=False) is None  # kernel absent


def test_kv_bits_explicit_wins():
    assert _resolve(256, 16, 2, 0) is None      # forced off
    assert _resolve(128, 16, 2, 8) == 8         # forced on, uncovered: honored


def test_kv_bits_auto_off_with_draft():
    eng = _bare_engine(kv_bits=None)
    eng.draft = object()
    eng._n_attn_heads, eng._n_kv_heads, eng._head_dim = 16, 2, 256
    eng._resolve_kv_bits(True)
    assert eng.kv_bits is None


# ---- cache build / accounting / trim / rewind on the tiny hybrid -----------

@pytest.fixture(scope="module")
def tiny():
    return build_tiny()


def _loaded_engine(tiny, kv_bits):
    eng = _bare_engine(model=tiny, kv_bits=kv_bits)
    eng._reset_cache()
    return eng


def test_reset_cache_quantizes_attention_layers_only(tiny):
    eng = _loaded_engine(tiny, kv_bits=8)
    kinds = [type(c) for c in eng._cache]
    assert cache_utils.QuantizedKVCache in kinds
    assert cache_utils.ArraysCache in kinds       # GDN layers untouched
    assert cache_utils.KVCache not in kinds
    # the hybrid rollback machinery must still classify as recoverable
    assert eng._pld_hybrid is True
    assert eng._trimmable is False

    fp16 = _loaded_engine(tiny, kv_bits=None)
    assert cache_utils.QuantizedKVCache not in [type(c) for c in fp16._cache]


def test_quantized_prefill_and_kv_bytes_accounting(tiny):
    from chad import mlx_qsdpa
    mlx_qsdpa.install()

    q8 = _loaded_engine(tiny, kv_bits=8)
    fp16 = _loaded_engine(tiny, kv_bits=None)
    b_q8 = q8._measure_kv_bytes_per_token()
    b_fp = fp16._measure_kv_bytes_per_token()
    assert b_q8 > 0 and b_fp > 0
    # 8-bit + scales/biases is ~0.53x of an fp16 cache and ~0.28x of the fp32
    # cache this tiny model runs — either way, well under 0.7x
    assert 0.2 * b_fp < b_q8 < 0.7 * b_fp


def test_trim_and_rewind_on_quantized_cache(tiny):
    from chad import mlx_qsdpa
    mlx_qsdpa.install()

    eng = _loaded_engine(tiny, kv_bits=8)
    base = [7, 12, 33, 8, 91, 40, 55, 2, 19, 77, 30, 4]
    fed = eng._prefill(list(base), chunk=6)
    assert fed == len(base)
    eng._cached_ids = list(base)
    eng._take_rewind_snapshot(len(base))
    assert eng._rewind_snap is not None and eng._rewind_snap["pos"] == len(base)

    extra = [64, 25, 3, 88]
    eng._prefill(list(extra), chunk=4)
    eng._cached_ids = base + extra
    qkv = [c for c in eng._cache
           if isinstance(c, cache_utils.QuantizedKVCache)]
    assert all(c.offset == len(base) + len(extra) for c in qkv)

    # rewind to the snapshot point exactly
    assert eng._rewind_to(base + extra, len(base)) == len(base)
    assert eng._cached_ids == base
    assert all(c.offset == len(base) for c in qkv)

    # and the cache still works: prefill something new on top
    assert eng._prefill([5, 6], chunk=2) == 2
    assert all(c.offset == len(base) + 2 for c in qkv)


def test_ckpt_path_keyed_by_kv_mode(tiny):
    a = _bare_engine(model=tiny, kv_bits=8)
    b = _bare_engine(model=tiny, kv_bits=None)
    a.model_id = b.model_id = "m"
    a.cache_dir = b.cache_dir = "/tmp/x"
    ids = [1, 2, 3]
    assert a._ckpt_path(ids) != b._ckpt_path(ids)


def test_warm_prefix_roundtrip_and_mode_isolation(tiny, tmp_path):
    """A quantized session's warm-prefix checkpoint must round-trip (hit on a
    fresh same-mode engine, restoring QuantizedKVCache layers and the hybrid
    rollback classification) and must NOT be loaded by an fp16 session."""
    def make(kv_bits):
        eng = _bare_engine(model=tiny, kv_bits=kv_bits)
        eng.model_id = "tiny"
        eng.cache_dir = str(tmp_path)
        eng.kv_cache_max_bytes = 0
        eng._warm_prefix_ids = None
        eng._reset_cache()
        return eng

    ids = [5, 9, 3, 7, 12, 44, 2, 8]
    assert make(8).warm_prefix(ids) == ("miss", len(ids))

    hit = make(8)
    assert hit.warm_prefix(ids) == ("hit", len(ids))
    kinds = {type(c) for c in hit._cache}
    assert cache_utils.QuantizedKVCache in kinds
    assert hit._pld_hybrid is True

    fp16 = make(None)
    status, _ = fp16.warm_prefix(ids)
    assert status == "miss"  # different key: never cross-loads the kv8 file
    assert cache_utils.QuantizedKVCache not in {type(c) for c in fp16._cache}
