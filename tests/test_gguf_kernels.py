"""GGUF block dequantizers: the Metal ``deq32_*`` functions against gguf.quants' numpy
decoders, bit for bit, on real blocks of every format the shipped GGUF uses.

``fixtures/gguf_blocks.npz`` maps each ggml type name to a uint8 ``[3, row_bytes]``
array: rows 0, R/2 and R-1 of the first tensor of that type in
Qwen3.8-27B-UD-IQ3_XXS.gguf, each cut to 4 blocks from the start of the row followed
by 4 from its middle (16 + 16 for Q8_0's 32-value blocks). Re-extract with::

    r = gguf.GGUFReader(path)
    t = next(t for t in r.tensors if int(t.tensor_type) == q and t.data.ndim == 2)
    bb = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(q)][1]
    nb, mid = (32 if q == 8 else 8), t.data.shape[1] // bb // 2
    rows = [0, t.data.shape[0] // 2, t.data.shape[0] - 1]
    blk = np.concatenate([t.data[rows, :nb // 2 * bb],
                          t.data[rows, mid * bb:(mid + nb // 2) * bb]], axis=1)
"""

import os

import numpy as np
import pytest
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, quants

from chad import mlx_gguf

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                        "gguf_blocks.npz")


def _blocks() -> dict[str, np.ndarray]:
    with np.load(_FIXTURE) as z:
        return {name: z[name] for name in z.files}


def test_registry_matches_gguf():
    for qtype, fmt in mlx_gguf.FORMATS.items():
        gt = GGMLQuantizationType(qtype)
        assert gt.name == fmt.name
        assert GGML_QUANT_SIZES[gt] == (fmt.block_values, fmt.block_bytes)


def test_fixture_covers_every_format():
    assert {GGMLQuantizationType[n].value for n in _blocks()} == set(mlx_gguf.FORMATS)


@pytest.mark.parametrize("name", sorted(f.name for f in mlx_gguf.FORMATS.values()))
def test_metal_matches_numpy_bit_exact(name):
    mx = pytest.importorskip("mlx.core")
    raw = _blocks()[name]
    qtype = GGMLQuantizationType[name].value
    k = raw.shape[1] // mlx_gguf.FORMATS[qtype].block_bytes * mlx_gguf.FORMATS[qtype].block_values
    ref = quants.dequantize(raw, GGMLQuantizationType(qtype))
    assert ref.shape == (raw.shape[0], k) and ref.dtype == np.float32
    w = mx.array(raw)
    got32 = mlx_gguf.dequantize(w, qtype, k, mx.float32)
    got16 = mlx_gguf.dequantize(w, qtype, k, mx.bfloat16)
    mx.eval(got32, got16)
    got = np.array(got32)
    bad = np.flatnonzero(got != ref)
    assert bad.size == 0, (name, bad[:8], got.ravel()[bad[:8]], ref.ravel()[bad[:8]])
    assert mx.array_equal(got16, mx.array(ref).astype(mx.bfloat16)).item()


def test_rejects_wrong_row_bytes():
    mx = pytest.importorskip("mlx.core")
    with pytest.raises(ValueError):
        mlx_gguf.dequantize(mx.zeros((2, 100), dtype=mx.uint8), 12, 256, mx.float32)
