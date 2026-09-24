"""gguf_pack: the loader that runs llama.cpp GGUF blocks of the qwen3_5 hybrid.

A layout slip here is silent: a wrong value-head permutation still yields tensors of
the right shape, and the model runs and talks nonsense. So each inverse transform is
pinned against a numpy copy of the converter's FORWARD transform (llama.cpp
`_LinearAttentionVReorderBase._reorder_v_heads`), and the whole path — header to
config, name mapping, row permutations, the out_proj gather, the Metal matmul — is
checked end to end on a tiny synthetic GGUF against mlx-lm's own model holding the
same weights.
"""

import json
import os

import numpy as np
import pytest

from chad import gguf_pack

gguf = pytest.importorskip("gguf")

NK, R, D = 4, 3, 32         # key heads, value heads per key head, head dim


def _geometry(**kw) -> gguf_pack.Geometry:
    base = dict(n_layers=4, hidden=64, intermediate=128, n_heads=4, n_kv_heads=2,
                head_dim=32, k_heads=NK, v_heads=NK * R, k_head_dim=32, v_head_dim=D,
                conv_kernel=4, full_attention_interval=4, rope_theta=1e7,
                rope_sections=[2, 1, 1, 0], rope_dims=8, rms_eps=1e-6, context=4096,
                vocab=512, bos=1, eos=2)
    base.update(kw)
    return gguf_pack.Geometry(**base)


def _to_tiled(t: np.ndarray, axis: int, head_dim: int) -> np.ndarray:
    """llama.cpp's converter: value heads grouped by key head -> tiled order."""
    t = np.moveaxis(t, axis, 0)
    shape = t.shape
    t = t.reshape(NK, R, head_dim, *shape[1:]).swapaxes(0, 1).reshape(shape)
    return np.moveaxis(t, 0, axis)


@pytest.mark.parametrize("head_dim", [1, D])
def test_v_head_order_inverts_the_converter(head_dim):
    g = _geometry()
    grouped = np.random.default_rng(0).standard_normal((NK * R * head_dim, 5))
    stored = _to_tiled(grouped, 0, head_dim)
    assert np.array_equal(stored[gguf_pack.v_head_order(g, head_dim)], grouped)


def test_qkv_rows_keep_q_and_k_and_reorder_v():
    g = _geometry()
    qk = 2 * NK * g.k_head_dim
    grouped = np.arange(qk + NK * R * D)
    stored = np.concatenate([grouped[:qk], _to_tiled(grouped[qk:], 0, D)])
    p = gguf_pack.place("blk.0.attn_qkv.weight", g)
    assert p is not None
    perm = gguf_pack.row_order(p, g, len(stored))
    assert perm is not None and np.array_equal(stored[perm], grouped)


def test_out_proj_gather_reproduces_the_grouped_product():
    g = _geometry()
    rng = np.random.default_rng(1)
    w = rng.standard_normal((6, NK * R * D))          # [hidden, value dims], grouped
    x = rng.standard_normal(NK * R * D)
    stored = _to_tiled(w, 1, D)                        # columns reordered on disk
    assert np.allclose(stored @ x[gguf_pack.out_proj_gather(g)], w @ x)


def test_every_layer_role_places_and_mtp_is_skipped():
    g = _geometry()
    for role in gguf_pack._LAYER_ROLES:
        p = gguf_pack.place(f"blk.2.{role}", g)
        assert p is not None and p.path.startswith("language_model.model.layers.2.")
    assert gguf_pack.place("blk.4.nextn.eh_proj.weight", g) is None
    assert gguf_pack.place("output.weight", g).path == "language_model.lm_head"


def test_an_unknown_tensor_raises_instead_of_dropping():
    with pytest.raises(ValueError, match="unmapped"):
        gguf_pack.place("blk.0.attn_sinks.weight", _geometry())


def test_routing_predicate():
    assert gguf_pack.is_gguf_pack({"model_type": "qwen3_5", "chad_gguf": {"file": "x.gguf"}})
    assert not gguf_pack.is_gguf_pack({"model_type": "qwen3_5", "gguf": {"file": "x.gguf"}})
    assert not gguf_pack.is_gguf_pack({"model_type": "qwen3_5"})


# ---------------------------------------------------------------- a tiny real GGUF

def _hf_weights(g: gguf_pack.Geometry, rng) -> dict[str, np.ndarray]:
    """Random weights in mlx-lm's (post-sanitize, grouped) layout."""
    w: dict[str, np.ndarray] = {}
    h, f = g.hidden, g.intermediate
    kd, vd = g.k_heads * g.k_head_dim, g.v_heads * g.v_head_dim
    pre = "language_model.model."

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    w[pre + "embed_tokens.weight"] = mat(g.vocab, h)
    w["language_model.lm_head.weight"] = mat(g.vocab, h)
    w[pre + "norm.weight"] = 1 + mat(h)
    for i in range(g.n_layers):
        p = f"{pre}layers.{i}."
        w[p + "input_layernorm.weight"] = 1 + mat(h)
        w[p + "post_attention_layernorm.weight"] = 1 + mat(h)
        w[p + "mlp.gate_proj.weight"] = mat(f, h)
        w[p + "mlp.up_proj.weight"] = mat(f, h)
        w[p + "mlp.down_proj.weight"] = mat(h, f)
        if (i + 1) % g.full_attention_interval:
            a = p + "linear_attn."
            w[a + "in_proj_qkv.weight"] = mat(2 * kd + vd, h)
            w[a + "in_proj_z.weight"] = mat(vd, h)
            w[a + "in_proj_a.weight"] = mat(g.v_heads, h)
            w[a + "in_proj_b.weight"] = mat(g.v_heads, h)
            w[a + "A_log"] = np.log(rng.uniform(1, 16, g.v_heads)).astype(np.float32)
            w[a + "dt_bias"] = 1 + mat(g.v_heads)
            w[a + "conv1d.weight"] = mat(2 * kd + vd, g.conv_kernel, 1)
            w[a + "norm.weight"] = 1 + mat(g.v_head_dim)
            w[a + "out_proj.weight"] = mat(h, vd)
        else:
            s = p + "self_attn."
            w[s + "q_proj.weight"] = mat(2 * g.n_heads * g.head_dim, h)
            w[s + "k_proj.weight"] = mat(g.n_kv_heads * g.head_dim, h)
            w[s + "v_proj.weight"] = mat(g.n_kv_heads * g.head_dim, h)
            w[s + "o_proj.weight"] = mat(h, g.n_heads * g.head_dim)
            w[s + "q_norm.weight"] = 1 + mat(g.head_dim)
            w[s + "k_norm.weight"] = 1 + mat(g.head_dim)
    return w


_GGUF_ROLE = {v.path: k for k, v in gguf_pack._LAYER_ROLES.items()}


def _write_gguf(path: str, g: gguf_pack.Geometry, w: dict[str, np.ndarray]) -> dict:
    """Write `w` the way llama.cpp's converter would (tiled value heads, -exp(A_log),
    squeezed conv), every projection Q8_0. Returns the Q8_0 round-tripped float
    weights in mlx-lm's layout — what the loaded model must compute with."""
    q8 = gguf.GGMLQuantizationType.Q8_0
    wr = gguf.GGUFWriter(path, "qwen35")
    a = "qwen35."
    wr.add_uint32(a + "block_count", g.n_layers)
    wr.add_uint32(a + "context_length", g.context)
    wr.add_uint32(a + "embedding_length", g.hidden)
    wr.add_uint32(a + "feed_forward_length", g.intermediate)
    wr.add_uint32(a + "attention.head_count", g.n_heads)
    wr.add_uint32(a + "attention.head_count_kv", g.n_kv_heads)
    wr.add_uint32(a + "attention.key_length", g.head_dim)
    wr.add_uint32(a + "attention.value_length", g.head_dim)
    wr.add_float32(a + "attention.layer_norm_rms_epsilon", g.rms_eps)
    wr.add_float32(a + "rope.freq_base", g.rope_theta)
    wr.add_uint32(a + "rope.dimension_count", g.rope_dims)
    wr.add_array(a + "rope.dimension_sections", g.rope_sections)
    wr.add_uint32(a + "ssm.conv_kernel", g.conv_kernel)
    wr.add_uint32(a + "ssm.state_size", g.k_head_dim)
    wr.add_uint32(a + "ssm.group_count", g.k_heads)
    wr.add_uint32(a + "ssm.time_step_rank", g.v_heads)
    wr.add_uint32(a + "ssm.inner_size", g.v_heads * g.v_head_dim)
    wr.add_uint32(a + "full_attention_interval", g.full_attention_interval)
    wr.add_uint32("tokenizer.ggml.bos_token_id", g.bos)
    wr.add_uint32("tokenizer.ggml.eos_token_id", g.eos)
    kd = g.k_heads * g.k_head_dim
    expect: dict[str, np.ndarray] = {}
    for key, v in w.items():
        if key.endswith("embed_tokens.weight"):
            name = "token_embd.weight"
        elif key == "language_model.lm_head.weight":
            name = "output.weight"
        elif key == "language_model.model.norm.weight":
            name = "output_norm.weight"
        else:
            parts = key.split(".")
            layer, rest = parts[3], ".".join(parts[4:])
            role = rest.removesuffix(".weight")
            name = f"blk.{layer}.{_GGUF_ROLE[role]}"
        stored = v
        if "linear_attn." in key:
            if key.endswith("in_proj_qkv.weight") or key.endswith("conv1d.weight"):
                qk = stored[:2 * kd]
                stored = np.concatenate([qk, _to_tiled(stored[2 * kd:], 0, D)])
            elif key.endswith("in_proj_z.weight"):
                stored = _to_tiled(stored, 0, D)
            elif key.endswith(("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias")):
                stored = _to_tiled(stored, 0, 1)
            elif key.endswith("out_proj.weight"):
                stored = _to_tiled(stored, 1, D)
            if key.endswith("A_log"):
                stored = -np.exp(stored)
            if key.endswith("conv1d.weight"):
                stored = stored[:, :, 0]
        if stored.ndim == 2 and not key.endswith("conv1d.weight"):
            raw = gguf.quants.quantize(stored, q8)
            wr.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=q8)
            back = gguf.quants.dequantize(raw, q8)
            if key.endswith("out_proj.weight"):
                back = back[:, gguf_pack.v_head_order(g, D)]
            else:
                p = gguf_pack.place(name, g)
                assert p is not None
                perm = gguf_pack.row_order(p, g, back.shape[0])
                back = back if perm is None else back[perm]
            expect[key] = back
        else:
            wr.add_tensor(name, np.ascontiguousarray(stored, dtype=np.float32))
            expect[key] = v
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    return expect


@pytest.fixture
def tiny(tmp_path):
    g = _geometry()
    w = _hf_weights(g, np.random.default_rng(7))
    path = str(tmp_path / "tiny.gguf")
    expect = _write_gguf(path, g, w)
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "tokenizer_config.json").write_text("{}")
    (donor / "tokenizer.json").write_text("{}")
    out = gguf_pack.materialize(path, cache_root=str(tmp_path / "cache"),
                                donor_dir=str(donor))
    return g, expect, out


def test_materialize_writes_a_routable_config(tiny):
    g, _, out = tiny
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    assert gguf_pack.is_gguf_pack(cfg)
    tc = cfg["text_config"]
    assert (tc["linear_num_value_heads"], tc["linear_num_key_heads"]) == (g.v_heads, g.k_heads)
    assert tc["num_hidden_layers"] == g.n_layers and tc["vocab_size"] == g.vocab
    assert os.path.isfile(os.path.join(out, "tokenizer_config.json"))


def test_loaded_model_matches_mlx_lm_on_the_same_weights(tiny):
    mx = pytest.importorskip("mlx.core")
    q35 = pytest.importorskip("mlx_lm.models.qwen3_5")
    g, expect, out = tiny
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    model = gguf_pack.load(out, cfg)
    ref = q35.Model(q35.ModelArgs(model_type="qwen3_5", text_config=cfg["text_config"]))
    ref.load_weights([(k, mx.array(v).astype(mx.float32 if k.endswith("A_log")
                                              else mx.bfloat16))
                      for k, v in expect.items()], strict=True)
    exact = q35.Model(q35.ModelArgs(model_type="qwen3_5", text_config=cfg["text_config"]))
    exact.load_weights([(k, mx.array(v)) for k, v in expect.items()], strict=True)

    def rel(a, b) -> float:
        return float(mx.linalg.norm(a - b)) / float(mx.linalg.norm(b))

    ids = mx.array(np.random.default_rng(3).integers(0, g.vocab, (1, 23)))
    for n in (1, 23):       # the GEMV (decode) and dequantize+matmul (prefill) paths
        truth = exact(ids[:, :n])
        got = model(ids[:, :n]).astype(mx.float32)
        bf16 = ref(ids[:, :n]).astype(mx.float32)
        # Both run a bf16 stream; the loaded model may round differently from mlx-lm's
        # (fp32 accumulation in the fused kernel), but no further from the fp32 truth
        # than bf16 itself puts mlx-lm. A layout error lands at ~100%.
        assert rel(got, truth) < 2 * rel(bf16, truth) + 1e-3, (n, rel(got, truth))


def test_fastpath_decode_matches_the_unfused_model(tiny, tmp_path):
    """The compiled S=1 step and the same-format fusions change the op graph, not
    the math: greedy-width decode logits stay within bf16 rounding of the stock
    module graph on the same GGUF."""
    mx = pytest.importorskip("mlx.core")
    from chad import mlx_fastpath

    g, _, out = tiny
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    ids = mx.array(np.random.default_rng(5).integers(0, g.vocab, (1, 9)))

    def run(model):
        cache = model.make_cache()
        logits = [model(ids[:, :6], cache=cache)[:, -1]]
        for i in range(6, 9):                 # S=1 decode steps through the cache
            logits.append(model(ids[:, i:i + 1], cache=cache)[:, -1])
        return mx.concatenate(logits).astype(mx.float32)

    stock = run(gguf_pack.load(out, cfg))
    fast_model = gguf_pack.load(out, cfg)
    assert mlx_fastpath.install(fast_model)
    fast = run(fast_model)
    err = float(mx.linalg.norm(fast - stock)) / float(mx.linalg.norm(stock))
    assert err < 0.02, err


def test_a_changed_file_gets_its_own_directory(tiny, tmp_path):
    """The directory is the engine's model id, and warm-prefix checkpoints are keyed on
    it: a file replaced under the same name must not inherit the old one's."""
    _, _, out = tiny
    with open(os.path.join(out, "config.json")) as f:
        path = json.load(f)["chad_gguf"]["file"]
    donor = str(tmp_path / "donor")
    again = gguf_pack.materialize(path, cache_root=str(tmp_path / "cache"), donor_dir=donor)
    assert again == out
    with open(path, "ab") as f:          # same name, different bytes
        f.write(b"\0" * 32)
    assert gguf_pack.materialize(path, cache_root=str(tmp_path / "cache"),
                                 donor_dir=donor) != out


def test_materialize_needs_a_tokenizer(tiny, tmp_path):
    _, _, out = tiny
    with open(os.path.join(out, "config.json")) as f:
        path = json.load(f)["chad_gguf"]["file"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="tokenizer.json"):
        gguf_pack.materialize(path, cache_root=str(tmp_path / "other"), donor_dir=str(empty))
