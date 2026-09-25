"""Loader for llama.cpp GGUF checkpoints of the qwen3_5 dense hybrid (Qwen3.8-27B class).

Unsloth's GGUFs of this model beat any MLX affine quant of the same size, and the gap
is the formats themselves: most of the bytes are i-quants (codebook grids with sign
bits, non-uniform 4-bit tables) fitted against an importance matrix. MLX's
`scale * q + bias` container cannot represent them, so re-quantizing would throw the
quality away. This loader keeps every tensor in its original GGUF blocks and runs
them through chad's own Metal decoders (`mlx_gguf`): nothing is dequantized ahead of
time, and resident memory is the file's size.

A GGUF is one file with no tokenizer.json and no HF config, while the rest of chad
reads both out of a model directory. `materialize()` builds that directory once, in
chad's cache: a `config.json` whose `text_config` is derived from the GGUF header
(plus a `gguf` key naming the file, which is what routes the engine here), and the
tokenizer files copied from a donor checkpoint of the same model. The donor's chat
template, not the GGUF's, becomes the model's template: the GGUF's defaults
`reasoning_effort` to xhigh where chad renders medium, and a byte-stable prompt is
worth more than the file's defaults.

What the loader undoes, and why. llama.cpp's converter rewrites a Qwen3.5 checkpoint
in four ways that mlx-lm's `qwen3_5` modules do not expect:

* Linear-attention value heads are stored in TILED order (k-head fastest) instead of
  grouped by k-head, so ggml can broadcast q/k across value heads with a repeat. Every
  tensor with a value-head axis moved: the V rows of `in_proj_qkv`, the rows of
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log` and `dt_bias`, the V channels of the
  conv kernel, and the INPUT columns of `out_proj`. Rows are independent blocks, so a
  row permutation is a permutation of raw block rows at load. `out_proj`'s columns
  are not: a 128-wide head is half a 256-value block. That one permutation is applied
  to the activation instead, as a gather in front of the matmul.
* `A_log` is stored as `-exp(A_log)`; the loader takes the log back (float32).
* The conv kernel is stored squeezed, `[channels, taps]`.
* The RMSNorm weights carry the Qwen3.8 `+1` shift already, except the GDN's gated
  norm. That is exactly mlx-lm's post-sanitize convention, so they load as stored.

The file's multi-token-prediction block (the last `blk.N`) is not read: chad drafts
with DFlash2, which reads the target's residual stream and does not care how the
target is quantized.
"""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Optional, cast

from .diag import log

if TYPE_CHECKING:  # mlx is imported lazily inside the functions so the module loads on Linux
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

_ARCH = "qwen35"
# Projections with fewer output rows than this load as plain bf16 linears.
_DENSE_ROWS = 64
# Chat template and tokenizer donor: the same Qwen3.8-27B vocabulary as every GGUF of
# it, from the repo that also carries the drafter (mlx_dflash.DONORS) and no target
# weights, so a GGUF session depends on ~1.2 GB of sidecar and never on another pack.
TOKENIZER_DONOR = "nathansutton/Qwen3.8-27B-DFlash2-MLX"
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                    "generation_config.json")
# Everything a GGUF session reads from the sidecar. `materialize()` copies whichever
# tokenizer files exist, so a download cut short before the template must not count
# as done: the pack directory built from it would be reused without one.
_SIDECAR_FILES = _TOKENIZER_FILES + ("dflash/config.json", "dflash/model.safetensors")
# The donor template renders medium by default; the GGUF's own template says xhigh.
REASONING_EFFORT_DEFAULT = "medium"


# Bump when what `materialize()` writes changes (the derived config, the donor files):
# it is part of the directory's name, so a stale directory is never reused.
_SCHEMA = 3
_KEY = "chad_gguf"


def is_gguf_pack(config: dict) -> bool:
    """True for a model directory `materialize()` built around a GGUF file."""
    return _KEY in config


# The default file is one of Unsloth's, fetched from their repo: re-hosting 13 GB that
# they revise would only go stale. A hub GGUF is spelled `owner/repo/file.gguf`.
HUB_REPO = "unsloth/Qwen3.8-27B-GGUF"
# What one GGUF session downloads besides the file: the drafter (1.15 GB) and tokenizer.
SIDECAR_GB = 1.2
# Download sizes for the disk preflight and the consent line — display honesty, not
# accounting: the files this loader has run. Any other file is sized by the hub, since
# the same repo holds 28 GB Q8_0 and 54 GB BF16 files that no table default covers.
_HUB_GB = {"Qwen3.8-27B-UD-Q2_K_XL.gguf": 9.8, "Qwen3.8-27B-UD-IQ3_XXS.gguf": 10.9,
           "Qwen3.8-27B-UD-IQ3_S.gguf": 12.0, "Qwen3.8-27B-UD-Q3_K_XL.gguf": 13.2,
           "Qwen3.8-27B-UD-IQ4_XS.gguf": 14.3}


def hub_spec(spec: str) -> Optional[tuple[str, str]]:
    """`(repo_id, filename)` when `spec` names a GGUF inside a hub repo,
    `owner/repo/path/to/file.gguf`; None for a local path or a plain repo id. A local
    file always wins: `resolve_file` looks at the disk first."""
    s = spec.strip()
    if not s.lower().endswith(".gguf") or s.startswith(("/", "~", ".")):
        return None
    parts = s.split("/")
    if len(parts) < 3 or not all(parts):
        return None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _hub_cached(repo_id: str, filename: str) -> Optional[str]:
    """`filename` of `repo_id` in the local HF cache, or None (the hub's "known
    missing upstream" sentinel folded into None)."""
    from huggingface_hub import try_to_load_from_cache
    hit = try_to_load_from_cache(repo_id, filename)
    return hit if isinstance(hit, str) else None


def resolve_file(spec: str,
                 cached: Callable[[str, str], Optional[str]] = _hub_cached) -> Optional[str]:
    """The absolute path of the GGUF file `spec` names — a local path, or a hub file
    (`hub_spec`) already in the HF cache — or None when it names anything else (a model
    directory, a plain repo id, a path that is not there, a hub file not yet fetched).
    `~` is expanded and the suffix matched case-insensitively, so `--model ~/x.GGUF`
    routes here instead of falling through to the hub as a repo id and failing there."""
    path = os.path.expanduser(spec.strip())
    if path.lower().endswith(".gguf") and os.path.isfile(path):
        return os.path.abspath(path)
    hub = hub_spec(spec)
    if hub is not None:
        hit = cached(*hub)
        return os.path.abspath(hit) if hit else None
    return None


def hub_cached(spec: str,
               cached: Callable[[str, str], Optional[str]] = _hub_cached) -> bool:
    """True when the hub GGUF `spec` names AND the sidecar it decodes with (the
    drafter and tokenizer from `TOKENIZER_DONOR`) are on this disk — what "the model
    is downloaded" means for a GGUF session."""
    hub = hub_spec(spec)
    if hub is None:
        return False
    return cached(*hub) is not None and all(
        cached(TOKENIZER_DONOR, f) is not None for f in _SIDECAR_FILES)


def fetch_hub(spec: str) -> str:
    """Download the hub GGUF `spec` names and the sidecar repo into the HF cache
    (resumable, progress on stderr); returns the file's local path."""
    from huggingface_hub import hf_hub_download, snapshot_download
    hub = hub_spec(spec)
    if hub is None:
        raise ValueError(f"{spec!r} is not owner/repo/file.gguf")
    snapshot_download(TOKENIZER_DONOR)
    return hf_hub_download(*hub)


def _hub_file_gb(repo_id: str, filename: str) -> Optional[float]:
    """The size in GB the hub reports for `filename` of `repo_id`, or None when the
    hub cannot be asked (offline, not found, network down)."""
    import httpx
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    try:
        size = get_hf_file_metadata(hf_hub_url(repo_id, filename)).size
    except (OSError, httpx.HTTPError):
        return None
    return size / 1e9 if size is not None else None


def hub_download_gb(spec: str,
                    ask_hub: Callable[[str, str], Optional[float]] = _hub_file_gb) -> float:
    """Approximate download for the hub GGUF `spec` names plus its sidecar. A file
    outside the table is sized by the hub; when the hub cannot answer, the download
    cannot start either, so the heaviest known file stands in for the message."""
    hub = hub_spec(spec)
    if hub is None:
        return max(_HUB_GB.values()) + SIDECAR_GB
    known = _HUB_GB.get(os.path.basename(hub[1]))
    gb = known if known is not None else ask_hub(*hub)
    return (gb if gb is not None else max(_HUB_GB.values())) + SIDECAR_GB


def file_size(config: dict) -> int:
    """The GGUF file's size in bytes, from the stamp `materialize()` wrote: the weight
    bytes of a model directory that has no safetensors to measure."""
    return int(config[_KEY]["size"])


@dataclass(frozen=True)
class Geometry:
    """The header facts every layout transform needs."""
    n_layers: int           # decoder layers, NOT counting the MTP block
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    k_heads: int            # linear-attention key heads (ssm.group_count)
    v_heads: int            # linear-attention value heads (ssm.time_step_rank)
    k_head_dim: int         # ssm.state_size
    v_head_dim: int
    conv_kernel: int
    full_attention_interval: int
    rope_theta: float
    rope_sections: list[int]
    rope_dims: int
    rms_eps: float
    context: int
    vocab: int
    bos: int
    eos: int


def _field(reader, key: str):
    f = reader.fields.get(key)
    if f is None:
        raise ValueError(f"GGUF header has no {key!r}")
    return f.contents()


def read_geometry(reader) -> Geometry:
    arch = _field(reader, "general.architecture")
    if arch != _ARCH:
        raise ValueError(f"GGUF architecture {arch!r}; this loader implements {_ARCH!r}")

    def a(key: str):
        return _field(reader, f"{_ARCH}.{key}")

    nextn = reader.fields.get(f"{_ARCH}.nextn_predict_layers")
    n_mtp = int(nextn.contents()) if nextn is not None else 0
    v_heads, inner = int(a("ssm.time_step_rank")), int(a("ssm.inner_size"))
    embd = next((t for t in reader.tensors if t.name == "token_embd.weight"), None)
    if embd is None:
        raise ValueError("GGUF has no token_embd.weight")
    return Geometry(
        n_layers=int(a("block_count")) - n_mtp,
        hidden=int(a("embedding_length")),
        intermediate=int(a("feed_forward_length")),
        n_heads=int(a("attention.head_count")),
        n_kv_heads=int(a("attention.head_count_kv")),
        head_dim=int(a("attention.key_length")),
        k_heads=int(a("ssm.group_count")),
        v_heads=v_heads,
        k_head_dim=int(a("ssm.state_size")),
        v_head_dim=inner // v_heads,
        conv_kernel=int(a("ssm.conv_kernel")),
        full_attention_interval=int(a("full_attention_interval")),
        rope_theta=float(a("rope.freq_base")),
        rope_sections=[int(s) for s in a("rope.dimension_sections")],
        rope_dims=int(a("rope.dimension_count")),
        rms_eps=float(a("attention.layer_norm_rms_epsilon")),
        context=int(a("context_length")),
        vocab=int(embd.shape[1]),
        bos=int(_field(reader, "tokenizer.ggml.bos_token_id")),
        eos=int(_field(reader, "tokenizer.ggml.eos_token_id")),
    )


def text_config(g: Geometry) -> dict:
    """The `text_config` mlx-lm's qwen3_5 ModelArgs reads, from the header alone."""
    sections = [s for s in g.rope_sections if s] or [11, 11, 10]
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": g.hidden,
        "intermediate_size": g.intermediate,
        "num_hidden_layers": g.n_layers,
        "num_attention_heads": g.n_heads,
        "num_key_value_heads": g.n_kv_heads,
        "head_dim": g.head_dim,
        "rms_norm_eps": g.rms_eps,
        "vocab_size": g.vocab,
        "max_position_embeddings": g.context,
        "linear_num_value_heads": g.v_heads,
        "linear_num_key_heads": g.k_heads,
        "linear_key_head_dim": g.k_head_dim,
        "linear_value_head_dim": g.v_head_dim,
        "linear_conv_kernel_dim": g.conv_kernel,
        "full_attention_interval": g.full_attention_interval,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attn_output_gate": True,
        "partial_rotary_factor": g.rope_dims / g.head_dim,
        "rope_parameters": {
            "type": "default",
            "mrope_interleaved": True,
            "mrope_section": sections,
            "partial_rotary_factor": g.rope_dims / g.head_dim,
            "rope_theta": g.rope_theta,
        },
        "bos_token_id": g.bos,
        "eos_token_id": g.eos,
    }


# ---------------------------------------------------------------- value-head order

def v_head_order(g: Geometry, head_dim: int) -> "np.ndarray":
    """Index array that takes a TILED value-head axis (as stored) to grouped order:
    `grouped = tiled[idx]`. Also the gather that feeds `out_proj`: the grouped
    activation, taken at this index, lands in the stored (tiled) column order...
    see `out_proj_gather` for that direction."""
    import numpy as np
    per_k = g.v_heads // g.k_heads
    return (np.arange(g.v_heads * head_dim)
            .reshape(per_k, g.k_heads, head_dim).transpose(1, 0, 2).reshape(-1))


def out_proj_gather(g: Geometry) -> "np.ndarray":
    """`x_tiled = x_grouped[idx]`: the activation permutation in front of `out_proj`,
    whose stored columns are in tiled value-head order."""
    import numpy as np
    per_k = g.v_heads // g.k_heads
    return (np.arange(g.v_heads * g.v_head_dim)
            .reshape(g.k_heads, per_k, g.v_head_dim).transpose(1, 0, 2).reshape(-1))


@dataclass(frozen=True)
class Placement:
    """Where one GGUF tensor goes in the mlx-lm tree, and what undoes its layout."""
    path: str                   # mlx-lm parameter path, without the `.weight` suffix
    kind: str                   # "linear" | "embedding" | "vector" | "conv" | "a_log"
    row_perm: str = ""          # "qkv" | "v" | "v1" (per-head scalar) | ""
    col_gather: bool = False    # out_proj: permute the activation, not the weight


_LAYER_ROLES: dict[str, Placement] = {
    "attn_norm.weight": Placement("input_layernorm", "vector"),
    "post_attention_norm.weight": Placement("post_attention_layernorm", "vector"),
    "ffn_gate.weight": Placement("mlp.gate_proj", "linear"),
    "ffn_up.weight": Placement("mlp.up_proj", "linear"),
    "ffn_down.weight": Placement("mlp.down_proj", "linear"),
    # linear-attention (GatedDeltaNet) layers
    "attn_qkv.weight": Placement("linear_attn.in_proj_qkv", "linear", row_perm="qkv"),
    "attn_gate.weight": Placement("linear_attn.in_proj_z", "linear", row_perm="v"),
    "ssm_alpha.weight": Placement("linear_attn.in_proj_a", "linear", row_perm="v1"),
    "ssm_beta.weight": Placement("linear_attn.in_proj_b", "linear", row_perm="v1"),
    "ssm_a": Placement("linear_attn.A_log", "a_log", row_perm="v1"),
    "ssm_dt.bias": Placement("linear_attn.dt_bias", "vector", row_perm="v1"),
    "ssm_conv1d.weight": Placement("linear_attn.conv1d", "conv", row_perm="qkv"),
    "ssm_norm.weight": Placement("linear_attn.norm", "vector"),
    "ssm_out.weight": Placement("linear_attn.out_proj", "linear", col_gather=True),
    # full-attention layers
    "attn_q.weight": Placement("self_attn.q_proj", "linear"),
    "attn_k.weight": Placement("self_attn.k_proj", "linear"),
    "attn_v.weight": Placement("self_attn.v_proj", "linear"),
    "attn_output.weight": Placement("self_attn.o_proj", "linear"),
    "attn_q_norm.weight": Placement("self_attn.q_norm", "vector"),
    "attn_k_norm.weight": Placement("self_attn.k_norm", "vector"),
}

_GLOBAL_ROLES: dict[str, Placement] = {
    "token_embd.weight": Placement("language_model.model.embed_tokens", "embedding"),
    "output.weight": Placement("language_model.lm_head", "linear"),
    "output_norm.weight": Placement("language_model.model.norm", "vector"),
}


def place(name: str, g: Geometry) -> Optional[Placement]:
    """The placement for GGUF tensor `name`, None for the MTP block's tensors.
    Raises on a name this loader does not know: silently dropping a tensor would
    leave a randomly initialized module in the model."""
    if name in _GLOBAL_ROLES:
        return _GLOBAL_ROLES[name]
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "blk" and parts[1].isdigit():
        layer = int(parts[1])
        if layer >= g.n_layers:
            return None
        role = _LAYER_ROLES.get(parts[2])
        if role is not None:
            return Placement(f"language_model.model.layers.{layer}.{role.path}",
                             role.kind, role.row_perm, role.col_gather)
    raise ValueError(f"unmapped GGUF tensor {name!r}")


def row_order(p: Placement, g: Geometry, rows: int) -> Optional["np.ndarray"]:
    """The row permutation that takes the stored rows to mlx-lm's order, or None."""
    import numpy as np
    if p.row_perm == "v1":
        return v_head_order(g, 1)
    if p.row_perm == "v":
        return v_head_order(g, g.v_head_dim)
    if p.row_perm == "qkv":
        qk = 2 * g.k_heads * g.k_head_dim
        if rows != qk + g.v_heads * g.v_head_dim:
            raise ValueError(f"{p.path}: {rows} rows, expected q|k|v")
        return np.concatenate([np.arange(qk), qk + v_head_order(g, g.v_head_dim)])
    return None


# ---------------------------------------------------------------- model directory

def _place(src: str, dst: str) -> None:
    """Copy `src` to `dst` through a private temp file and a rename, so `dst` is
    never seen half-written by a process that builds the same directory at once."""
    tmp = f"{dst}.{os.getpid()}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def materialize(gguf_path: str, cache_root: Optional[str] = None,
                donor_dir: Optional[str] = None) -> str:
    """Build (once) the model directory the engine loads a GGUF through; return it.

    The directory holds no weights: `config.json` names the GGUF file, and the
    tokenizer files come from `donor_dir` (default: the cached donor checkpoint).
    The directory's name carries a hash of the file's path, size and mtime (to the
    nanosecond: a file replaced by one of the same size within the same second must
    not be mistaken for it) and of `_SCHEMA`, because it is the engine's model id:
    warm-prefix checkpoints are keyed on it, and a checkpoint of one file's hybrid
    cache must never load into another's (an upload fixed under the same filename, a
    second copy elsewhere). Two processes building it at once write identical bytes,
    and every file lands by rename, the config last: a reader that finds the config
    finds whole tokenizer files."""
    import gguf

    gguf_path = os.path.abspath(gguf_path)
    st = os.stat(gguf_path)
    stamp = {"file": gguf_path, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
             "schema": _SCHEMA}
    digest = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:12]
    root = cache_root or os.path.join(os.path.expanduser("~/.cache/chad"), "gguf")
    out = os.path.join(root, f"{os.path.splitext(os.path.basename(gguf_path))[0]}-{digest}")
    cfg_path = os.path.join(out, "config.json")
    if os.path.isfile(cfg_path):
        return out
    if donor_dir is None:
        from huggingface_hub import snapshot_download
        donor_dir = snapshot_download(TOKENIZER_DONOR, allow_patterns=list(_TOKENIZER_FILES))
    g = read_geometry(gguf.GGUFReader(gguf_path))
    os.makedirs(out, exist_ok=True)
    if not os.path.isfile(os.path.join(donor_dir, "tokenizer.json")):
        raise ValueError(f"tokenizer donor {donor_dir} has no tokenizer.json")
    for name in _TOKENIZER_FILES:
        src = os.path.join(donor_dir, name)
        if os.path.isfile(src):
            _place(src, os.path.join(out, name))
    from importlib.metadata import version
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        # transformers 5 takes a local tokenizer whose config names no version for a
        # Mistral one with a known-bad split regex, and warns at every load.
        "transformers_version": version("transformers"),
        "tie_word_embeddings": False,
        "eos_token_id": [g.eos, g.bos],
        "text_config": text_config(g),
        _KEY: stamp,
    }
    tmp = f"{cfg_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(config, f, indent=1)
    os.replace(tmp, cfg_path)
    log.info("GGUF model directory built at %s for %s", out, gguf_path)
    return out


# ---------------------------------------------------------------- modules and load

_CLASSES: Optional[tuple[type, type]] = None


def _module_classes() -> tuple[type, type]:
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import mlx.core as mx
    import mlx.nn as nn

    from . import mlx_gguf

    class GGUFLinear(nn.Module):
        """One projection in its GGUF blocks: `weight` is uint8 `[out, row_bytes]`.

        Stands in for `nn.QuantizedLinear`. `gather`, when present, permutes the
        activation into the stored column order first (see the module docstring)."""

        def __init__(self, weight, qtype: int, in_dims: int, gather=None):
            super().__init__()
            self.weight = weight
            if gather is not None:
                self.gather = gather
            self.qtype, self.in_dims = qtype, in_dims
            self.out_dims = weight.shape[0]

        def __call__(self, x):
            if "gather" in self:
                x = mx.take(x, self["gather"], axis=-1)
            return mlx_gguf.matmul(x, self.weight, self.qtype, self.in_dims)

    class GGUFEmbedding(nn.Module):
        """The embedding table in its GGUF blocks: a lookup gathers the token rows'
        raw blocks and decodes only those."""

        def __init__(self, weight, qtype: int, dims: int, dtype):
            super().__init__()
            self.weight = weight
            self.qtype, self.dims, self.dtype = qtype, dims, dtype

        def __call__(self, ids):
            rows = self.weight[ids.reshape(-1)]
            out = mlx_gguf.dequantize(rows, self.qtype, self.dims, self.dtype)
            return out.reshape(*ids.shape, self.dims)

    _CLASSES = (GGUFLinear, GGUFEmbedding)
    return _CLASSES


def is_gguf_linear(module: "nn.Module") -> bool:
    """True for a projection this loader left in its GGUF blocks."""
    return _CLASSES is not None and isinstance(module, _CLASSES[0])


def fuse_rows(mods: "list[nn.Module]") -> "nn.Module":
    """One GGUF projection whose output rows are `mods`' rows stacked in order.

    Rows are independent runs of blocks, so stacking the raw bytes is exact. Only
    same-format, same-width partners fuse (a GGUF mixes formats per role and per
    layer), and never one with an input gather."""
    import mlx.core as mx
    first = mods[0]
    if not all(is_gguf_linear(m) for m in mods):
        raise ValueError("fuse_rows: not all GGUF projections")
    if len({(m.qtype, m.in_dims) for m in mods}) != 1 or any("gather" in m for m in mods):
        raise ValueError("fuse_rows: partners differ in format or width")
    w = mx.contiguous(mx.concatenate([m.weight for m in mods], axis=0))
    mx.eval(w)
    GGUFLinear, _ = _module_classes()
    return GGUFLinear(w, first.qtype, first.in_dims)


def row_slice(module: "nn.Module", start: int, stop: int) -> "nn.Module":
    """Rows [start, stop) of a GGUF projection as a projection of its own; the weight
    is a view of the original's buffer, not a copy."""
    import mlx.core as mx
    w = module.weight[start:stop]
    mx.eval(w)
    GGUFLinear, _ = _module_classes()
    return GGUFLinear(w, module.qtype, module.in_dims)


# The loaded model, saved once beside the config. Reading the GGUF means parsing its
# header in Python and copying every tensor out of a numpy memmap into its own Metal
# buffer (rows permuted on the way), ~50 s for the 13 GB default on every start; the
# saved arrays load through mlx's own reader like any safetensors pack. They are the
# same GGUF blocks in the same bytes, not a requantization. Bump the version when
# `_convert` builds different arrays or modules from the same file.
_REPACK = "weights.safetensors"
_REPACK_KEY = "chad_gguf_repack"
_REPACK_VERSION = 1
# Free space left on the volume after the repack is written.
_REPACK_HEADROOM = 2 << 30


@dataclass(frozen=True)
class Slot:
    """One module `_convert` put in place of mlx-lm's default: what the repack
    needs to build it again around the saved weight."""
    kind: str           # "linear" (GGUF blocks), "embedding", or "dense" (bf16 Linear)
    qtype: int
    in_dims: int
    out_dims: int


def load(model_path: str, config: Optional[dict] = None) -> "nn.Module":
    """Build mlx-lm's qwen3_5 model around the GGUF that `model_path`'s config names.

    Every quantized tensor stays in its blocks; float tensors (norms, the conv kernel,
    `dt_bias`) are cast to bf16, and `A_log` stays float32 as mlx-lm keeps it. The
    first load converts from the file and saves the result in `model_path`; every
    later one reads that."""
    if config is None:
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
    if not is_gguf_pack(config):
        raise ValueError(f"{model_path} is not a GGUF model directory")
    model = _load_repack(model_path, config)
    if model is not None:
        return model
    model, slots = _convert(config)
    _save_repack(model_path, model, slots)
    return model


def _assign(model: "nn.Module", path: str, value: "nn.Module | mx.array") -> None:
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else parent[part]
    parent[parts[-1]] = value


def _empty_model(config: dict) -> "nn.Module":
    from mlx_lm.models import qwen3_5 as q35
    return q35.Model(q35.ModelArgs(model_type="qwen3_5", text_config=config["text_config"]))


def _load_repack(model_path: str, config: dict) -> "Optional[nn.Module]":
    """The model from the saved repack, or None when there is none or it was written
    by another version of `_convert` (the caller converts again and overwrites it)."""
    import mlx.core as mx
    import mlx.nn as nn

    path = os.path.join(model_path, _REPACK)
    if not os.path.isfile(path):
        return None
    try:
        # SAFETY: mx.load's stub unions .npy/.npz returns; a .safetensors path with
        # return_metadata=True returns (name -> array, name -> string metadata).
        arrays, meta = cast("tuple[dict[str, mx.array], dict[str, str]]",
                            mx.load(path, return_metadata=True))
        slots = _parse_slots(meta.get(_REPACK_KEY, ""))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
        log.warning("GGUF repack at %s unreadable (%s); converting again", path, e)
        return None
    if slots is None:
        return None
    GGUFLinear, GGUFEmbedding = _module_classes()
    model = _empty_model(config)
    for name, s in slots.items():
        w = arrays[name + ".weight"]
        if s.kind == "linear":
            _assign(model, name, GGUFLinear(w, s.qtype, s.in_dims, arrays.get(name + ".gather")))
        elif s.kind == "embedding":
            _assign(model, name, GGUFEmbedding(w, s.qtype, s.in_dims, mx.bfloat16))
        else:
            lin = nn.Linear(s.in_dims, s.out_dims, bias=False)
            lin.weight = w
            _assign(model, name, lin)
    try:
        model.load_weights(list(arrays.items()), strict=True)
    except ValueError as e:
        log.warning("GGUF repack at %s does not fit the model (%s); converting again",
                    path, e)
        return None
    model.eval()
    mx.eval(model.parameters())
    log.info("GGUF loaded from repack %s (%s)", path, config[_KEY]["file"])
    return model


def _parse_slots(text: str) -> Optional[dict[str, Slot]]:
    """The repack's module table, or None when it is missing or another version's.
    A malformed table raises KeyError/TypeError/ValueError for the caller to treat
    as unreadable."""
    if not text:
        return None
    head = json.loads(text)
    if head["version"] != _REPACK_VERSION:
        return None
    return {name: Slot(kind=str(s["kind"]), qtype=int(s["qtype"]), in_dims=int(s["in_dims"]),
                       out_dims=int(s["out_dims"]))
            for name, s in head["slots"].items()}


def _save_repack(model_path: str, model: "nn.Module", slots: dict[str, Slot]) -> None:
    """Write the converted model beside its config, by rename so a reader never sees
    half a file. Best effort: without the space for it every start converts, as
    before, and says so."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    arrays = dict(tree_flatten(model.parameters()))
    need = sum(a.nbytes for a in arrays.values())
    free = shutil.disk_usage(model_path).free
    if free < need + _REPACK_HEADROOM:
        log.warning("GGUF repack skipped: %.1f GB free at %s, needs %.1f GB; every start "
                    "will convert the file again", free / 1e9, model_path, need / 1e9)
        return
    _prune_stale_repacks(os.path.dirname(os.path.abspath(model_path)))
    head = {"version": _REPACK_VERSION,
            "slots": {name: asdict(s) for name, s in slots.items()}}
    final = os.path.join(model_path, _REPACK)
    tmp = os.path.join(model_path, f"weights.{os.getpid()}.tmp.safetensors")
    try:
        mx.save_safetensors(tmp, arrays, metadata={_REPACK_KEY: json.dumps(head)})
        os.replace(tmp, final)
        log.info("GGUF repack written: %s (%.1f GB)", final, need / 1e9)
    except OSError as e:
        log.warning("GGUF repack not written (%s); every start will convert again", e)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _prune_stale_repacks(root: str) -> None:
    """Delete the repacks no load can reach again: a directory whose GGUF is gone or
    was replaced (a new upstream revision), or that an older schema built. Each is
    the size of the model; the small config and tokenizer files stay."""
    for entry in os.scandir(root):
        weights = os.path.join(entry.path, _REPACK)
        if not os.path.isfile(weights):
            continue
        try:
            with open(os.path.join(entry.path, "config.json")) as f:
                stamp = json.load(f).get(_KEY, {})
            st = os.stat(stamp["file"])
            live = (stamp.get("schema") == _SCHEMA and st.st_size == stamp.get("size")
                    and st.st_mtime_ns == stamp.get("mtime_ns"))
        except (OSError, ValueError, KeyError, TypeError):
            live = False
        if not live:
            os.remove(weights)
            log.info("GGUF repack removed, its file changed or is gone: %s", weights)


def _convert(config: dict) -> "tuple[nn.Module, dict[str, Slot]]":
    """The model built from the GGUF file itself, and the modules it placed."""
    import gguf
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten

    from . import mlx_gguf

    reader = gguf.GGUFReader(config[_KEY]["file"])
    g = read_geometry(reader)
    model = _empty_model(config)
    want = set(dict(tree_flatten(model.parameters())))
    GGUFLinear, GGUFEmbedding = _module_classes()
    dtype = mx.bfloat16
    got: set[str] = set()
    formats: dict[str, int] = {}
    slots: dict[str, Slot] = {}

    def assign(path: str, value: "nn.Module | mx.array") -> None:
        _assign(model, path, value)

    for t in reader.tensors:
        p = place(t.name, g)
        if p is None:
            continue
        raw = np.asarray(t.data)
        qtype = int(t.tensor_type)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            # gguf-py hands BF16 back as raw bytes, twice the element count: read as
            # floats they would be garbage of the wrong shape, not an error.
            raise ValueError(f"{t.name}: BF16 tensors are not supported")
        quantized = t.tensor_type not in (gguf.GGMLQuantizationType.F32,
                                          gguf.GGMLQuantizationType.F16)
        perm = row_order(p, g, raw.shape[0])
        if perm is not None:
            raw = raw[perm]
        if p.kind in ("linear", "embedding"):
            if not quantized:
                raise ValueError(f"{t.name}: float projection {t.tensor_type.name}")
            if qtype not in mlx_gguf.FORMATS:
                raise ValueError(f"{t.name}: unsupported GGUF format {t.tensor_type.name}")
            in_dims = int(t.shape[0])
            w = mx.array(raw)
            if p.kind == "linear" and raw.shape[0] < _DENSE_ROWS:
                # The GDN's per-head gate projections are a few dozen rows: too few
                # to occupy the fused kernel, and a few hundred KB each as bf16.
                lin = nn.Linear(in_dims, raw.shape[0], bias=False)
                dense = mlx_gguf.dequantize(w, qtype, in_dims, dtype)
                if p.col_gather:        # dense: permute the columns themselves
                    dense = dense[:, mx.array(v_head_order(g, g.v_head_dim))]
                lin.weight = dense
                assign(p.path, lin)
                slots[p.path] = Slot("dense", qtype, in_dims, raw.shape[0])
                got.add(p.path + ".weight")
                continue
            slots[p.path] = Slot(p.kind, qtype, in_dims, raw.shape[0])
            if p.kind == "embedding":
                assign(p.path, GGUFEmbedding(w, qtype, in_dims, dtype))
            else:
                gather = (mx.array(out_proj_gather(g).astype(np.uint32))
                          if p.col_gather else None)
                assign(p.path, GGUFLinear(w, qtype, in_dims, gather))
            got.add(p.path + ".weight")
            name = t.tensor_type.name
            formats[name] = formats.get(name, 0) + 1
            mx.eval(w)
            continue
        if quantized:
            raise ValueError(f"{t.name}: quantized {p.kind} tensor")
        v = raw.astype(np.float32)
        if p.kind == "a_log":
            arr = mx.array(np.log(-v))
            key = p.path
        else:
            arr = mx.array(v[:, :, None] if p.kind == "conv" else v).astype(dtype)
            key = p.path + ".weight" if p.kind in ("vector", "conv") \
                and not p.path.endswith("dt_bias") else p.path
        assign(key, arr)
        got.add(key)
    missing = want - got
    if missing:
        raise ValueError(f"GGUF left {len(missing)} parameter(s) unset, e.g. "
                         f"{sorted(missing)[:3]}")
    model.eval()
    mx.eval(model.parameters())
    log.info("GGUF loaded: %s, %d layers, formats %s", config[_KEY]["file"],
             g.n_layers, ", ".join(f"{k}x{v}" for k, v in sorted(formats.items())))
    return model, slots
