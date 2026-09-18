"""Loader for Prism ML's Hadamard-folded ternary packs (`prism_hadamard_qwen35`).

These packs are a Qwen3.8-27B checkpoint whose projections are stored in a rotated
basis: each weight matrix is multiplied by a fixed sign vector and transformed by a
blockwise Hadamard rotation offline, then quantized to 2-bit affine g128 whose three
levels reproduce the ternary set {-s, 0, +s}. The rotation costs no extra bits and no
extra weight traffic, but it is not optional: the matching transform has to be applied
to the ACTIVATIONS at runtime, and the embedding table needs the inverse transform
after its lookup. mlx-lm's ordinary affine loader would find weights/scales/biases of
exactly the right shapes, skip both transforms, and return plausible-looking garbage
rather than an error — so the pack is routed here on `model_type` instead.

The pack ships its own `runtime/` Python and asks callers to `sys.path.insert` it.
chad does not import code out of a model download; `Packed` below is a reimplementation
of that module's forward, and `load()` refuses any pack whose declared quantization is
not the 2-bit affine g128 container it was written against.

What the loader normalizes, and why:

* The residual stream runs in bf16, like every other checkpoint chad loads. The pack
  stores its scales in fp16 and its norms, conv and gate projections in fp32, and MLX
  promotes a mixed-dtype op to the wider type — so taken as shipped, the first
  RMSNorm turns the stream fp32 and every projection after it runs (and every
  activation lives) at twice the width. Everything floating is cast to bf16 once here,
  exactly what mlx-lm's own loader does to a quantized checkpoint's side tensors.
* The pack carries one sign vector per rotated module, but they are identical for
  every module of one input width (the whole model rotates the residual stream, the
  MLP intermediate and the attention/GDN output each with a single fixed vector). The
  loader shares one array per width, which is what lets the fast-path fuse
  same-input projections (gate|up, qkv|z, q|k|v) behind ONE rotation.
* The matmul goes through `mlx_qmm_mma.qmm`, so the speculative verify widths take
  the small-M MMA kernel exactly as the shipped 3-bit does.

The pack bundles no drafter and declares `mtp: false`; `mlx_dflash` borrows the
Qwen3.8-27B DFlash2 sidecar from a sibling repo (a drafter reads the target's residual
stream, which the quantization only perturbs). Its chat template defaults
`reasoning_effort` to xhigh where the shipped model's defaults to medium — the same
template otherwise — so the engine carries `REASONING_EFFORT_DEFAULT` for it.
"""

import contextlib
import json
import logging
import math
import os
from typing import TYPE_CHECKING, Any, Optional, cast

from . import mlx_qmm_mma
from .config import flag as _flag
from .diag import log

if TYPE_CHECKING:  # mlx is imported lazily inside the functions so the module loads on Linux
    import mlx.core as mx

_MODEL_TYPE = "prism_hadamard_qwen35"
_BLOCKS = (512, 1024, 2048, 4096)
BITS, GROUP_SIZE = 2, 128

# The pack's template is the Qwen3.8 template with `reasoning_effort|default('xhigh')`
# where the shipped model's says medium. Passing medium explicitly renders byte-identically
# to the shipped default, and think-token decode is two thirds of wall on this model.
REASONING_EFFORT_DEFAULT = "medium"


def is_prism_pack(config: dict) -> bool:
    return config.get("model_type") == _MODEL_TYPE


class _NotAModelMismatch(logging.Filter):
    """Drops transformers' "model of type `prism_hadamard_qwen35` to instantiate a model
    of type ``" warning. AutoTokenizer resolves the checkpoint's config to pick a
    tokenizer class, finds no transformers model for the pack's model_type, falls back
    to the base config and warns that the MODEL may not load. chad never instantiates
    a transformers model (prism_pack.load builds the MLX model itself), so the warning
    is about something that does not happen — and it printed across the banner."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _MODEL_TYPE not in record.getMessage()


@contextlib.contextmanager
def quiet_tokenizer_load():
    """Scope for loading a Prism pack's tokenizer without the model-type mismatch
    warning above; every other transformers message still comes through."""
    logger = logging.getLogger("transformers.configuration_utils")
    flt = _NotAModelMismatch()
    logger.addFilter(flt)
    try:
        yield
    finally:
        logger.removeFilter(flt)


def rotate(x, block: int, signs=None, inverse: bool = False):
    """Apply the pack's blockwise Hadamard rotation to activations, in x's dtype.

    `signs` is the module's ±1 vector, multiplied in before the transform (after it
    when `inverse`); None means the caller already folded it into whatever produced
    x (an RMSNorm weight, a projection's rows), and the rotation is the transform
    alone: ONE kernel. With the vector it is two.

    The transform runs in the activation dtype. The pack's reference runtime casts to
    fp32 around it (four kernels, fp32-width temporaries), which guards nothing
    measurable on this model: the ±1 multiply is exact at any width, and teacher-forced
    code NLL over 7k tokens of three files moved by at most 0.002, in no consistent
    direction (1.3830 / 1.6812 / 1.3576 native against 1.3847 / 1.6827 / 1.3573), with
    prefill 1-3% faster. CHAD_PRISM_ROT_FP32 restores the fp32 transform everywhere,
    as the A/B arm."""
    import mlx.core as mx

    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError(f"Hadamard block {block} does not divide width {shape[-1]}")
    fp32 = _flag("CHAD_PRISM_ROT_FP32")
    if fp32:
        x = x.astype(mx.float32)
    if signs is not None:
        signs = signs.astype(x.dtype)       # a no-op for a pack loaded by load()
        if not inverse:
            x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block),
                              scale=1 / math.sqrt(block)).reshape(shape)
    if signs is not None and inverse:
        x = x * signs
    return x.astype(dtype) if fp32 else x


_PACKED: Optional[type] = None


def _packed_class():
    global _PACKED
    if _PACKED is not None:
        return _PACKED
    import mlx.core as mx
    import mlx.nn as nn

    class Packed(nn.Module):
        """One rotated projection (or the rotated embedding table).

        Stands in for `nn.QuantizedLinear` / `nn.Embedding`. `block == 0` means the
        tensor was stored unrotated and this is a plain 2-bit affine matmul."""

        # chad's fused-projection paths and the MMA probe read these off
        # QuantizedLinear; the same names make a Packed eligible for both.
        bits, group_size, mode = BITS, GROUP_SIZE, "affine"

        def __init__(self, weight, scales, biases, block, signs, embedding, dtype):
            super().__init__()
            self.weight, self.scales, self.biases = weight, scales, biases
            if signs is not None:
                self.signs = signs
            self.block, self.embedding, self.dtype = block, embedding, dtype
            # Set by the fast-path once it has folded this module's sign vector into
            # whatever feeds it; the rotation is then the transform alone.
            self.signs_folded = False

        def __call__(self, x):
            if self.embedding:
                shape = x.shape
                out = mx.dequantize(
                    self.weight[x.reshape(-1)], self.scales[x.reshape(-1)],
                    self.biases[x.reshape(-1)], group_size=GROUP_SIZE, bits=BITS,
                ).reshape(*shape, -1).astype(self.dtype)
                return rotate(out, self.block, self.signs, True) if self.block else out
            if self.block:
                x = rotate(x, self.block, None if self.signs_folded else self.signs)
            return mlx_qmm_mma.qmm(x, self.weight, self.scales, self.biases,
                                   GROUP_SIZE, BITS)

    _PACKED = Packed
    return Packed


def is_packed(module) -> bool:
    return _PACKED is not None and isinstance(module, _PACKED)


def load(model_path: str, config: Optional[dict] = None) -> tuple[Any, dict]:
    """Build the qwen3_5 model these packs carry, with rotated projections in place.

    Returns what `mlx_lm.utils.load_model` returns: (model, config). The pack bundles
    an FP16 vision tower chad has no use for; only the `language_model.` half is read,
    which is also 0.92 GB less resident memory."""
    import mlx.core as mx
    from mlx_lm.models import qwen3_5 as q35

    if config is None:
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
    if not is_prism_pack(config):
        raise ValueError(f"not a Prism pack: model_type={config.get('model_type')!r}")
    quant = config.get("quantization") or {}
    if (quant.get("bits"), quant.get("group_size"), quant.get("mode")) \
            != (BITS, GROUP_SIZE, "affine"):
        raise ValueError(f"unsupported Prism container {quant!r}; this loader "
                         f"implements {BITS}-bit affine g{GROUP_SIZE} only")

    model = q35.Model(q35.ModelArgs(model_type="qwen3_5",
                                    text_config=config["text_config"]))
    # SAFETY: mx.load's return type widens across .npy/.npz, but it returns a
    # name -> array mapping for every .safetensors input, which is the only
    # extension this loader ever passes it.
    loaded = cast(dict, mx.load(os.path.join(model_path, "model.safetensors")))
    weights = {k: v for k, v in loaded.items() if k.startswith("language_model.")}
    del loaded
    dtype = mx.bfloat16
    # Sign vectors too: they are ±1, exact at any width, and a wider one would promote
    # the activation it multiplies.
    for k, v in weights.items():
        if v.dtype != mx.uint32:
            weights[k] = v.astype(dtype)
    Packed = _packed_class()
    seen: set[str] = set()
    # One array per DISTINCT sign vector, keyed on content. Identity is what the
    # fast-path's fusion check reads, so it must not depend on the order the pack lists
    # its modules in: keyed on width alone, a first-seen vector that differs from the
    # projections' (the embedding's, say) would leave every later module of that width
    # unshared and the whole fast-path declined.
    sign_keys = [k for k in weights if k.endswith(".signs")]
    mx.eval([weights[k] for k in sign_keys])          # one sync, not one per module
    shared: dict[bytes, "mx.array"] = {}
    for k in sign_keys:
        v = weights[k]
        weights[k] = shared.setdefault(bytes(memoryview(v.view(mx.uint16))), v)
    for record in config["modules"]:
        path = record["path"]
        if path in seen:
            raise ValueError(f"duplicate packed module {path}")
        seen.add(path)
        block = record["block"]
        if block and block not in _BLOCKS:
            raise ValueError(f"unsupported Hadamard block {block}")
        key = "language_model." + path
        signs = weights.get(key + ".signs")
        if block and signs is None:
            raise ValueError(f"{path}: rotated module with no sign vector")
        # mlx Modules are dicts, so the pack's dotted paths walk by subscript; a
        # numeric segment indexes the plain list `model.layers` is.
        parts = key.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else parent[part]
        parent[parts[-1]] = Packed(
            weights[key + ".weight"], weights[key + ".scales"],
            weights[key + ".biases"], block, signs, record["embedding"], dtype)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    log.info("Prism pack loaded: %d rotated modules, %d-bit affine g%d, bf16 stream, "
             "%d shared sign vector(s), vision tower skipped",
             len(seen), BITS, GROUP_SIZE, len(shared))
    return model, config
