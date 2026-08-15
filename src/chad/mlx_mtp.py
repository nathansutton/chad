"""Qwen3.8-class MTP (multi-token-prediction) head: sidecar build + load + forward.

The Qwen3.8 checkpoints ship a 1-layer MTP head — `fc` over
[norm(embedding), norm(hidden)] + one full-attention DecoderLayer + a final
norm, sharing the main model's embeddings and lm_head — that mainstream
loaders (mlx-lm, transformers) STRIP on load. This module rebuilds it as a
first-class mlx module so the engine can self-speculate: draft k tokens with
the cheap head, verify them in one batched main-model forward.

Semantics follow the serving-engine reference implementations: the head's
`hidden` input is the main model's POST-final-norm hidden state (exactly what
mlx-lm's `Qwen3_5TextModel.__call__` returns), and `pre_fc_norm_hidden` is
applied on top — the double normalization is how the head was trained. Draft
chaining feeds the head's own output hidden back as the next step's `hidden`.

The head is attention-only (plain KVCache — natively trimmable), so
speculative rollback needs no recurrent snapshots on the MTP side; only the
main model's hybrid cache needs the engine's snapshot/trim machinery.

Weights are not part of the quantized text-only conversions, so they load
from a SIDECAR safetensors file, resolved in order:
  1. $CHAD_MTP_PATH (explicit file)
  2. <model_dir>/mtp.safetensors (a conversion that kept the head)
  3. ~/.cache/chad/mtp/<model-dir-basename>.safetensors (built locally)
Build one from an original-checkpoint shard holding the `mtp.*` tensors:
  uv run python -m chad.mlx_mtp <shard.safetensors> --model <model_dir>
"""

import os
from typing import Any, Optional

from .diag import log

_SIDECAR_DIR = os.path.expanduser("~/.cache/chad/mtp")


def build(args: Any):
    """Construct an (unquantized) MTP head module for a qwen3_5 TextModelArgs."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import DecoderLayer

    class _MTPHead(nn.Module):
        def __init__(self, args):
            super().__init__()
            h, eps = args.hidden_size, args.rms_norm_eps
            self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=eps)
            self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=eps)
            self.fc = nn.Linear(2 * h, h, bias=False)
            # layer_idx chosen so DecoderLayer builds its FULL-attention form
            # ((idx+1) % full_attention_interval == 0); the head has no GDN.
            self.layers = [DecoderLayer(args, args.full_attention_interval - 1)]
            self.norm = nn.RMSNorm(h, eps=eps)

        def __call__(self, embeds, hidden, cache=None, mask=None):
            x = mx.concatenate(
                [self.pre_fc_norm_embedding(embeds),
                 self.pre_fc_norm_hidden(hidden)], axis=-1)
            x = self.fc(x)
            x = self.layers[0](x, mask=mask, cache=cache)
            return self.norm(x)

        def make_cache(self):
            from mlx_lm.models.cache import KVCache
            return KVCache()

    return _MTPHead(args)


def _sidecar_candidates(model_dir: str) -> list:
    out = []
    env = os.environ.get("CHAD_MTP_PATH")
    if env:
        out.append(os.path.expanduser(env))
    out.append(os.path.join(model_dir, "mtp.safetensors"))
    out.append(os.path.join(
        _SIDECAR_DIR, os.path.basename(os.path.normpath(model_dir)) + ".safetensors"))
    return out


def _mtp_layers(model_dir: str) -> int:
    """mtp_num_hidden_layers from the on-disk config (0 when absent/unreadable)."""
    import json
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        return int(tc.get("mtp_num_hidden_layers") or 0)
    except Exception:  # noqa: BLE001
        return 0


def _quant_defaults(model_dir: str) -> tuple:
    """(group_size, bits) the base conversion used, so the head matches."""
    import json
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            q = json.load(f).get("quantization") or {}
        return int(q.get("group_size", 64)), int(q.get("bits", 4))
    except Exception:  # noqa: BLE001 — sane default beats a failed load
        return 64, 4


def load_head(model: Any, model_dir: str) -> Optional[Any]:
    """Load the MTP head for a loaded qwen3_5 model, or None (no sidecar found,
    wrong arch, or any failure — MTP is a pure speed feature, never load-bearing)."""
    try:
        import mlx.core as mx
        import mlx.nn as nn

        args = getattr(getattr(model, "language_model", None), "args", None)
        if args is None or _mtp_layers(model_dir) != 1:
            # mlx-lm's TextModelArgs drops unknown config keys, so the head
            # count must come from the on-disk config, not the parsed args.
            return None
        path = next((p for p in _sidecar_candidates(model_dir)
                     if p and os.path.isfile(p)), None)
        if path is None:
            return None
        weights = mx.load(path)
        meta_gs, meta_bits = _read_meta(path)
        head = build(args)
        if meta_bits:
            nn.quantize(head, group_size=meta_gs, bits=meta_bits)
        head.load_weights(list(weights.items()))
        head.eval()
        mx.eval(head.parameters())
        log.info("MTP head loaded (%s, %s)", path,
                 f"{meta_bits}-bit g{meta_gs}" if meta_bits else "unquantized")
        return head
    except Exception as e:  # noqa: BLE001 — never break model load over MTP
        log.warning("MTP head load failed (%s); decoding without MTP", e)
        return None


def _read_meta(path: str) -> tuple:
    """(group_size, bits) recorded at build time; (0, 0) = unquantized."""
    import json
    try:
        # safetensors header: 8-byte little-endian length + JSON with __metadata__
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        meta = hdr.get("__metadata__") or {}
        return int(meta.get("group_size", 0)), int(meta.get("bits", 0))
    except Exception:  # noqa: BLE001
        return 0, 0


def build_sidecar(shard_path: str, model_dir: str, out_path: Optional[str] = None,
                  bits: Optional[int] = None, group_size: Optional[int] = None) -> str:
    """Extract `mtp.*` tensors from an original-checkpoint shard, quantize them
    to match the base conversion, and save the sidecar. Returns the output path."""
    import json

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    from mlx_lm.models.qwen3_5 import ModelArgs, TextModelArgs
    targs = TextModelArgs.from_dict(
        ModelArgs.from_dict(cfg).text_config)
    gs0, bits0 = _quant_defaults(model_dir)
    gs, b = group_size or gs0, bits or bits0

    raw = mx.load(shard_path)
    mtp = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
    if not mtp:
        raise ValueError(f"no mtp.* tensors in {shard_path}")
    # Original Qwen3.5-family checkpoints store RMSNorm weights ZERO-CENTERED
    # (gamma = 1 + w); mlx's RMSNorm is plain-gamma. mlx-lm shifts the main
    # model's norms at conversion (qwen3_5.TextModel.sanitize) — the head's
    # norms need the same +1.0 or every normalization in it is off by ~1
    # (measured: teacher-forced acceptance 0% unshifted, 73% shifted).
    norm_sfx = ("input_layernorm.weight", "post_attention_layernorm.weight",
                "q_norm.weight", "k_norm.weight")
    mtp = {k: (v + 1.0
               if (k.endswith(norm_sfx)
                   or k in ("norm.weight", "pre_fc_norm_embedding.weight",
                            "pre_fc_norm_hidden.weight"))
               else v)
           for k, v in mtp.items()}
    head = build(targs)
    head.load_weights(list(mtp.items()))
    nn.quantize(head, group_size=gs, bits=b)
    mx.eval(head.parameters())

    if out_path is None:
        os.makedirs(_SIDECAR_DIR, exist_ok=True)
        out_path = os.path.join(
            _SIDECAR_DIR,
            os.path.basename(os.path.normpath(model_dir)) + ".safetensors")
    flat = dict(tree_flatten(head.parameters()))
    mx.save_safetensors(out_path, flat,
                        metadata={"group_size": str(gs), "bits": str(b)})
    return out_path


def main(argv=None):  # pragma: no cover - thin CLI
    import argparse
    ap = argparse.ArgumentParser(
        description="Build a quantized MTP sidecar from an original shard")
    ap.add_argument("shard", help="safetensors shard containing mtp.* (bf16)")
    ap.add_argument("--model", required=True,
                    help="base (quantized) model dir the sidecar pairs with")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    a = ap.parse_args(argv)
    out = build_sidecar(a.shard, a.model, a.out, a.bits, a.group_size)
    print(out)


if __name__ == "__main__":  # pragma: no cover
    main()
