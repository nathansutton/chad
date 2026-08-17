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

import math
import os
from typing import Any, Optional

from .diag import log

_SIDECAR_DIR = os.path.expanduser("~/.cache/chad/mtp")


class DepthPolicy:
    """Cost-model draft schedule: choose the depth that maximizes expected
    committed tokens per unit round time.

    A round of depth d costs T(d) = V + d*H (one width-(d+1) verify plus d
    head steps) and commits E[tokens](d) = 1 + sum_{k=1..d} prod_{i<k} p_i,
    where p_i is the estimated acceptance of draft position i GIVEN the
    prefix before it was accepted. Greedy marginal rule: extend to position
    d+1 exactly while

        prod_{i<=d+1} p_i  >  h * (1 + E_d) / (1 + d*h)

    with h = H/V the head step's cost relative to the verify forward. The
    p_i are per-position EMAs seeded with an optimistic, gently decaying
    prior (the first rounds should draft, not stall); a fully accepted round
    also transfers bounded optimism to the first unreached position, so deep
    depths become reachable inside a short window. On hot prose this runs
    to the cap; on cold prompts it collapses to 1 and then to 0 (an adaptive
    skip costs exactly what a serial step costs).

    The pending top-2 logit margin (of the row that produced the pending
    token) additionally caps p_0/p_1: a near-tie next token is exactly where
    the head is about to be wrong, whatever its recent streak says.
    """

    MAX_DEPTH = 8

    def __init__(self, max_depth: int, h: float, costs: Optional[list] = None):
        """`costs`, when given, is the measured round cost T(d) for depth
        d = 0..max_depth (any unit): T(d) = V(d+1) + d*H with the verify-width
        ladder as measured on THIS stack, which is not width-flat (the S>1
        forward pays nearly per-row above S=2 outside the fused envelopes).
        The scalar-h model is the width-flat special case T(d) = 1 + d*h."""
        self.max_depth = max(0, min(max_depth, self.MAX_DEPTH))
        self.h = h
        if costs is not None and len(costs) >= self.max_depth + 1:
            self.costs = [float(c) for c in costs]
        else:
            self.costs = [1.0 + d * h for d in range(self.MAX_DEPTH + 1)]
        self.ema = [0.85 * (0.98 ** i) for i in range(self.MAX_DEPTH)]
        self.alpha = 0.15
        self.margin: Optional[float] = None   # pending top-2 logit margin
        self.streak = 0                       # consecutive fully-accepted rounds
        self._skips = 0                       # consecutive adaptive skips

    def depth(self) -> int:
        cap = self.max_depth
        if cap <= 0:
            return 0
        reach, expected, d = 1.0, 0.0, 0
        while d < cap:
            p = self.ema[d]
            if self.margin is not None:
                # Confidence gates on the first two positions only; deeper
                # positions are governed by the EMAs' chained product.
                if d == 0:
                    p = min(p, 1.0 / (1.0 + math.exp(-self.margin / 2.0)))
                elif d == 1:
                    p = min(p, 1.0 / (1.0 + math.exp(-self.margin / 3.0)))
            reach *= p
            # Extend to d+1 iff tokens-per-time improves:
            #   (1+E+reach)/T(d+1) > (1+E)/T(d)
            if reach * self.costs[d] <= \
                    (1.0 + expected) * (self.costs[d + 1] - self.costs[d]):
                break
            expected += reach
            d += 1
        if d == 0:
            # An adaptive skip is free, but the EMAs only observe drafted
            # rounds — a hard stretch would otherwise lock drafting out for
            # the rest of the turn even after the content turns easy. Probe
            # depth 1 every 16th consecutive skip (~one EMA half-life).
            self._skips += 1
            if self._skips >= 16:
                self._skips = 0
                return 1
        else:
            self._skips = 0
        return d

    def record(self, proposed: int, accepted: int, stopped_early: bool) -> None:
        """Fold one round's outcome into the per-position EMAs. Positions
        before `accepted` observed a success; the position AT `accepted`
        observed a failure only when the walk actually rejected there (not
        when the round ended on a committed stop token); deeper positions
        observe nothing. A FULLY accepted round transfers bounded optimism
        (toward 0.95, never past it) to the first unreached position."""
        a = self.alpha
        e = self.ema
        for i in range(min(accepted, len(e))):
            e[i] += a * (1.0 - e[i])
        if accepted < proposed and not stopped_early and accepted < len(e):
            e[accepted] += a * (0.0 - e[accepted])
        elif accepted == proposed and proposed > 0 and accepted < len(e):
            if e[accepted] < 0.95:
                e[accepted] += a * (0.95 - e[accepted])
        self.streak = self.streak + 1 if (proposed > 0 and accepted == proposed) \
            else 0


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
        islands = {k: v for k, v in weights.items()
                   if k.startswith("precision_islands.")}
        if islands:
            weights = {k: v for k, v in weights.items()
                       if not k.startswith("precision_islands.")}
        meta_gs, meta_bits = _read_meta(path)
        if not meta_bits and "fc.scales" in weights:
            # Foreign sidecar (e.g. an arena head artifact) without build
            # metadata: infer the packing from fc's shapes. fc is (H, 2H):
            # 2H / packed-cols u32 values per word -> bits, 2H / scale-cols
            # -> group size.
            in_dim = 2 * int(args.hidden_size)
            meta_bits = 32 // (in_dim // int(weights["fc.weight"].shape[1]))
            meta_gs = in_dim // int(weights["fc.scales"].shape[1])
        head = build(args)
        if meta_bits:
            nn.quantize(head, group_size=meta_gs, bits=meta_bits)
        head.load_weights(list(weights.items()))
        n_isl = _install_islands(head, islands) if islands else 0
        head.eval()
        mx.eval(head.parameters())
        log.info("MTP head loaded (%s, %s%s)", path,
                 f"{meta_bits}-bit g{meta_gs}" if meta_bits else "unquantized",
                 f", {n_isl} bf16 precision-island rows" if n_isl else "")
        return head
    except Exception as e:  # noqa: BLE001 — never break model load over MTP
        log.warning("MTP head load failed (%s); decoding without MTP", e)
        return None


def _install_islands(head: Any, islands: dict) -> int:
    """Wrap the head's q/k/v projections with exact-row precision islands.

    A sidecar may carry `precision_islands.{q,k,v}.{weight,indices}`: selected
    output rows of the proposal head's attention projections kept in bf16
    (the rows with the largest fp32 reconstruction error under the packed
    4-bit quantization — a draft-quality recovery, worth ~real acceptance at
    depth). The wrapped projection runs the packed QMV unchanged, computes the
    island rows exactly, and overwrites just those outputs. Head-only: the
    main model never sees these arrays. Returns the number of island rows."""
    import mlx.core as mx
    import mlx.nn as nn

    class _IslandLinear(nn.Module):
        def __init__(self, inner, w, idx):
            super().__init__()
            self.inner = inner
            self.island_w = w
            self.island_idx = idx.astype(mx.int32)

        def __call__(self, x):
            y = self.inner(x)
            w = self.island_w
            if w.dtype != x.dtype:
                # One-time cast to the compute dtype (a per-call astype in the
                # matmul would re-launch the cast kernel every draft step).
                w = w.astype(x.dtype)
                self.island_w = w
            exact = x @ w.T
            idx = self.island_idx.reshape((1,) * (y.ndim - 1) + (-1,))
            return mx.put_along_axis(y, idx, exact, axis=-1)

    attn = head.layers[0].self_attn
    total = 0
    for name in ("q", "k", "v"):
        w = islands.get(f"precision_islands.{name}.weight")
        idx = islands.get(f"precision_islands.{name}.indices")
        if w is None or idx is None:
            continue
        if w.shape[0] != idx.shape[0]:
            raise ValueError(
                f"precision island {name}: weight rows {w.shape[0]} != "
                f"indices {idx.shape[0]}")
        proj = f"{name}_proj"
        setattr(attn, proj, _IslandLinear(getattr(attn, proj), w, idx))
        total += int(w.shape[0])
    return total


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
