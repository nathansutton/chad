#!/usr/bin/env python3
"""Build the 6-bit MLX quant of Ornith-1.0-35B — the higher-bit TB2 arm for Macs
with 48 GB+ unified memory (e.g. an EC2 mac-m4pro.metal host).

Output: ~28.5 GB of weights (6-bit group-64 affine everywhere, router gates pinned
to 8-bit by mlx_lm's own qwen3_5 quant predicate, A_log kept fp32). Against a
48 GB Mac's ~36 GB Metal working-set budget that leaves room for a 120k-token
window with margin. This is the same quant CLASS as the llama.cpp Q6_K reference
arm, so scores are comparable-in-kind — but per the fidelity notes in README.md it
is still a DIFFERENT arm from the shipped 2-bit quant: score it, label it.

The one non-obvious part is the expert-stacking shim: the released bf16 checkpoint
stores 256 per-expert tensors per layer (`mlp.experts.<i>.gate_proj.weight`), while
mlx_lm >= 0.31.3 only stacks the fused-HF format (`mlp.experts.gate_up_proj`) and
raises "Received N parameters not in model" on the per-expert layout. The shim
stacks them into the SwitchLinear layout before the usual pipeline; it is lazy-safe
(builds graph nodes, no materialization until save).

Usage (from the repo root; downloads ~65 GB of bf16 on first run):
    uv run python benchmarks/tb2/quantize_q6.py
    uv run python benchmarks/tb2/quantize_q6.py --dry-run     # predict size only
    uv run python benchmarks/tb2/quantize_q6.py --src /path/to/local/bf16
"""
import argparse
import re
from pathlib import Path

DEFAULT_SRC = "deepreinforce-ai/Ornith-1.0-35B"
DEFAULT_OUT = str(Path(__file__).resolve().parent / "models" / "Ornith-1.0-35B-q6-mlx")
BITS = 6
GROUP = 64


def install_expert_stacking_shim():
    """Stack the checkpoint's per-expert keys into SwitchLinear tensors on load."""
    import mlx.core as mx
    from mlx_lm.models import qwen3_5_moe

    orig = qwen3_5_moe.Model.sanitize
    if getattr(orig, "_expert_stacking_shim", False):
        return

    def sanitize(self, weights):
        weights = orig(self, weights)
        pat = re.compile(
            r"^(?P<prefix>language_model\.model\.layers\.\d+\.mlp)"
            r"\.experts\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
        groups: dict = {}
        for k in weights:
            m = pat.match(k)
            if m:
                groups.setdefault(
                    (m["prefix"], m["proj"]), []).append((int(m["idx"]), k))
        for (prefix, proj), items in groups.items():
            items.sort()
            stacked = mx.stack([weights.pop(k) for _, k in items], axis=0)
            weights[f"{prefix}.switch_mlp.{proj}.weight"] = stacked
        return weights

    sanitize._expert_stacking_shim = True  # type: ignore[attr-defined]
    qwen3_5_moe.Model.sanitize = sanitize


def dry_run(src: str):
    """Load the module tree lazily and print the predicted quantized footprint."""
    from mlx_lm.utils import load

    print(f"[dry-run] loading module tree from {src} (lazy)...")
    model, _ = load(src, lazy=True)
    total = router = 0
    for path, module in model.named_modules():
        if not hasattr(module, "to_quantized") or not hasattr(module, "weight"):
            continue
        if module.weight.shape[-1] % GROUP != 0:
            continue
        n = module.weight.size
        if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
            router += n
        else:
            total += n
    overhead = 32.0 / GROUP  # fp16 scale+bias per group
    gb = (total * (BITS + overhead) + router * (8 + overhead)) / 8 / 1e9
    print(f"[dry-run] {total + router:,} quantized params -> ~{gb:.2f} GB "
          f"({BITS}-bit gs{GROUP}, routers 8-bit)")
    print("[dry-run] no files written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="bf16 source: HF repo id (downloaded on demand) or local dir")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    install_expert_stacking_shim()
    if args.dry_run:
        dry_run(args.src)
        return

    if Path(args.out).exists():
        raise SystemExit(f"out exists: {args.out} (delete it or pick another --out)")
    from mlx_lm.convert import convert
    print(f"[convert] {args.src} -> {args.out}  ({BITS}-bit gs{GROUP}, routers 8-bit)")
    # No custom predicate: mlx_lm's qwen3_5 quant_predicate applies (routers 8-bit)
    # and its cast_predicate keeps A_log fp32.
    convert(hf_path=args.src, mlx_path=args.out,
            quantize=True, q_bits=BITS, q_group_size=GROUP)
    print(f"[done] {args.out} — serve it with benchmarks/tb2/serve_q6.sh")


if __name__ == "__main__":
    main()
