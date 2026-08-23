"""Same model, same Mac, stock engines — the numbers behind the README comparison table.

One question: what does chad's engine buy over pointing a generic local-model tool at the
SAME weights on the SAME laptop? Two arms, each measured with its own native benchmark
on a ~512-token prompt and a 128-token generation, one engine resident at a time:

  llama    stock llama.cpp via `llama-bench` (pp512 / tg128, its defaults)
  chad     `chad-bench --prefill-tokens 512 --gen-tokens 128`, once serially
           (CHAD_NO_DFLASH=1) and once with the default DFlash2 block drafter

The GGUF is Unsloth's `Qwen3.8-27B-UD-Q3_K_XL` — the same dynamic-quant recipe chad's MLX
checkpoint is named after — so the weights on each side are as close as two formats get.
Decode is memory-bandwidth bound, so serial decode should land in the same place on every
engine; the gap is the drafter.

Run one arm at a time on a 24 GB box. Each arm loads ~13 GB, and a second resident engine
pushes the run into swap (and has panicked this machine on load/teardown cycling):

    uv run python benchmarks/stock/stock.py llama
    uv run python benchmarks/stock/stock.py chad
    uv run python benchmarks/stock/stock.py table      # print the collected rows

Ollama is not a separate arm: it runs llama.cpp's engine underneath (no speculative
decoding on this model). `_runs/ollama.json` is one hand-run measurement on the same GGUF
(FROM-only Modelfile, num_ctx 2048, temperature 0, /api/generate counters) and matches the
llama.cpp decode number; importing a GGUF into Ollama needs ~45 GB of scratch disk, so it
is not scripted.

Requirements: `brew install llama.cpp`, and the GGUF (the script downloads it into the
shared Hugging Face cache on first use, ~13 GB; `STOCK_GGUF=<path>` points at one you
already have). Results accumulate in
`benchmarks/stock/_runs/<arm>.json` — committed, as the record behind the table; `table`
renders them as markdown.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "_runs")
GGUF_REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF_FILE = "Qwen3.8-27B-UD-Q3_K_XL.gguf"
PROMPT_TOKENS = 512
GEN_TOKENS = 128
REPS = 3

# The same filler chad-bench tiles, so every arm reads comparable text. Serial decode is
# content-independent (bandwidth); what the drafter accepts is not, which is why the
# chad arm also reports the real-context numbers from benchmarks/spec_decode.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))


def _gguf() -> str:
    """The GGUF path: `STOCK_GGUF=<path>` if you already have the file, else the shared
    Hugging Face cache (downloaded on first use)."""
    local = os.environ.get("STOCK_GGUF")
    if local:
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(GGUF_REPO, GGUF_FILE)


def _save(arm: str, row: dict) -> None:
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, f"{arm}.json"), "w") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))


def _need(binary: str, hint: str) -> str:
    p = shutil.which(binary)
    if not p:
        sys.exit(f"{binary} not on PATH — {hint}")
    return p


# --------------------------------------------------------------------------- llama.cpp
def arm_llama() -> None:
    bench = _need("llama-bench", "brew install llama.cpp")
    gguf = _gguf()
    cmd = [bench, "-m", gguf, "-p", str(PROMPT_TOKENS), "-n", str(GEN_TOKENS),
           "-r", str(REPS), "-o", "json"]
    print(" ".join(cmd), flush=True)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    rows = json.loads(out)
    pp = next(r for r in rows if r["n_prompt"] == PROMPT_TOKENS and r["n_gen"] == 0)
    tg = next(r for r in rows if r["n_gen"] == GEN_TOKENS and r["n_prompt"] == 0)
    _save("llama", {
        "engine": "llama.cpp",
        "version": f"build {pp.get('build_number', '?')} ({pp.get('build_commit', '?')})",
        "model": GGUF_FILE, "prompt_tokens": PROMPT_TOKENS, "gen_tokens": GEN_TOKENS,
        "prefill_tps": round(pp["avg_ts"], 1), "decode_tps": round(tg["avg_ts"], 1),
        "reps": REPS, "flags": "llama-bench defaults (all layers on Metal, flash-attn auto)",
    })


# -------------------------------------------------------------------------------- chad
def _chad_bench(env: dict) -> dict:
    cmd = [sys.executable, "-m", "chad.bench", "--prefill-tokens", str(PROMPT_TOKENS),
           "--gen-tokens", str(GEN_TOKENS)]
    print(" ".join(f"{k}={v}" for k, v in env.items()), " ".join(cmd), flush=True)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                         env={**os.environ, **env}).stdout
    print(out)
    got: dict = {}
    for line in out.splitlines():
        body = re.sub(r"^\s*\d\.\s+(?=[a-z])", "", line.lower())   # "1. prefill (cold) ..." -> "prefill (cold) ..."
        m = re.search(r"([\d.]+)\s*tok/s", body)
        if m and body.startswith("prefill (cold)"):
            got["prefill_tps"] = float(m.group(1))
        elif m and body.startswith("decode"):
            got["decode_tps"] = float(m.group(1))
        elif body.startswith("warm step"):
            got["warm_step_new_tokens"] = int(re.search(r"(\d+) new tok", body).group(1))
        elif "s of prefill for the follow-up turn" in body:
            got["warm_step_prefill_s"] = float(re.search(r"([\d.]+) s of prefill", body).group(1))
    if "prefill_tps" not in got or "decode_tps" not in got:
        sys.exit("could not parse chad-bench output")
    return got


def arm_chad() -> None:
    import chad
    serial = _chad_bench({"CHAD_NO_DFLASH": "1"})
    time.sleep(10)   # let Metal release the first load before the second
    default = _chad_bench({})
    _save("chad", {
        "engine": "chad", "version": chad.__version__,
        "model": "nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX",
        "prompt_tokens": PROMPT_TOKENS, "gen_tokens": GEN_TOKENS,
        "serial": {**serial, "flags": "CHAD_NO_DFLASH=1"},
        "default": {**default, "flags": "DFlash2 block drafter, per-round width schedule"},
    })


# ------------------------------------------------------------------------------- table
def table() -> None:
    def load(arm):
        p = os.path.join(RUNS, f"{arm}.json")
        return json.load(open(p)) if os.path.exists(p) else None
    ll, ch = load("llama"), load("chad")
    rows = [("Engine", "Prefill (512-tok prompt)", "Decode (128 tok)", "Speculative decoding")]
    if ll:
        rows.append(("llama.cpp `llama-bench` (stock)", f"{ll['prefill_tps']:.0f} tok/s",
                     f"{ll['decode_tps']:.1f} tok/s", "none for this model"))
    if ch:
        rows.append(("**chad**, serial (`CHAD_NO_DFLASH=1`)", f"{ch['serial']['prefill_tps']:.0f} tok/s",
                     f"{ch['serial']['decode_tps']:.1f} tok/s", "off"))
        rows.append(("**chad**, default", f"{ch['default']['prefill_tps']:.0f} tok/s",
                     f"**{ch['default']['decode_tps']:.1f} tok/s**", "DFlash2 block drafter"))
    w = [max(len(r[i]) for r in rows) for i in range(4)]
    for n, r in enumerate(rows):
        print("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")
        if n == 0:
            print("|" + "|".join("-" * (x + 2) for x in w) + "|")


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "table"
    {"llama": arm_llama, "chad": arm_chad, "table": table}[arm]()
