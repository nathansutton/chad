"""Same model, same Mac, stock engines — the numbers behind the README comparison table.

One question: what does chad's engine buy over pointing a generic local-model tool at the
SAME weights on the SAME laptop? Two arms, each measured with its own native benchmark
on a ~512-token prompt and a 128-token generation, one engine resident at a time:

  llama         stock llama.cpp via `llama-bench` (pp512 / tg128, its defaults)
  llama-dflash  llama.cpp's own DFlash2 path (build 10658 and later): `llama-server` on the
                same GGUF, once serially and once with the DFlash2 drafter in GGUF form
                (`-md … --spec-type draft-dflash`), greedy, same 512-token filler prompt
  chad          `chad-bench --prefill-tokens 512 --gen-tokens 128`, once serially
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

Ollama is not a separate arm: it runs llama.cpp's engine underneath, and was measured
without speculative decoding (llama.cpp's DFlash2 support says nothing about Ollama's).
`_runs/ollama.json` is one hand-run measurement on the same GGUF
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
# The DFlash2 drafter in GGUF form; Q4_K_M (1.1 GB) is the width of chad's 4-bit sidecar.
DRAFT_REPO = "incoai/Qwen3.8-27B-DFlash2-GGUF"
DRAFT_FILE = "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
DRAFT_N_MAX = 7          # chad's default draft width (Engine.dflash_num_draft)
VERIFY_WIDTHS = "1,2,4,8,16"
SERVER_PORT = 8093
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


# ------------------------------------------------------------------ llama.cpp + DFlash2
def _llama_bin(name: str) -> str:
    """`STOCK_LLAMA_BIN=<dir>` runs a specific build (e.g. an unpacked release tarball)
    without replacing the brew install other benchmarks are pinned to."""
    d = os.environ.get("STOCK_LLAMA_BIN")
    return os.path.join(d, name) if d else _need(name, "brew install llama.cpp")


def _post(path: str, body: dict, timeout: float = 600) -> dict:
    import urllib.request
    req = urllib.request.Request(f"http://127.0.0.1:{SERVER_PORT}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_healthy(proc: subprocess.Popen, timeout: float = 900) -> None:
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with {proc.returncode} while loading")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    sys.exit("llama-server did not come up")


def _server_run(extra: list, log_name: str) -> dict:
    """One llama-server load: tile chad-bench's filler to exactly PROMPT_TOKENS ids with the
    server's own tokenizer, then REPS greedy PROMPT_TOKENS -> GEN_TOKENS completions with
    the prompt cache off, so every rep pays the same prefill. Rates are the server's
    `timings`, which exclude load and HTTP."""
    from chad.bench import _FILLER
    cmd = [_llama_bin("llama-server"), "-m", _gguf(), "-c", "4096", "-np", "1",
           "--host", "127.0.0.1", "--port", str(SERVER_PORT), *extra]
    print(" ".join(cmd), flush=True)
    import tempfile
    log_path = os.path.join(tempfile.gettempdir(), log_name)   # server logs are not the record
    print(f"server log: {log_path}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        _wait_healthy(proc)
        chunk = _post("/tokenize", {"content": _FILLER, "add_special": False})["tokens"]
        ids = (chunk * (PROMPT_TOKENS // len(chunk) + 1))[:PROMPT_TOKENS]
        req = {"prompt": ids, "n_predict": GEN_TOKENS, "temperature": 0.0,
               "cache_prompt": False, "seed": 0}
        _post("/completion", {**req, "n_predict": 16})          # warmup, not recorded
        reps = []
        for _ in range(REPS):
            t = _post("/completion", req)["timings"]
            reps.append({k: t[k] for k in ("prompt_n", "prompt_per_second", "predicted_n",
                                           "predicted_per_second", "draft_n",
                                           "draft_n_accepted") if k in t})
            print(json.dumps(reps[-1]), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    row = {"prefill_tps": round(sum(r["prompt_per_second"] for r in reps) / len(reps), 1),
           "decode_tps": round(sum(r["predicted_per_second"] for r in reps) / len(reps), 1),
           "generated": [r["predicted_n"] for r in reps], "reps": reps}
    drafted = sum(r.get("draft_n", 0) for r in reps)
    if drafted:
        row["acceptance"] = round(sum(r.get("draft_n_accepted", 0) for r in reps) / drafted, 3)
    return row


def arm_llama_dflash() -> None:
    """llama.cpp's own DFlash2 path (b10658+), serial and drafted on one build and one
    instrument, so the speedup is read against its own control rather than against the
    llama-bench row from an older build."""
    ver = subprocess.run([_llama_bin("llama-server"), "--version"], capture_output=True,
                         text=True).stderr
    m = re.search(r"build (\d+), commit (\w+)", ver)
    from huggingface_hub import hf_hub_download
    draft = hf_hub_download(DRAFT_REPO, DRAFT_FILE)
    # What a verify round costs by width: a round checks draft+1 tokens in one batch, so
    # w / t_s(w) seconds. Near-flat t/s across 1..8 means the batch costs ~w serial steps
    # and even full acceptance cannot pay for the round; the drafter's speedup is bounded
    # by this, not by acceptance.
    cmd = [_llama_bin("llama-bench"), "-m", _gguf(), "-p", VERIFY_WIDTHS, "-n", "0",
           "-r", str(REPS), "-o", "json"]
    print(" ".join(cmd), flush=True)
    probe = {str(r["n_prompt"]): round(r["avg_ts"], 2) for r in json.loads(
        subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)}
    print(json.dumps(probe), flush=True)
    time.sleep(10)
    serial = _server_run([], "llama-dflash-serial.log")
    time.sleep(10)   # let Metal release the first load before the second
    default = _server_run(["-md", draft, "--spec-type", "draft-dflash",
                           "--spec-draft-n-max", str(DRAFT_N_MAX)], "llama-dflash.log")
    _save("llama-dflash", {
        "engine": "llama.cpp", "version": f"build {m.group(1)} ({m.group(2)})" if m else ver,
        "model": GGUF_FILE, "draft": DRAFT_FILE,
        "prompt_tokens": PROMPT_TOKENS, "gen_tokens": GEN_TOKENS,
        "verify_width_tps": {**probe, "flags": f"llama-bench -p {VERIFY_WIDTHS} -n 0"},
        "serial": {**serial, "flags": "llama-server, greedy, cache_prompt off"},
        "default": {**default, "flags": f"llama-server -md {DRAFT_FILE} --spec-type "
                                        f"draft-dflash --spec-draft-n-max {DRAFT_N_MAX}"},
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
        "model": "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf (native)",
        "prompt_tokens": PROMPT_TOKENS, "gen_tokens": GEN_TOKENS,
        "serial": {**serial, "flags": "CHAD_NO_DFLASH=1"},
        "default": {**default, "flags": "DFlash2 block drafter, per-round width schedule"},
    })


# ------------------------------------------------------------------------------- table
def table() -> None:
    def load(arm):
        p = os.path.join(RUNS, f"{arm}.json")
        return json.load(open(p)) if os.path.exists(p) else None
    def build(row):
        m = re.search(r"build (\d+)", row["version"])
        return m.group(1) if m else "?"
    ll, ld, ch = load("llama"), load("llama-dflash"), load("chad")
    rows = [("Engine", "Prefill (512-tok prompt)", "Decode (128 tok)", "Speculative decoding")]
    if ll:
        rows.append((f"llama.cpp `llama-bench` (stock, build {build(ll)})",
                     f"{ll['prefill_tps']:.0f} tok/s", f"{ll['decode_tps']:.1f} tok/s",
                     "off in this benchmark"))
    if ld:
        name = f"llama.cpp `llama-server` (build {build(ld)})"
        rows.append((f"{name}, serial", f"{ld['serial']['prefill_tps']:.0f} tok/s",
                     f"{ld['serial']['decode_tps']:.1f} tok/s", "off"))
        rows.append((name, f"{ld['default']['prefill_tps']:.0f} tok/s",
                     f"{ld['default']['decode_tps']:.1f} tok/s",
                     f"DFlash2 drafter ({ld['draft'].rsplit('-', 1)[-1].removesuffix('.gguf')} GGUF)"))
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
    {"llama": arm_llama, "llama-dflash": arm_llama_dflash, "chad": arm_chad,
     "table": table}[arm]()
