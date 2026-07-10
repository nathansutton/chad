"""Harbor agent adapter for **chad** on **Terminal-Bench 2** (leaderboard artifact).

This is the exact adapter behind the Terminal-Bench 2.0 numbers in the top-level
README — published so the run is reproducible. See `benchmarks/tb2/README.md` for the
full recipe (serve Ornith with llama.cpp, install harbor, run `run_tb2.sh`).

WHY CHAD RUNS *INSIDE* THE TASK CONTAINER
-----------------------------------------
Terminal-Bench 2's 89 tasks are general terminal work verified by *container
end-state*: build a Cython/CompCert/POV-Ray binary, configure a git webserver, crack a
7z hash, recover a WAL'd database, boot a QEMU VM. The effects live in installed
packages, running services, and files anywhere in the container — not one synced
workdir. So chad must run INSIDE the task container, where its bash/write/edit tools
operate the real shell and every side effect is captured.

WHY IN-CONTAINER INSTALL IS FEASIBLE
------------------------------------
- All 89 TB2 tasks declare ``allow_internet = true`` (verified across the dataset), so
  the in-container ``uv sync`` install has network.
- Containers are small (~2GB RAM / 10GB disk, per-task ``docker_image``), but chad is
  torch-free on Linux: ``mlx``/``mlx-lm`` are ``sys_platform == 'darwin'`` gated, and
  the ``llama`` backend needs only transformers/tokenizers (+ jedi/tree-sitter/
  rustworkx). Its uv venv fits.

DESIGN NOTES
------------
- Install runs in ``setup()`` (Harbor's separate agent-setup budget), so it does NOT
  eat the scored ``[agent] timeout_sec`` in run().
- Generation is remote: chad runs in-container pointed at a model server you provide.
  The Mac-native path (default, ``chad_backend=openai``) is ``mlx_lm.server`` on the
  Mac running the benchmark, reached from containers via ``host.docker.internal`` —
  the whole run stays on one machine. ``chad_backend=llama`` instead speaks llama.cpp's
  raw ``/completion`` (token-id prompts) for a GGUF served on a GPU box; that is the
  substitution the maintainer's reference runs use, only because the dev Mac's 24 GB
  can't hold the 35B. No weights load in the container either way.
- Sampling defaults to **temp 1.0** to match Ornith's reported TB2.1 recipe. chad sends
  temperature per-request; it does NOT send top_p/top_k/min_p, so keep the server's
  defaults neutral (mlx_lm.server already is; on llama-server pass
  ``--top-p 1.0 --top-k 0 --min-p 0``) for full recipe fidelity.
- chad runs as ``environment.default_user`` (Harbor's convention); the install runs as
  root (needs to write /opt and bootstrap uv).

Register:  --agent-import-path harbor_chad_tb2:ChadAgent
Model:     -m mlx/nathansutton/Ornith-1.0-35B-...   (provider prefix stripped; the raw
           HF id is only used to label the run — no weights load in the container)
Agent kwargs (--ak key=value):
    chad_base_url=http://host.docker.internal:8080/v1
                             REQUIRED  model server origin, AS REACHABLE FROM INSIDE
                             the task containers (host.docker.internal for a server on
                             the machine running the benchmark; a LAN/Tailscale IP for
                             a separate box). For chad_backend=openai include the /v1;
                             for chad_backend=llama it is the bare llama-server origin.
    chad_backend=openai|llama  which remote engine chad uses (default openai —
                             mlx_lm.server on a Mac; llama = llama.cpp /completion)
    chad_tokenizer=<hf-repo>  HF repo whose tokenizer matches the served model's vocab.
                             REQUIRED for llama (GGUF repos ship no tokenizer);
                             defaults to the model id for openai (the MLX repo ships it)
    chad_temp=1.0            sampling temperature (default 1.0 — TB2 reference)
    chad_think=true|false    reasoning channel (default true)
    chad_timeout_sec=1500    host-side wall on the chad process (default 1500; keep < the
                             task's [agent] timeout so chad is killed cleanly, not by harbor)
    chad_project=<path>      chad checkout to upload (default: the repo this file lives in)
    chad_workdir=<dir>       container dir to run chad in (default: auto via `pwd`, else /app)
"""
from __future__ import annotations

import os
import shlex
import shutil
import tempfile
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment

# This file lives in benchmarks/tb2/ inside the chad repo; the checkout to upload is the
# repo root, two levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PROVIDER_PREFIXES = ("openai/", "hosted_vllm/", "mlx/", "litellm/")
_CHAD_SRC = "/opt/chad-src"   # clean checkout uploaded here; uv builds .venv (py3.11) beside it.


class ChadAgent(BaseAgent):
    def __init__(
        self,
        *args,
        chad_base_url: str | None = None,
        chad_backend: str = "openai",
        chad_tokenizer: str | None = None,
        chad_temp: str | float = 1.0,
        chad_think: str | bool = True,
        chad_timeout_sec: str | int = 1500,
        chad_project: str | None = None,
        chad_workdir: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._base_url = chad_base_url
        if chad_backend not in ("openai", "llama"):
            raise ValueError(f"chad_backend must be 'openai' or 'llama', got {chad_backend!r}")
        self._backend = chad_backend
        self._tokenizer = chad_tokenizer
        self._temp = str(chad_temp)
        self._think = str(chad_think).lower() not in ("false", "0", "no", "off")
        self._timeout = int(chad_timeout_sec)
        self._project = chad_project or _REPO_ROOT
        self._workdir = chad_workdir
        self._installed = False

    @staticmethod
    def name() -> str:
        return "chad"

    def version(self) -> str | None:
        return "chad-tb2-0.1"

    def _chad_model(self) -> str | None:
        m = self.model_name
        if not m:
            return None
        for p in _PROVIDER_PREFIXES:
            if m.startswith(p):
                return m[len(p):]
        return m

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install chad INTO the task container (Harbor's separate setup budget).

        Uploads a clean copy of the checkout (no Mac .venv/.git/caches) and installs it
        LOCK-FAITHFULLY (`uv sync --frozen`) so the container gets the EXACT host versions:
        this model's chat template renders differently across transformers releases, and a
        loose `pip install` resolve once gave the model a prompt it wouldn't emit tool calls
        against. uv brings its own py3.11 (the task python is often 3.6-3.9); mlx is
        darwin-gated out, so no Apple binaries are pulled.
        """
        if not self._base_url:
            raise ValueError("chad_base_url is required (model server origin as seen "
                             "from inside the containers, e.g. "
                             "--ak chad_base_url=http://host.docker.internal:8080/v1)")
        self.logger.info(f"[chad-tb2] installing chad into container at {_CHAD_SRC}")
        clean = Path(tempfile.mkdtemp(prefix="chad_src_"))
        try:
            shutil.copytree(
                self._project, clean / "src",
                ignore=shutil.ignore_patterns(
                    ".venv", "__pycache__", "*.pyc", ".git", ".mypy_cache",
                    ".pytest_cache", ".ruff_cache", "dist", "traces", ".gstack"))
            await environment.upload_dir(clean / "src", _CHAD_SRC)
        finally:
            shutil.rmtree(clean, ignore_errors=True)

        # Bootstrap uv WITHOUT assuming curl: TB2 images are minimal and vary (fix-git is
        # Debian-slim with python3+pip but no curl/wget/uv). Try, in order: already-present
        # uv, `pip install uv` (arch-safe wheel; +--break-system-packages for PEP-668 images),
        # then curl/wget/apt/apk installers. All 89 tasks allow_internet=true, so any of these
        # can reach the network. Then install chad lock-faithfully (`uv sync --frozen`) so the
        # container gets the EXACT host dep versions — this model's chat template renders
        # differently across transformers releases.
        install = (
            "set -e; export HOME=/root PATH=/root/.local/bin:/usr/local/bin:$PATH; "
            "have(){ command -v \"$1\" >/dev/null 2>&1; }; "
            "if ! have uv; then have pip  && { pip  install -q uv || pip  install -q --break-system-packages uv || true; }; fi; "
            "if ! have uv; then have pip3 && { pip3 install -q uv || pip3 install -q --break-system-packages uv || true; }; fi; "
            "if ! have uv; then "
            "  if have curl; then curl -LsSf https://astral.sh/uv/install.sh | sh; "
            "  elif have wget; then wget -qO- https://astral.sh/uv/install.sh | sh; "
            "  elif have apt-get; then apt-get update -qq && apt-get install -y -qq curl && curl -LsSf https://astral.sh/uv/install.sh | sh; "
            "  elif have apk; then apk add --no-cache curl && curl -LsSf https://astral.sh/uv/install.sh | sh; "
            "  fi; "
            "fi; "
            "export PATH=/root/.local/bin:/usr/local/bin:$PATH; "
            "have uv || { echo 'FATAL: could not bootstrap uv'; exit 3; }; "
            f"cd {_CHAD_SRC} && uv sync --frozen --no-dev"
        )
        res = await environment.exec(install, timeout_sec=1800, user="root")
        if res.return_code != 0:
            tail = (res.stderr or res.stdout or "")[-1200:]
            # tree-sitter-language-pack is a native maturin/pyo3 wheel. On an image with no
            # matching wheel, uv source-builds it and cargo can fail (TB2 qemu-startup:
            # `exit status 101`), which errored the trial before chad ran a single step.
            # It only backs symbol ranking, and repomap import-guards it — so retry once
            # without it rather than forfeiting the task. Loud on purpose: a degraded run
            # must be visible in the log, never silently mistaken for a clean one.
            self.logger.warning(
                "[chad-tb2] install failed rc=%s; retrying WITHOUT "
                "tree-sitter-language-pack (symbol ranking degrades). tail: %s",
                res.return_code, tail)
            degraded = install + " --no-install-package tree-sitter-language-pack"
            res = await environment.exec(degraded, timeout_sec=1800, user="root")
            if res.return_code != 0:
                tail = (res.stderr or res.stdout or "")[-1200:]
                raise RuntimeError(
                    f"[chad-tb2] in-container install failed rc={res.return_code} "
                    f"(both full and degraded): {tail}")
            self.logger.warning("[chad-tb2] install OK in DEGRADED mode (no tree-sitter)")
        else:
            self.logger.info("[chad-tb2] install OK")
        self._installed = True

    async def _detect_workdir(self, environment: BaseEnvironment) -> str:
        if self._workdir:
            return self._workdir
        try:
            res = await environment.exec("pwd", timeout_sec=15)
            wd = (res.stdout or "").strip().splitlines()[-1].strip() if res.stdout else ""
        except Exception:
            wd = ""
        # '/' would be a degenerate cwd — fall back to the TB convention.
        if not wd or wd == "/":
            wd = "/app"
        return wd

    async def run(self, instruction, environment, context, *args, **kwargs) -> None:
        if not self._installed:
            # Harbor always calls setup() first, but be defensive: a fresh run() without a
            # prior successful setup() has no chad to invoke.
            await self.setup(environment)

        workdir = await self._detect_workdir(environment)
        run_user = environment.default_user if environment.default_user is not None else "root"
        self.logger.info(f"[chad-tb2] run: workdir={workdir} user={run_user} "
                         f"backend={self._backend} temp={self._temp}")
        logdir = Path(self.logs_dir).resolve()
        logdir.mkdir(parents=True, exist_ok=True)

        model = self._chad_model()
        trace = "/tmp/chad.prefill.jsonl"
        stdout_c = "/tmp/chad.stdout.log"   # chad writes here IN the container
        # The leaderboard requires an ATIF trajectory for every PASSING trial. chad rewrites
        # this file after every step (src/chad/atif.py), so it survives the timeout SIGKILL
        # exactly like the stdout log. Verify a job with `validate_atif.py <jobdir>`.
        traj_c = "/tmp/chad.trajectory.json"
        # Only CHAD_* env is added; PATH stays the container default so the model's bash uses
        # the TASK's toolchain/python, not chad's venv.
        env = {
            "CHAD_MODEL": model or "",
            "CHAD_NO_SESSION_LOG": "1",
            "CHAD_NO_SKILLS": "1",          # no personal-skill leakage into benchmark prompts
            "CHAD_PREFILL_TRACE": trace,
            "CHAD_TRAJECTORY_JSON": traj_c,   # ATIF-v1.7; required for passing trials
            "CHAD_TEMP": self._temp,        # TB2 reference sampling temperature
            "PYTHONUNBUFFERED": "1",
            "HOME": "/root" if str(run_user) in ("root", "0") else os.environ.get("HOME", "/root"),
        }

        parts = [f"{_CHAD_SRC}/.venv/bin/chad", shlex.quote(str(instruction)), "--yolo",
                 "--backend", self._backend, "--base-url", shlex.quote(self._base_url or "")]
        if self._tokenizer:
            parts += ["--tokenizer", shlex.quote(self._tokenizer)]
        if not self._think:
            parts.append("--no-think")
        # Redirect chad's stdout to a FILE INSIDE THE CONTAINER (not exec's return value):
        # on the timeout path `environment.exec` raises before returning, so a return-value
        # capture loses the whole trajectory on exactly the trials we most need to see (the
        # ~15-20min flails). PYTHONUNBUFFERED=1 keeps the file current, so it survives the
        # timeout SIGKILL — same durability trick as the prefill trace. We download it below
        # regardless of how the exec ended.
        cmd = " ".join(parts) + f" > {stdout_c} 2>&1"

        rc = -1
        try:
            res = await environment.exec(cmd, cwd=workdir, env=env,
                                         timeout_sec=self._timeout, user=run_user)
            rc = res.return_code
        except Exception as e:  # noqa: BLE001 — a timeout/mid-run env failure still leaves the tree scored
            self.logger.warning(f"[chad-tb2] exec raised (likely {self._timeout}s timeout): {e}")
        self.logger.info(f"[chad-tb2] chad exec rc={rc}")

        # Pull the durable in-container stdout + prefill trace (best-effort, survives timeout).
        # trajectory.json lands at the SAME path harbor's own agents use, so the
        # submission bundle and `validate_atif.py` find it without special-casing chad.
        for src, dst in ((stdout_c, "chad.stdout.log"), (trace, "chad.prefill.jsonl"),
                         (traj_c, "trajectory.json")):
            try:
                await environment.download_file(src, logdir / dst)
            except Exception as e:  # noqa: BLE001 — telemetry is best-effort
                self.logger.info(f"[chad-tb2] could not pull {src}: {e}")

        # A few TB2 tasks are git-based (fix-git, git-leak-recovery, git-multibranch). chad may
        # have edited as a different uid than the verifier expects; whitelist so the verifier's
        # git ops don't fatal on "dubious ownership".
        try:
            await environment.exec("git config --global --add safe.directory '*'",
                                   user="root", timeout_sec=15)
        except Exception:  # noqa: BLE001
            pass
