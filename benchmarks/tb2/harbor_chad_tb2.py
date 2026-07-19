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
  chad speaks llama.cpp's raw ``/completion`` (token-id prompts, ``chad_backend=llama``,
  the only supported backend) against a GGUF served on a GPU box — that is the arm the
  maintainer's reference runs use. llama.cpp is the only server that passes ``<think>``
  back verbatim and reports real cache telemetry; an OpenAI-shaped server (mlx_lm.server)
  strips the thinking block and is NOT usable for a reasoning model. No weights load in
  the container.
- Sampling defaults to **temp 1.0** to match Ornith's reported TB2.1 recipe. chad sends
  temperature per-request, plus min_p/top_p ONLY when armed (chad_min_p/chad_top_p;
  0 = off = not sent); it never sends top_k. Keep the server's defaults neutral (on
  llama-server pass ``--top-p 1.0 --top-k 0 --min-p 0``) for full recipe fidelity.
- chad runs as ``environment.default_user`` (Harbor's convention); the install runs as
  root (needs to write /opt and bootstrap uv).

Register:  --agent-import-path harbor_chad_tb2:ChadAgent
Model:     -m mlx/nathansutton/Ornith-1.0-35B-...   (provider prefix stripped; the raw
           HF id is only used to label the run — no weights load in the container)
Agent kwargs (--ak key=value):
    chad_base_url=http://host.docker.internal:8081
                             REQUIRED  llama-server origin (bare, no /v1), AS REACHABLE
                             FROM INSIDE the task containers (host.docker.internal for a
                             server on the machine running the benchmark; a LAN/Tailscale
                             IP for a separate box).
    chad_backend=llama       which remote engine chad uses (default and only supported
                             value: llama = llama.cpp /completion)
    chad_tokenizer=<hf-repo>  HF repo whose tokenizer matches the served model's vocab.
                             REQUIRED (GGUF repos ship no tokenizer).
    chad_temp=1.0            sampling temperature (default 1.0 — TB2 reference)
    chad_min_p=0.05          quant-tail sampling filter; unset (default) = 0 =
                             OFF, byte-identical to pre-088 chad. CHAD_MIN_P per-request.
    chad_top_p=0.9           nucleus sampling filter (parity); unset (default) =
                             0 = OFF. CHAD_TOP_P per-request.
    chad_think=true|false    reasoning channel (default true)
    chad_think_ceiling=6000  close-and-continue think ceiling; unset (default) =
                             feature OFF, byte-identical to pre-086 chad. Passed through as
                             --think-ceiling.
    chad_timeout_sec=1500    host-side wall on the chad process (default 1500; keep < the
                             task's [agent] timeout so chad is killed cleanly, not by harbor)
    chad_deadline_margin_s=60  seconds of headroom between chad's own wall budget
                             (--turn-budget-s = timeout - margin) and the exec SIGKILL, so
                             the governor/wrap-up window can land a partial before the kill
0 disables the deadline plumbing.
    chad_review_pass=false   arm the early-finish self-review: a clean
                             finish with >30%% of the wall budget left triggers one fresh-
                             context verification pass. DEFAULT OFF — the 085 gate showed it
                             fires mostly on already-correct tasks (+140-584s wall, 0 flips
                             on the fast arm); arm explicitly for A/Bs only.
    chad_disable=<a,b>       comma list of harness levers to switch off in-container
                             (CHAD_DISABLE; see src/chad/levers.py) — the one-flag OFF
                             arm for lever A/Bs. Unknown names fail loudly at startup.
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
        chad_backend: str = "llama",
        chad_tokenizer: str | None = None,
        chad_temp: str | float = 1.0,
        chad_min_p: str | float = 0.0,
        chad_top_p: str | float = 0.0,
        chad_think: str | bool = True,
        chad_think_ceiling: str | int | None = None,
        chad_timeout_sec: str | int = 1500,
        chad_deadline_margin_s: str | int = 60,
        chad_review_pass: str | bool = False,
        chad_disable: str | None = None,
        chad_project: str | None = None,
        chad_workdir: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._base_url = chad_base_url
        if chad_backend != "llama":
            raise ValueError(
                f"chad_backend must be 'llama' (the only supported backend), "
                f"got {chad_backend!r}")
        self._backend = chad_backend
        self._tokenizer = chad_tokenizer
        self._temp = str(chad_temp)
        self._min_p = str(chad_min_p)
        self._top_p = str(chad_top_p)
        self._think = str(chad_think).lower() not in ("false", "0", "no", "off")
        self._think_ceiling = int(chad_think_ceiling) if chad_think_ceiling not in (None, "") else None
        self._timeout = int(chad_timeout_sec)
        # Deadline plumbing: tell chad its own wall budget so the governor arms on
        # every run and the wrap-up window fires before the exec SIGKILL. Margin is the
        # headroom left for chad to wrap up; 0 disables the whole deadline path.
        self._deadline_margin_s = int(chad_deadline_margin_s)
        self._review_pass = str(chad_review_pass).lower() not in ("false", "0", "no", "off")
        # Lever ablation arm (levers.py): comma list forwarded as CHAD_DISABLE inside the
        # container, so an A/B's OFF arm is one --ak instead of a code edit. Unknown lever
        # names are a hard error at chad startup (by design), so typos fail loudly.
        self._disable = (chad_disable or "").strip()
        self._project = chad_project or _REPO_ROOT
        self._workdir = chad_workdir
        self._installed = False
        self._tok_cached = False   # set once the tokenizer is confirmed in the HF cache
        self._tok_bundle_ok = False  # bundled cache verified in-container (offline load works)

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
            # jobs/dataset/models/*.tsv are run artifacts that live INSIDE the
            # project tree: gigabytes the container doesn't need, and jobs/ MUTATES
            # while a sweep is running — copying it here raced a concurrent cleanup
            # and failed a trial at setup (log-summary-date-ranges, 2026-07-12).
            shutil.copytree(
                self._project, clean / "src",
                ignore=shutil.ignore_patterns(
                    ".venv", "__pycache__", "*.pyc", ".git", ".mypy_cache",
                    ".pytest_cache", ".ruff_cache", "dist", "traces", ".gstack",
                    "jobs", "dataset", "models", "*.tsv"))
            tok_bundled = self._bundle_tokenizer(clean / "src")
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
        await self._prefetch_tokenizer(environment, bundled=tok_bundled)
        self._installed = True

    def _bundle_tokenizer(self, clean_src: Path) -> bool:
        """Copy the tokenizer repo into the upload so container setup never hits the Hub.

        The leaderboard's agent-setup budget is a hard 360s (no multipliers, no
        overrides — CI rejects both), and the ONLY >360s setups in the 2026-07-16
        full run were two trials where the in-container Hub fetch hung until
        `_prefetch_tokenizer`'s 600s exec cap (polyglot-rust-c / polyglot-c-py,
        both errored ≈613s = permanent reward-0 in a submission). Snapshotting
        host-side (host HF cache makes it a no-op after the first trial) and
        shipping `.hf_cache/` with the checkout turns that network dependence
        into ~20MB of upload; in-container "prefetch" becomes an offline load
        check. Best-effort: any failure falls back to the old network path.
        """
        tok_id = self._tokenizer or self._chad_model()
        if not tok_id:
            return False
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self.logger.warning("[chad-tb2] huggingface_hub not importable host-side; "
                                "keeping in-container Hub prefetch")
            return False
        try:
            # Tokenizer/config text files only — never weights (*.safetensors/*.gguf).
            snap = Path(snapshot_download(
                repo_id=tok_id,
                allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"]))
        except Exception as e:  # noqa: BLE001 — bundling is best-effort by design
            self.logger.warning(f"[chad-tb2] host-side tokenizer snapshot failed ({e}); "
                                "keeping in-container Hub prefetch")
            return False
        # Rebuild a MINIMAL cache instead of copying the host's models--* folder:
        # the host cache can hold full weight blobs and stale revisions (a naive
        # copytree here once produced a 51 GB upload). Only this revision's small
        # files, dereferenced, plus the refs/main pointer offline resolution needs.
        folder = "models--" + tok_id.replace("/", "--")
        rev = snap.name
        dest = clean_src / ".hf_cache" / folder
        allowed = (".json", ".txt", ".model", ".jinja")
        copied = 0
        for f in snap.rglob("*"):
            if not f.is_file() or f.suffix not in allowed or f.name.startswith("model"):
                continue  # name.startswith('model') skips model.safetensors.index.json
            target = dest / "snapshots" / rev / f.relative_to(snap)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)   # copyfile follows the blob symlink
            copied += 1
        if not copied:
            self.logger.warning(f"[chad-tb2] tokenizer snapshot for {tok_id} had no "
                                "bundleable files; keeping in-container Hub prefetch")
            return False
        (dest / "refs").mkdir(parents=True, exist_ok=True)
        (dest / "refs" / "main").write_text(rev)
        size_mb = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e6
        self.logger.info(f"[chad-tb2] tokenizer bundled into upload "
                         f"({tok_id}@{rev[:8]}, {copied} files, {size_mb:.1f} MB)")
        return True

    async def _prefetch_tokenizer(self, environment: BaseEnvironment,
                                  bundled: bool = False) -> None:
        """Warm the tokenizer into the container's HF cache during setup (the separate
        setup budget) and REQUIRE it to succeed.

        With ``bundled=True`` (see `_bundle_tokenizer`) this is an OFFLINE load check
        against the uploaded ``.hf_cache`` — no network, bounded at 90s — and the Hub
        loop below is only the fallback. Every timeout here is sized so install
        (~15-30s measured) + prefetch stays under the leaderboard's hard 360s
        agent-setup cap: the old 600s exec cap was itself the thing that blew it.

        The llama backend loads only a tokenizer client-side (`completion_engine.py`,
        `AutoTokenizer.from_pretrained`, unguarded) — generation is proxied over HTTP,
        but that one call still hits the
        Hub at chad startup. An unauthenticated pull there can be rate-limited, which
        crashed a whole trial before chad ran a single step (financial-document-
        processor: "Can't load tokenizer ... unauthenticated requests"). Retrying with
        backoff here and RAISING on failure — instead of the old silent degrade to an
        online load racing the rate limiter — turns that into a harbor-recorded setup/
        env failure (task re-runnable) rather than a guaranteed forfeit.
        """
        tok_id = self._tokenizer or self._chad_model()
        if not tok_id:
            self.logger.warning("[chad-tb2] no tokenizer id resolvable (no chad_tokenizer "
                                "and no model name); skipping prefetch")
            return
        served = self._chad_model()
        if self._tokenizer and served and self._tokenizer != served:
            # Vocab is quant-invariant across Ornith MLX/GGUF repos, so a differing repo
            # still *works* — but it doubles the Hub pulls and confuses triage.
            self.logger.warning(
                "[chad-tb2] served model (%s) and chad_tokenizer (%s) are different HF "
                "repos — confirm they share the same vocab family before trusting this run",
                served, tok_id)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
        if not hf_token:
            self.logger.warning(
                "[chad-tb2] no HF_TOKEN/HUGGING_FACE_HUB_TOKEN in the operator env — "
                "prefetch will hit the Hub unauthenticated and may be rate-limited; "
                "export one (see benchmarks/tb2/README.md)")
        tok = shlex.quote(tok_id)
        token_export = (f"export HF_TOKEN={shlex.quote(hf_token)} "
                        f"HUGGING_FACE_HUB_TOKEN={shlex.quote(hf_token)}; " if hf_token else "")
        if bundled:
            verify = (
                "export HOME=/root PATH=/root/.local/bin:/usr/local/bin:$PATH "
                f"HF_HUB_CACHE={_CHAD_SRC}/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
                # a+rX up front: run() may execute chad as a non-root user, and an image
                # with a restrictive umask could upload the bundle unreadable to it.
                f"chmod -R a+rX {_CHAD_SRC}/.hf_cache 2>/dev/null || true; "
                f"cd {_CHAD_SRC} && .venv/bin/python -c \"from transformers import AutoTokenizer; "
                f"AutoTokenizer.from_pretrained({tok!r})\" && echo '[chad-tb2] tokenizer cached'"
            )
            try:
                res = await environment.exec(verify, timeout_sec=90, user="root")
            except Exception as e:  # noqa: BLE001 — fall back to the network path below
                self.logger.warning(f"[chad-tb2] bundled-tokenizer verify exec failed: {e}")
                res = None
            if res is not None and "tokenizer cached" in (res.stdout or ""):
                self._tok_bundle_ok = True
                self._tok_cached = True
                self.logger.info("[chad-tb2] tokenizer loaded OFFLINE from bundled .hf_cache")
                return
            self.logger.warning("[chad-tb2] bundled tokenizer failed offline load — "
                                "falling back to Hub prefetch")
        fetch = (
            # hf-xet (the Rust CAS/fast-transfer client, pulled in by transformers'
            # huggingface_hub dep) throws opaque "Reqwest error: builder error" in
            # constrained container networks — a widely reported class of failure
            # (huggingface_hub#3266, xet-core#850/#581). Force the plain-HTTP downloader;
            # a single small tokenizer repo has nothing to gain from chunked CAS transfer.
            # HF_HUB_CACHE points the fallback fetch at the SAME world-readable checkout
            # cache the bundle uses. It used to fill /root/.cache — which a non-root
            # run_user can't even traverse, so run()'s forced-offline load then crashed
            # at chad startup on exactly the trials the prefetch exists to save.
            "export HOME=/root PATH=/root/.local/bin:/usr/local/bin:$PATH "
            f"HF_HUB_CACHE={_CHAD_SRC}/.hf_cache HF_HUB_DISABLE_XET=1; "
            + token_export +
            f"cd {_CHAD_SRC}; ok=0; "
            "for i in 1 2 3; do "
            f"  .venv/bin/python -c \"from transformers import AutoTokenizer; "
            f"AutoTokenizer.from_pretrained({tok!r})\" && {{ ok=1; break; }}; "
            "  echo \"[chad-tb2] tokenizer prefetch attempt $i failed; retrying\"; "
            "  sleep $((i * 5)); "
            "done; "
            f"[ $ok = 1 ] && {{ chmod -R a+rX {_CHAD_SRC}/.hf_cache 2>/dev/null || true; "
            "echo '[chad-tb2] tokenizer cached'; } || echo '[chad-tb2] tokenizer prefetch FAILED'"
        )
        # 240s cap (was 600): under the leaderboard's 360s setup budget a hung fetch
        # must die early enough to leave a legible error, not blow the whole phase.
        res = await environment.exec(fetch, timeout_sec=240, user="root")
        if "tokenizer cached" not in (res.stdout or ""):
            tail = (res.stdout or res.stderr or "")[-1200:]
            raise RuntimeError(
                f"[chad-tb2] tokenizer prefetch failed after 3 attempts for {tok_id!r} — "
                f"failing setup() instead of risking a silent online-load forfeit at chad "
                f"startup. tail: {tail}")
        self._tok_cached = True
        self.logger.info("[chad-tb2] tokenizer pre-fetched into HF cache")

    async def _detect_home(self, environment: BaseEnvironment, run_user) -> str:
        """The RUN USER's home INSIDE the container. This used to fall back to the
        HOST's $HOME for a non-root user — a path like /Users/<operator> that doesn't
        exist in the container, pointing chad's session dir (and the session.log we
        download) somewhere unwritable. Ask the container itself; /tmp is the
        always-writable fallback (chad creates $HOME/.chad beneath whatever this is)."""
        if str(run_user) in ("root", "0"):
            return "/root"
        try:
            res = await environment.exec('echo "${HOME:-}"', timeout_sec=15, user=run_user)
            h = (res.stdout or "").strip().splitlines()[-1].strip() if res.stdout else ""
            if h.startswith("/") and h != "/":
                return h
        except Exception:  # noqa: BLE001 — home detection is best-effort
            pass
        return "/tmp"

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
        home = await self._detect_home(environment, run_user)
        trace = "/tmp/chad.prefill.jsonl"
        stdout_c = "/tmp/chad.stdout.log"   # chad writes here IN the container
        # chad's diagnostic session log (diag.py -> $HOME/.chad/session.log): the ONLY place
        # the governor / wrap-up-window / step-cap decisions surface (they're log.info, not
        # stdout). We KEEP it on for evals — the container is throwaway and the log is
        # secret-redacted — and download it so a gate can confirm those levers fired (plan
        # 085 mechanism metrics). PYTHONUNBUFFERED + RotatingFileHandler keep it current, so
        # it survives the timeout SIGKILL like the stdout/trajectory files.
        session_c = f"{home}/.chad/session.log"
        # The leaderboard requires an ATIF trajectory for every PASSING trial. chad rewrites
        # this file after every step (src/chad/atif.py), so it survives the timeout SIGKILL
        # exactly like the stdout log. Verify a job with `validate_atif.py <jobdir>`.
        traj_c = "/tmp/chad.trajectory.json"
        # Only CHAD_* env is added; PATH stays the container default so the model's bash uses
        # the TASK's toolchain/python, not chad's venv.
        env = {
            "CHAD_MODEL": model or "",
            "CHAD_NO_SKILLS": "1",          # no personal-skill leakage into benchmark prompts
            "CHAD_PREFILL_TRACE": trace,
            "CHAD_TRAJECTORY_JSON": traj_c,   # ATIF-v1.7; required for passing trials
            "CHAD_TEMP": self._temp,        # TB2 reference sampling temperature
            "CHAD_MIN_P": self._min_p,      # quant-tail filter; "0" = OFF (default)
            "CHAD_TOP_P": self._top_p,      # nucleus parity; "0" = OFF (default)
            # Ornith's real window. Without this the remote-backend default is a
            # conservative 32k, which makes the in-container chad compact at ~30k — and
            # every client-side
            # compaction rewrites the transcript, which costs a FULL re-prefill on the
            # warm-prefix server. The server's own memory clamps bound the KV growth;
            # the client should use the window (its ctx-limit fallback caps at 120k).
            "CHAD_MAX_CONTEXT": "262144",
            "PYTHONUNBUFFERED": "1",
            "HOME": home,
        }
        if self._disable:
            env["CHAD_DISABLE"] = self._disable
        # Only force OFFLINE tokenizer load when setup() confirmed the cache is warm —
        # otherwise a cold cache + offline would guarantee the very startup crash
        # _prefetch_tokenizer exists to avoid. setup() now raises on a failed prefetch,
        # so this is always true by the time run() gets here; the guard stays for the
        # defensive setup() call above (a fresh run() without a prior setup()).
        if self._tok_cached:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            # BOTH warm paths land in the checkout cache: the bundled upload and the
            # fallback Hub prefetch (which used to fill root's ~/.cache — unreadable by
            # a non-root run_user, so forcing offline mode then GUARANTEED the startup
            # crash it exists to prevent). World-readable under /opt either way.
            env["HF_HUB_CACHE"] = f"{_CHAD_SRC}/.hf_cache"
        env["HF_HUB_DISABLE_XET"] = "1"  # see _prefetch_tokenizer; applies to any fallback pull too
        _hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if _hf:
            env["HF_TOKEN"] = _hf
            env["HUGGING_FACE_HUB_TOKEN"] = _hf

        parts = [f"{_CHAD_SRC}/.venv/bin/chad", shlex.quote(str(instruction)), "--yolo",
                 "--backend", self._backend, "--base-url", shlex.quote(self._base_url or "")]
        if self._tokenizer:
            parts += ["--tokenizer", shlex.quote(self._tokenizer)]
        if not self._think:
            parts.append("--no-think")
        if self._think_ceiling is not None:
            parts += ["--think-ceiling", str(self._think_ceiling)]
        # Hand chad a wall budget under the exec timeout so its governor + wrap-up
        # window arm (and, when armed, its early-finish review pass has a budget to measure
        # against). Leave `_deadline_margin_s` of headroom for chad to land a partial before
        # `environment.exec` SIGKILLs it at `self._timeout`.
        turn_budget_s = self._timeout - self._deadline_margin_s
        if turn_budget_s > 0:
            parts += ["--turn-budget-s", str(turn_budget_s)]
            if self._review_pass:
                parts.append("--review-pass")
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
                         (traj_c, "trajectory.json"), (session_c, "chad.session.log")):
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
