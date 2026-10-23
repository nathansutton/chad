"""chad in this process: the arm that is not a subprocess.

The weights load once per block (`start`), and each trial builds a fresh `Agent` on that
engine, as the shipped CLI builds it: sampler preset first, `CHAD_*` second, the
RAM-aware context limit, yolo mode, the default step budget.

This arm is exempt from the CLI arms' isolation on purpose. MLX needs Metal, which no
Seatbelt profile is trusted with, so the process keeps the real home, and chad confines
every bash command the model runs with its own sandbox, as it does for a user in yolo
mode.
"""
from __future__ import annotations

import contextlib
import functools
import io
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import catalog
import workspace
from catalog import JsonValue

from harness import AtifDoc, Solved, Trial

if TYPE_CHECKING:
    from chad.atif import TrajectoryRecorder
    from chad.base_engine import BaseEngine
    from chad.completion_engine import CompletionEngine
    from chad.engine import Engine

KV_CACHE_MAX_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class Loaded:
    engine: BaseEngine
    ctx_limit: int
    load_s: float
    backend: str = "mlx"


def load_engine(model: str, thinking: bool, *, backend: str = "mlx", server: str = "",
                tokenizer: str = "") -> Loaded:
    """The engine as the shipped CLI builds it: preset first, CHAD_* env second.

    `backend="llama"` is `chad --backend llama`: the same Agent on a `CompletionEngine`
    against the llama-server at `server`, rendering its prompts with the local
    `tokenizer`. No weights load in this process, and the window is the server's own."""
    if backend == "llama":
        return prime(llama_engine(server, tokenizer), thinking, backend)
    from chad.engine import Engine
    return prime(Engine(model_id=model, cache_dir=os.path.join(catalog.ROOT, "_kv"),
                        kv_cache_max_bytes=KV_CACHE_MAX_BYTES), thinking, backend)


def llama_engine(server: str, tokenizer: str) -> CompletionEngine:
    """chad's llama backend on the server at `server`, as the CLI builds it: the window
    left at 0 so `load()` reads the one the server enforces."""
    import harnesses

    from chad.completion_engine import CompletionEngine
    return CompletionEngine(model_id=harnesses.SERVED_MODEL, base_url=server,
                            tokenizer_id=tokenizer, effective_ctx=0)


def prime(eng: Engine | CompletionEngine, thinking: bool, backend: str) -> Loaded:
    """Sampler preset, CHAD_* overrides, load, then the context limit the load measured."""
    from chad.cli import _compute_ctx_limit, apply_sampler_env, apply_sampler_preset
    apply_sampler_preset(eng, thinking=thinking)
    apply_sampler_env(eng)
    load_s = eng.load()
    # SAFETY: CompletionEngine declares `reasoning_effort_default: None`, narrower than
    # the protocol's `str | None` and so not a BaseEngine to mypy; it never assigns the
    # attribute, and the CLI hands Agent this same object.
    return Loaded(cast("BaseEngine", eng), _compute_ctx_limit(eng), load_s, backend)


@contextlib.contextmanager
def _stdin_from_devnull() -> Iterator[None]:
    """Launched from a terminal, a confirm prompt would go to `input()` behind the
    redirected stdout — an invisible question no timeout interrupts. Closed at the file
    descriptor as well as `sys.stdin`, because spawned shells inherit fd 0."""
    saved_fd, saved_stdin = os.dup(0), sys.stdin
    devnull = open(os.devnull, encoding="utf-8")
    try:
        os.dup2(devnull.fileno(), 0)
        sys.stdin = devnull
        yield
    finally:
        sys.stdin = saved_stdin
        os.dup2(saved_fd, 0)
        os.close(saved_fd)
        devnull.close()


class ChadInProcess:
    name = "chad"

    def __init__(self, model: str, thinking: bool, max_steps: int | None,
                 load: Callable[[str, bool], Loaded] | None = None, root: str = catalog.ROOT,
                 *, backend: str = "mlx", server: str = "", tokenizer: str = "",
                 served: str = ""):
        self.model, self.thinking, self.max_steps = model, thinking, max_steps
        self.root = root
        # `served` is the file the server reports loading: the record that a llama block
        # ran the bytes it names, read by the caller while it checks the server is up.
        self.server, self.served = server, served
        self._load = load or functools.partial(load_engine, backend=backend, server=server,
                                               tokenizer=tokenizer)
        self._loaded: Loaded | None = None
        self._recorder: TrajectoryRecorder | None = None

    def version(self) -> str:
        import chad
        return chad.__version__

    def start(self) -> Mapping[str, JsonValue]:
        from chad import levers
        from chad.tools import active_schemas
        loaded = self._loaded = self._load(self.model, self.thinking)
        engine: dict[str, JsonValue] = {"backend": loaded.backend}
        if loaded.backend == "llama":
            engine |= {"server": self.server, "served_file": self.served}
        return {
            **engine,
            "model": self.model, "thinking": self.thinking, "ctx_limit": loaded.ctx_limit,
            "max_steps": self.max_steps, "load_s": round(loaded.load_s, 1),
            "levers_active": levers.active(),
            "chad_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("CHAD_")},
            "tools": sorted(s["function"]["name"] for s in active_schemas()),
        }

    def prepare(self, home: str) -> None:
        """Nothing to write: this arm runs in the real home (see the module docstring)."""

    def solve(self, trial: Trial) -> Solved:
        from chad import atif, levers
        from chad.agent import Agent

        if self._loaded is None:
            raise RuntimeError("solve() before start(): the engine is not loaded")
        deadline = time.time() + trial.wall_cap_s
        fires_before = levers.fire_counts()
        start_dir, saved_path = os.getcwd(), os.environ.get("PATH", "")
        try:
            os.chdir(trial.workspace)
            os.environ["PATH"] = workspace.trial_env()["PATH"]
            # Before the Agent: it binds the recorder it is built with.
            self._recorder = atif.start(trial.trajectory)
            build = functools.partial(Agent, self._loaded.engine, thinking=self.thinking,
                                      mode="yolo", ctx_limit=self._loaded.ctx_limit,
                                      should_stop=lambda: time.time() > deadline,
                                      emit=lambda _kind, _text: None)
            # No --max-steps keeps the agent's own default rather than restating it here.
            agent = build() if self.max_steps is None else build(max_steps=self.max_steps)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()), _stdin_from_devnull():
                agent.run_turn(trial.prompt, stream=False)
            wall = time.time() - t0
        finally:
            os.chdir(start_dir)
            os.environ["PATH"] = saved_path
        fired = {k: v - fires_before.get(k, 0) for k, v in levers.fire_counts().items()
                 if v > fires_before.get(k, 0)}
        # The deadline stops the agent a moment BEFORE the cap, so the wall time alone
        # reads a capped trial as one that finished at 1199.8 s.
        return Solved(wall, agent.interrupted, {
            "steps": sum(1 for m in agent.messages if m.get("role") == "assistant"),
            "tool_dispatches": agent.tool_dispatches,
            "gen_tokens": agent.gen_tokens, "think_tokens": agent.think_tokens,
            "prefill_tokens": agent.prefill_tokens, "peak_ctx": agent.peak_ctx,
            "gen_time_s": round(agent.gen_time, 1),
            "levers_fired": fired,
        })

    def trajectory(self) -> AtifDoc | None:
        return self._recorder.to_dict() if self._recorder else None
