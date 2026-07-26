"""MLX runtime defaults, applied before mlx is first imported.

mlx latches these when its backend initializes, so they cannot be set from
`Engine.load()` — by then it is too late. `chad/__init__.py` calls
`apply_defaults()` at import, which is early enough for every entry point
(`chad`, `chad serve`, `chad-bench`, or `from chad.engine import Engine` in
someone else's script) because chad imports mlx lazily inside functions.

Values are measured offline on the real model, end-to-end, and only adopted when
the measurement can separate them from baseline. Nothing here needs a patched
mlx: these are env vars read by the stock PyPI wheel, so a `uvx chad-code` user
gets the tuned configuration with no build step and nothing to configure.

Anything already set in the environment is left alone, so a user override wins
and the sweep can measure an arm without fighting its own defaults.
Set CHAD_NO_MLX_TUNING=1 to skip the whole thing.
"""

from __future__ import annotations

import os

# name -> (value, why)
# Empty until `chad-bench --sweep` justifies an entry; see docs/benchmarks.md.
DEFAULTS: dict[str, tuple[str, str]] = {}


def apply_defaults() -> dict[str, str]:
    """Set tuned MLX vars that the user has not already chosen. Returns what was
    actually applied (empty when tuning is off or the user set everything)."""
    if os.environ.get("CHAD_NO_MLX_TUNING"):
        return {}
    applied = {}
    for name, (value, _why) in DEFAULTS.items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def describe() -> str:
    """One line per tuned var, for `chad levers` / diagnostics."""
    if not DEFAULTS:
        return "MLX runtime defaults: none (stock mlx settings)"
    rows = [f"  {n}={v}  — {why}" for n, (v, why) in DEFAULTS.items()]
    return ("MLX runtime defaults (override by exporting the var, or "
            "CHAD_NO_MLX_TUNING=1):\n" + "\n".join(rows))
