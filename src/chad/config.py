"""Single source of truth for CHAD_* configuration.

Typed accessors so every flag parses env the same way, and one place docs/tests can point
at (the substrate the docs drifted away from). The int/float helpers fold in
the lenient-parse contract used in `agent.py`: a non-numeric value warns and degrades
to the default instead of raising, so a typo in a budget knob can't abort startup.

Callers that resolve a value ONCE at import for hot-loop reasons keep doing so
(`agent._PREFILL_TRACE`, `diag._DISABLED`, `validate.VALIDATE`) — they just call `flag`/
`env_str` at import instead of hand-rolling `os.environ.get`. Nothing here is read on the
per-call path unless the caller chooses to; these are thin wrappers, not a live registry.
"""

import logging
import os

log = logging.getLogger("chad")  # same named logger as diag.log; no chad-module imports


def flag(name: str) -> bool:
    """A CHAD_* boolean: truthy iff the var is set to a non-empty value. Matches the
    repo's dominant `bool(os.environ.get(...))` / `not os.environ.get(...)` convention.
    NOTE: any non-empty value is true, including "0" — use `eq(name, "1")` when a var
    demands a strict `== "1"` opt-in rather than mere presence."""
    return bool(os.environ.get(name))


def eq(name: str, expected: str) -> bool:
    """Strict equality against the raw env value (no strip/case-fold). For the rare var
    whose contract is exactly `== "1"` and would be loosened by `flag`."""
    return os.environ.get(name) == expected


def env_str(name, default=None):
    """The env value if set to a non-empty string, else `default`. Mirrors the common
    `os.environ.get(name) or <fallback>` idiom (empty string collapses to the default)."""
    return os.environ.get(name) or default


def env_int(name, default=None):
    """Parse an int from env var `name`, or fall back to `default`. Unset/empty → default;
    a non-numeric value warns and degrades to the default instead of raising (lenient-parse
    rule) so a typo can't abort startup."""
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        log.warning("ignoring non-integer %s=%r; using default %r", name, v, default)
        return default


def env_float(name, default=None):
    """float sibling of `env_int` — same lenient-parse contract."""
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        log.warning("ignoring non-float %s=%r; using default %r", name, v, default)
        return default
