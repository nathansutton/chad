"""Diagnostic session log for chad (extracted from agent.py).

Throughput / prefill numbers and a readable trace — the user query, tool-call args
(bash commands, write/edit content), and result previews — are diagnostics, not UX, so
they go to ~/.chad/session.log (tail it live) instead of cluttering the transcript.
Dogfooding showed sessions can "do nothing then stop" with no way to see why: which
tool args were used, whether a search returned anything, whether the model tried to
declare done, which cache served the turn. This module is that trace.

The log is bounded (5 MB x3 rotation) and previews pass through a best-effort secret
redactor, but it still records command/file previews in plaintext outside the repo, so
treat it as sensitive. None of this touches the model-facing transcript or the tool
results the model sees — it is the diagnostic trace only.
"""
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler

from . import config

_LOG_DIR = os.path.expanduser("~/.chad")
log = logging.getLogger("chad")
# Local privacy opt-out: set CHAD_NO_SESSION_LOG (any truthy value) to disable the
# diagnostic session log entirely, matching the CHAD_NO_VALIDATE convention. When opted
# out we install a NullHandler (so the many log.info calls stay cheap no-ops and Python
# never warns about missing handlers) and never create ~/.chad for the log's sake.
_DISABLED = config.flag("CHAD_NO_SESSION_LOG")
if _DISABLED:
    log.addHandler(logging.NullHandler())
    log.propagate = False
else:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        # Bounded rotation (5MB x3) on the NAMED logger, not root, so other libraries'
        # logging isn't redirected into our session file. propagate=False keeps these
        # lines off the root handlers. Nothing else relies on basicConfig/root here.
        _handler = RotatingFileHandler(
            os.path.join(_LOG_DIR, "session.log"), maxBytes=5_000_000, backupCount=3)
        _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.setLevel(logging.INFO)
        log.addHandler(_handler)
        log.propagate = False
    except OSError:
        pass


# Coarse, conservative secret masking for what gets WRITTEN TO THE LOG only.
# Two branches:
#   1. A known key prefix (Bearer/sk-/ghp_/xox/aws_secret/api_key) followed by a
#      value in a WIDE class that includes base64 punctuation (+ / =). The old
#      single-class regex stopped at the first `/`, so an AWS secret like
#      `wJalrXUtnFEMI/K7MDENG/bPx…` was only half-masked — the wide class fixes that.
#   2. A bare 32+ high-entropy blob in a NARROW class (no `/` or `+`) so ordinary
#      file paths and URLs are left untouched — over-redaction silently guts the
#      log's diagnostic value, which is worse than the status quo.
_SECRET_RE = re.compile(
    r"(?i)"
    r"(bearer\s+|sk-|ghp_|xox[baprs]-|aws_secret_access_key\s*=\s*|api[_-]?key\s*[=:]\s*)"
    r"([A-Za-z0-9_\-+/]{16,}={0,2})"
    r"|([A-Za-z0-9_\-]{32,})")


def redact(s: str) -> str:
    """Mask secrets in a log preview. Known-prefix secrets keep their prefix and mask
    the (wide) value; bare high-entropy blobs are masked whole. Length-preserving so
    `<redacted:NN>` still carries a rough signal of what was there."""
    def _mask(m):
        if m.group(1) is not None:  # known-prefix secret (wide value class)
            return m.group(1) + "<redacted:" + str(len(m.group(2))) + ">"
        return "<redacted:" + str(len(m.group(3))) + ">"  # bare high-entropy blob
    return _SECRET_RE.sub(_mask, s)


def warn_footer(warnings) -> list:
    """One-line footer summarizing config/discovery warnings for the `/mcp` and
    `/skills` summaries: `(N warning(s): a; b; c …)`, showing the first three. Returns
    [] when there are none. Shared so the format lives in exactly one place."""
    if not warnings:
        return []
    return [f"({len(warnings)} warning(s): " + "; ".join(warnings[:3])
            + (" …" if len(warnings) > 3 else "") + ")"]


def args_preview(args, n=160):
    """One-line, redacted, length-capped preview of tool-call args for the log."""
    try:
        s = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        s = str(args)
    s = s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"
    return redact(s)


def result_preview(result, n=120):
    """One-line, redacted preview of a tool result, with its full length annotated."""
    s = str(result).replace("\n", "⏎")
    extra = f"  [{len(str(result))} chars]"
    return redact((s if len(s) <= n else s[:n] + "…")) + extra
