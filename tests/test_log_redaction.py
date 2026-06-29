"""Tests for log-preview secret redaction (_redact).

Two correctness properties:
  1. Long high-entropy blobs / known key prefixes ARE masked to <redacted:NN>.
  2. Ordinary short commands, code, and file paths are LEFT UNCHANGED (the key
     property — over-redaction would gut the log's diagnostic value).
"""
from chad.diag import redact as _redact


def check(desc, cond):
    assert cond, desc


def test_redacts_secrets():
    # --- should be redacted -----------------------------------------------------
    hex40 = "a3f9c1e2b4d6071829abcdef0123456789abcdef"  # 40 hex chars
    out = _redact(hex40)
    check("40-char hex blob masked", "<redacted:40>" in out and hex40 not in out)

    b64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5"  # 48 base64 chars
    out = _redact(b64)
    check("48-char base64 blob masked", "<redacted:" in out and b64 not in out)

    bearer_tok = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"  # 42 chars
    out = _redact(f"Authorization: Bearer {bearer_tok}")
    check("Bearer <40+ token> masked", "<redacted:" in out and bearer_tok not in out)
    check("Bearer prefix preserved", "Bearer" in out)

    sk = "sk-" + "0123456789abcdef0123456789abcdef0123"  # sk- + 36 chars
    out = _redact(sk)
    check("sk- key masked", "<redacted:" in out and "0123456789abcdef0123456789abcdef0123" not in out)

    ghp = "ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"
    out = _redact(ghp)
    check("ghp_ token masked", "<redacted:" in out)

    apikey = "api_key=AbCdEf1234567890AbCdEf1234567890XyZ"
    out = _redact(apikey)
    check("api_key=... masked", "<redacted:" in out)

    # base64 secret with `/` after a known prefix. The old single value class stopped
    # at the first `/`, leaking the tail of e.g. a real AWS secret; the wide prefixed
    # class must mask the WHOLE value.
    aws = "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    out = _redact(aws)
    check("aws secret with / fully masked (no tail leak)",
          "<redacted:" in out and "K7MDENG" not in out and "EXAMPLEKEY" not in out)
    check("aws_secret prefix preserved", out.startswith("aws_secret_access_key="))

    b64eq = "Authorization: Bearer YWxhZGRpbjpvcGVuc2VzYW1l+abc/def=="
    out = _redact(b64eq)
    check("base64 +/= after Bearer fully masked",
          "<redacted:" in out and "abc/def" not in out
          and out.startswith("Authorization: Bearer "))

    # length boundary: 31 chars below the floor (untouched), 32 at the floor (masked)
    b31 = "a" * 31
    check("31-char bare blob UNCHANGED (below 32 floor)", _redact(b31) == b31)
    b32 = "b" * 32
    check("32-char bare blob masked (at floor)", _redact(b32) == "<redacted:32>")


def test_leaves_normal_text_unchanged():
    # --- should be UNCHANGED (no over-redaction) --------------------------------
    for normal in [
        "ls -la",
        "def foo(): pass",
        "src/agent.py",
        "git status --porcelain",
        'grep -n "TURN start" agent.py',
        "/home/user/project/tests/test_log_redaction.py",  # long path w/ slashes must survive
        "the quick brown fox jumps over the lazy dog",
        "import logging from logging.handlers",
        "result: 42 chars written to file",
    ]:
        check(f"unchanged: {normal!r}", _redact(normal) == normal)


def test_session_log_opt_out():
    # CHAD_NO_SESSION_LOG opts out of the diagnostic file log. diag installs its handler
    # at import time, so test in a fresh subprocess (reload would leave the first import's
    # RotatingFileHandler attached to the shared named logger). Assert only a NullHandler
    # is attached and that redact/args_preview still work without the file handler.
    import os
    import subprocess
    import sys

    script = (
        "from chad import diag\n"
        "import logging\n"
        "diag.log.info('x')\n"
        "names = [type(h).__name__ for h in diag.log.handlers]\n"
        "assert names == ['NullHandler'], names\n"
        "assert diag.redact('a' * 32) == '<redacted:32>'\n"
        "assert diag.args_preview({'k': 'v'})\n"
        "print('OK', names)\n"
    )
    env = dict(os.environ, CHAD_NO_SESSION_LOG="1")
    out = subprocess.run([sys.executable, "-c", script], env=env,
                         capture_output=True, text=True)
    check(f"opt-out subprocess OK (stderr={out.stderr})", out.returncode == 0)
    check("only NullHandler attached", "OK ['NullHandler']" in out.stdout)


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
