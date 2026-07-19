"""Contract tests for config.py — the single source of truth for CHAD_* env parsing.

`config` is where every flag/int/float knob now funnels, so its parse rules ARE
the product's behavior: `flag` truthy iff non-empty (so "0" is TRUE — a footgun worth
pinning), `eq` strict, the int/float helpers lenient (warn + default, never raise). These
table-tests lock those contracts so a future edit can't silently loosen `flag` into `eq`
or make a garbled budget knob crash startup.

Run: `uv run python test_config.py`
"""

import pytest

from chad import config

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def test_flag(monkeypatch):
    monkeypatch.delenv("CHAD_X", raising=False)
    check("unset -> False", config.flag("CHAD_X") is False)
    monkeypatch.setenv("CHAD_X", "")
    check("empty -> False", config.flag("CHAD_X") is False)
    monkeypatch.setenv("CHAD_X", "1")
    check("'1' -> True", config.flag("CHAD_X") is True)
    # FOOTGUN pinned: flag is presence-based, so ANY non-empty value is truthy —
    # including "0" and "false". Vars that need strict opt-in must use `eq`, not `flag`.
    monkeypatch.setenv("CHAD_X", "0")
    check("'0' -> True (presence, not value)", config.flag("CHAD_X") is True)
    monkeypatch.setenv("CHAD_X", "anything")
    check("arbitrary -> True", config.flag("CHAD_X") is True)


def test_eq(monkeypatch):
    monkeypatch.delenv("CHAD_X", raising=False)
    check("unset != '1'", config.eq("CHAD_X", "1") is False)
    monkeypatch.setenv("CHAD_X", "1")
    check("'1' == '1'", config.eq("CHAD_X", "1") is True)
    # strict: no strip/case-fold, and (unlike flag) "0"/"true" do NOT satisfy `== "1"`
    monkeypatch.setenv("CHAD_X", "0")
    check("'0' != '1'", config.eq("CHAD_X", "1") is False)
    monkeypatch.setenv("CHAD_X", " 1 ")
    check("' 1 ' != '1' (no strip)", config.eq("CHAD_X", "1") is False)


def test_env_str(monkeypatch):
    monkeypatch.delenv("CHAD_X", raising=False)
    check("unset -> default None", config.env_str("CHAD_X") is None)
    check("unset -> given default", config.env_str("CHAD_X", "d") == "d")
    monkeypatch.setenv("CHAD_X", "")
    check("empty -> default (falsy collapse)", config.env_str("CHAD_X", "d") == "d")
    monkeypatch.setenv("CHAD_X", "/some/path")
    check("value passes through", config.env_str("CHAD_X") == "/some/path")


def test_env_int(monkeypatch):
    monkeypatch.delenv("CHAD_X", raising=False)
    check("unset -> default None", config.env_int("CHAD_X") is None)
    check("unset -> given default", config.env_int("CHAD_X", 7) == 7)
    monkeypatch.setenv("CHAD_X", "")
    check("empty -> default", config.env_int("CHAD_X", 7) == 7)
    monkeypatch.setenv("CHAD_X", "131072")
    check("valid parses", config.env_int("CHAD_X") == 131072)
    monkeypatch.setenv("CHAD_X", "0")
    check("'0' -> 0 (not treated as unset)", config.env_int("CHAD_X", 7) == 0)
    # LENIENT contract: a non-numeric value degrades to the default, does NOT
    # raise — a typo in a budget knob must not abort startup. (cli._env_int is the strict
    # sibling that DOES raise; test_cli.py pins that.)
    monkeypatch.setenv("CHAD_X", "not-a-number")
    check("non-numeric -> default, no raise", config.env_int("CHAD_X", 7) == 7)
    monkeypatch.setenv("CHAD_X", "  ")
    check("whitespace int -> default", config.env_int("CHAD_X", 7) == 7)


def test_env_float(monkeypatch):
    monkeypatch.delenv("CHAD_X", raising=False)
    check("unset -> default None", config.env_float("CHAD_X") is None)
    check("unset -> given default", config.env_float("CHAD_X", 1.5) == 1.5)
    monkeypatch.setenv("CHAD_X", "")
    check("empty -> default", config.env_float("CHAD_X", 1.5) == 1.5)
    monkeypatch.setenv("CHAD_X", "2.5")
    check("valid parses", config.env_float("CHAD_X") == 2.5)
    monkeypatch.setenv("CHAD_X", "not-a-float")
    check("non-numeric -> default, no raise", config.env_float("CHAD_X", 1.5) == 1.5)


if __name__ == "__main__":
    for t in (test_flag, test_eq, test_env_str, test_env_int, test_env_float):
        with pytest.MonkeyPatch.context() as mp:
            t(mp)
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
