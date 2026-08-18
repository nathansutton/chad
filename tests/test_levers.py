"""Tests for the lever registry (levers.py).

chad 2.0.0 ships eight bash-route levers, all default ON; `CHAD_DISABLE` exists
so a leave-one-out ablation can switch one off without editing the tree.

The registry's most important property is that it FAILS LOUD on a typo. A
misspelled lever in an ablation script would otherwise run the unmodified
harness and report a per-lever delta of zero — which reads as "this lever does
nothing", the single most expensive way for this machinery to lie. That is the
first thing asserted below.
"""
import pytest

from chad import levers

LEAN_SET = {
    "env_manifest", "bash_read_skeleton", "bash_empty_diagnose",
    "bash_trim_keep_failures", "verify_baseline", "bash_line_clip",
    "edit_miss_diagnose", "rg_replace_flag_note",
}


def _clear(monkeypatch):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)


def test_registry_is_exactly_the_lean_set():
    assert set(levers.LEVERS) == LEAN_SET


def test_all_levers_default_on(monkeypatch):
    _clear(monkeypatch)
    for name in levers.LEVERS:
        assert levers.enabled(name), name
    assert levers.active() == sorted(levers.LEVERS)


def test_disable_one(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "bash_line_clip")
    assert not levers.enabled("bash_line_clip")
    assert levers.enabled("bash_trim_keep_failures")
    assert "bash_line_clip" not in levers.active()


def test_disable_all_is_bare(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "all")
    assert levers.active() == []
    for name in levers.LEVERS:
        assert not levers.enabled(name), name


def test_typo_in_disable_fails_loud(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "bash_line_clop")
    with pytest.raises(levers.UnknownLever):
        levers.validate_env()
    with pytest.raises(levers.UnknownLever):
        levers.enabled("bash_line_clip")


def test_unregistered_name_raises(monkeypatch):
    _clear(monkeypatch)
    with pytest.raises(levers.UnknownLever):
        levers.enabled("no_such_lever")
    with pytest.raises(levers.UnknownLever):
        levers.fired("no_such_lever")


def test_disable_read_live_not_cached(monkeypatch):
    """The eval harness flips CHAD_DISABLE between tasks in-process."""
    _clear(monkeypatch)
    assert levers.enabled("verify_baseline")
    monkeypatch.setenv("CHAD_DISABLE", "verify_baseline")
    assert not levers.enabled("verify_baseline")
    monkeypatch.delenv("CHAD_DISABLE")
    assert levers.enabled("verify_baseline")


def test_fired_counts_and_disabled_flag(monkeypatch):
    _clear(monkeypatch)
    before = levers.fire_counts().get("bash_line_clip", 0)
    levers.fired("bash_line_clip", removed=3)
    assert levers.fire_counts()["bash_line_clip"] == before + 1


def test_as_dict_shape():
    d = levers.as_dict()
    assert set(d) == LEAN_SET
    for meta in d.values():
        assert meta["description"]
