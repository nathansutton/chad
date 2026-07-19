"""Tests for the lever registry (levers.py) and per-model profiles (profiles.py).

These two modules exist to make a benchmark number falsifiable. `levers` lets an
ablation driver switch one behavior off without editing the tree, so a bundle of N
harness fixes can be attributed with N runs instead of N reverts. `profiles` draws the
line between core harness (fires on a real condition, helps any model) and model-specific
accommodation (encodes what Ornith needs), so a reviewer can ask how much of chad's score
survives `--profile generic` and get an answer.

The registry's most important property is that it FAILS LOUD on a typo. A misspelled
lever in an ablation script would otherwise run the unmodified harness and report a
per-lever delta of zero — which reads as "this lever does nothing", the single most
expensive way for this machinery to lie. That is the first thing asserted below.
"""
import os

import pytest

from chad import levers, profiles


def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_all_levers_default_on(monkeypatch):
    _clear(monkeypatch, "CHAD_DISABLE")
    assert levers.active() == sorted(levers.LEVERS)
    for name in levers.LEVERS:
        assert levers.enabled(name), f"{name} should default ON"


def test_disable_switches_off_only_the_named_lever(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "compact_notice")
    assert not levers.enabled("compact_notice")
    assert levers.enabled("compact_offload"), "disabling one lever must not touch another"
    assert "compact_notice" not in levers.active()


def test_disable_accepts_several_and_tolerates_whitespace(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", " compact_notice , plan_review ")
    assert not levers.enabled("compact_notice")
    assert not levers.enabled("plan_review")
    assert levers.enabled("profile_prompt")


def test_typo_in_disable_is_a_hard_error(monkeypatch):
    """The whole point. A silent typo makes an ablation measure the unmodified harness."""
    monkeypatch.setenv("CHAD_DISABLE", "compact_notic")  # missing the 'e'
    with pytest.raises(levers.UnknownLever):
        levers.enabled("compact_notice")


def test_guarding_an_unregistered_lever_raises(monkeypatch):
    """A guard the ablation driver cannot see must not silently pass."""
    _clear(monkeypatch, "CHAD_DISABLE")
    with pytest.raises(levers.UnknownLever):
        levers.enabled("no_such_lever")


def test_env_is_read_live_not_cached_at_import(monkeypatch):
    """ablate.py sets CHAD_DISABLE per subprocess, but the eval harness may also flip it
    in-process between tasks; a value frozen at import would silently ignore that."""
    _clear(monkeypatch, "CHAD_DISABLE")
    assert levers.enabled("plan_review")
    monkeypatch.setenv("CHAD_DISABLE", "plan_review")
    assert not levers.enabled("plan_review")


# === profiles ===============================================================

def test_profile_resolves_from_model_id(monkeypatch):
    _clear(monkeypatch, "CHAD_PROFILE")
    assert profiles.resolve("nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX").name == "ornith"
    assert profiles.resolve("hosted_vllm/ornith-35b-q6k").name == "ornith"
    assert profiles.resolve("Qwen/Qwen2.5-Coder-7B").name == "generic"


def test_unlabelled_run_is_an_ornith_run(monkeypatch):
    """chad ships exactly one model family; a run with no model id is one of those."""
    _clear(monkeypatch, "CHAD_PROFILE")
    assert profiles.resolve(None).name == "ornith"


def test_explicit_profile_overrides_the_model_id(monkeypatch):
    """The experiment the profile layer exists to enable: force `generic` against an
    Ornith endpoint and measure what the accommodations were worth."""
    monkeypatch.setenv("CHAD_PROFILE", "generic")
    assert profiles.resolve("hosted_vllm/ornith-35b-q6k").name == "generic"


def test_unknown_profile_is_a_hard_error(monkeypatch):
    monkeypatch.setenv("CHAD_PROFILE", "ornithh")
    with pytest.raises(ValueError):
        profiles.resolve("whatever")


def test_generic_profile_carries_no_ornith_accommodation(monkeypatch):
    _clear(monkeypatch, "CHAD_DISABLE")
    monkeypatch.setenv("CHAD_PROFILE", "generic")
    assert profiles.prompt_block("hosted_vllm/ornith-35b-q6k") == ""
    monkeypatch.setenv("CHAD_PROFILE", "ornith")
    assert "tool_response" in profiles.prompt_block(None)


def test_profile_block_is_gated_by_its_lever(monkeypatch):
    monkeypatch.setenv("CHAD_PROFILE", "ornith")
    monkeypatch.setenv("CHAD_DISABLE", "profile_prompt")
    assert profiles.prompt_block(None) == ""


def test_profile_block_is_additive_never_load_bearing(monkeypatch):
    """An ablation that bricks the agent measures nothing: the <tool_call> emission
    contract must survive --profile generic, or a generic run cannot call tools at all."""
    _clear(monkeypatch, "CHAD_DISABLE")
    monkeypatch.setenv("CHAD_PROFILE", "generic")
    from chad import prompt
    generic = prompt.build_system_prompt("Qwen/Qwen2.5-Coder-7B")
    assert "<tool_call>" in generic, "tool-call contract must be core, not profile"
    assert os.getcwd() in generic, "dynamic context still appended below the block"
