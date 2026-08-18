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


def test_all_levers_default_off(monkeypatch):
    """The 1.10.0 contract: a default run is the bare model + tool loop. The measured
    record (2026-08 clean-slate arms) is what earned this polarity; a lever ships ON
    only behind a positive pre-registered contrast."""
    _clear(monkeypatch, "CHAD_ENABLE", "CHAD_DISABLE", "CHAD_LEAN")
    assert levers.active() == []
    for name in levers.LEVERS:
        assert not levers.enabled(name), f"{name} should default OFF"


def test_lean_arm_starts_with_the_bash_route_levers_on(monkeypatch):
    """CHAD_LEAN hides `read`/`grep`, and four of their guarantees live INSIDE those
    tools. Re-arming the bash-route equivalents is what keeps hiding the tools an
    experiment about tool surface rather than a silent repeal of the result contract —
    and a lever bound to a hidden tool cannot fire, so an ablation would misread it as
    "exercised and useless"."""
    _clear(monkeypatch, "CHAD_ENABLE", "CHAD_DISABLE")
    monkeypatch.setenv("CHAD_LEAN", "1")
    assert levers.active() == sorted(levers.LEAN_DEFAULTS)
    for name in levers.LEAN_DEFAULTS:
        assert levers.enabled(name)
    # everything else stays at the bare default
    assert not levers.enabled("session_ledger")


def test_lean_defaults_still_compose_with_enable_and_disable(monkeypatch):
    """The lean set is a starting point, not an override: CHAD_ENABLE still adds and
    CHAD_DISABLE still subtracts, so leave-one-out stays a valid arm inside lean."""
    monkeypatch.setenv("CHAD_LEAN", "1")
    monkeypatch.setenv("CHAD_ENABLE", "session_ledger")
    monkeypatch.setenv("CHAD_DISABLE", "verify_baseline")
    assert levers.enabled("session_ledger")           # added
    assert levers.enabled("bash_read_skeleton")       # lean default kept
    assert not levers.enabled("verify_baseline")      # subtracted
    monkeypatch.setenv("CHAD_ENABLE", "all")
    assert levers.enabled("compact_notice"), "'all' must not be narrowed by the lean set"


def test_lean_defaults_are_all_registered():
    assert levers.LEAN_DEFAULTS <= set(levers.LEVERS)


def test_enable_all_turns_everything_on(monkeypatch):
    _clear(monkeypatch, "CHAD_DISABLE")
    monkeypatch.setenv("CHAD_ENABLE", "all")
    assert levers.active() == sorted(levers.LEVERS)
    for name in levers.LEVERS:
        assert levers.enabled(name)


def test_enable_switches_on_only_the_named_lever(monkeypatch):
    _clear(monkeypatch, "CHAD_DISABLE")
    monkeypatch.setenv("CHAD_ENABLE", "compact_notice")
    assert levers.enabled("compact_notice")
    assert not levers.enabled("compact_offload"), "enabling one lever must not touch another"
    assert levers.active() == ["compact_notice"]


def test_leave_one_out_idiom(monkeypatch):
    """The ablation driver's OFF arm: everything on minus exactly one lever."""
    monkeypatch.setenv("CHAD_ENABLE", "all")
    monkeypatch.setenv("CHAD_DISABLE", "compact_notice")
    assert not levers.enabled("compact_notice")
    assert levers.enabled("compact_offload"), "disabling one lever must not touch another"
    assert "compact_notice" not in levers.active()


def test_enable_accepts_several_and_tolerates_whitespace(monkeypatch):
    _clear(monkeypatch, "CHAD_DISABLE")
    monkeypatch.setenv("CHAD_ENABLE", " compact_notice , plan_review ")
    assert levers.enabled("compact_notice")
    assert levers.enabled("plan_review")
    assert not levers.enabled("profile_prompt")


def test_typo_in_enable_or_disable_is_a_hard_error(monkeypatch):
    """The whole point. A silent typo makes an ablation measure the bare harness."""
    monkeypatch.setenv("CHAD_ENABLE", "compact_notic")  # missing the 'e'
    with pytest.raises(levers.UnknownLever):
        levers.enabled("compact_notice")
    monkeypatch.setenv("CHAD_ENABLE", "all")
    monkeypatch.setenv("CHAD_DISABLE", "compact_notic")
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
