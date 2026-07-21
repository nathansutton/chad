"""E2e: a step whose text contains tool-call markers is NEVER a final answer.

The failure this closes: the model slides into an XML tool-call dialect for several
consecutive steps; the shared truncation counter is already spent, the done-audit
latch is consumed bouncing a garble, and a later garble is accepted as the final
answer with most of the wall budget still unspent. These tests drive the real
run_turn loop with scripted garbles and pin the three behaviors: own nudge budget +
exemplar, previous-garble context scrub, and the banked-note hard stop instead of
acceptance. The lever-off arm reproduces the legacy shared-counter behavior exactly
(the ablation contract).
"""


from chad import guardrails
from chad.agent import Agent
from chad.guardrails import GARBLE_NUDGE_CAP
from test_agent_e2e import ScriptedEngine, _agent, _tool_call

# A representative garble: closed <tool_call>, XML innards the repair pass cannot
# reconstruct.
GARBLE = '<tool_call>\n<function=write"\n</parameter>\n<parameter=path>x</parameter>\n</tool_call>'


def _script_prefix(f):
    return [
        _tool_call("write", path=str(f), content="print('hi')\n"),
        _tool_call("bash", command=f"python {f}"),   # verify -> clears unverified
    ]


def test_garbles_are_nudged_past_and_the_turn_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "done_audit,done_spec_recheck")  # isolate from the done bounces
    f = tmp_path / "out.py"
    script = _script_prefix(f) + [GARBLE, GARBLE,
                                  _tool_call("done", summary="recovered")]
    agent = _agent(script, max_steps=20)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == "recovered"
    assert agent.engine._i == len(script)
    tool_notes = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    garble_nudges = [c for c in tool_notes if "malformed" in c]
    assert len(garble_nudges) == 2
    # The 2nd consecutive garble gets the canonical exemplar…
    assert guardrails.TOOLCALL_EXEMPLAR in garble_nudges[1]
    assert guardrails.TOOLCALL_EXEMPLAR not in garble_nudges[0]
    # …and the FIRST garble's body is scrubbed from context (no wrong-dialect
    # few-shot left behind).
    assistants = [m["content"] for m in agent.messages if m.get("role") == "assistant"]
    assert guardrails.GARBLE_SCRUBBED in assistants
    assert sum(c == guardrails.GARBLE_SCRUBBED for c in assistants) == 1


def test_garble_exhaustion_hard_stops_instead_of_accepting(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE", "done_audit,done_spec_recheck")
    f = tmp_path / "out.py"
    script = _script_prefix(f) + [GARBLE] * (GARBLE_NUDGE_CAP + 1)
    # Big scripted ctx: the governor's cumulative-prefill budget (3x ctx) must not
    # fire before the garble budget does — this test is about the garble exit.
    agent = Agent(ScriptedEngine(script, effective_ctx=400000), mode="auto",
                  thinking=False, max_steps=30, ctx_limit=400000)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    # Consumed every garble + the hard stop — and the garble text was NOT returned.
    assert agent.engine._i == len(script)
    assert result.startswith("[stopped:")
    assert "malformed" in result
    assert "<tool_call>" not in result
    assert agent.budget_note  # progress note banked for continue/auto-continue


def test_lever_off_reproduces_the_shared_counter_accept(tmp_path, monkeypatch):
    """OFF arm (ablation): shared counter spends after 2 nudges and the 3rd garble is
    accepted as the final answer — the exact legacy behavior."""
    monkeypatch.setenv("CHAD_DISABLE", "garble_never_final,done_audit")
    f = tmp_path / "out.py"
    script = _script_prefix(f) + [GARBLE, GARBLE, GARBLE]
    agent = _agent(script, max_steps=20)

    result = agent.run_turn(f"write a script at {f} that prints hi")

    assert result == GARBLE  # the garble ships as the "answer" (the demonstrated bug)
    assistants = [m["content"] for m in agent.messages if m.get("role") == "assistant"]
    assert guardrails.GARBLE_SCRUBBED not in assistants  # no scrub on the OFF arm
