"""ATIF-v1.7 trajectory emitter.

The Terminal-Bench leaderboard rejects a submission whose passing trials lack an ATIF
trajectory, so these tests pin the schema invariants that harbor's pydantic models enforce:
`step_id` sequential from 1 document-wide, `source` in {system,user,agent}, every
`observation.results[].source_call_id` naming a `tool_call_id` in the SAME step, and no
extra keys (every ATIF model sets `extra: "forbid"`).

`tests/test_atif_harbor_contract.py` validates a real dump against harbor's actual models;
this file is the fast, dependency-free gate.
"""

import json

import pytest

from chad import atif

VALID_SOURCES = {"system", "user", "agent"}
STEP_KEYS = {"step_id", "timestamp", "source", "model_name", "reasoning_effort", "message",
             "reasoning_content", "tool_calls", "observation", "metrics",
             "is_copied_context", "llm_call_count", "extra"}


METRICS_KEYS = {"prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd",
                "prompt_token_ids", "completion_token_ids", "logprobs", "extra"}


@pytest.fixture(autouse=True)
def _reset():
    atif._reset_for_tests()
    yield
    atif._reset_for_tests()


def _messages():
    return [
        {"role": "system", "content": "you are chad"},
        {"role": "user", "content": "delete the temp files"},
        {"role": "assistant",
         "content": "<think>plan it</think>on it\n<tool_call>"
                    '{"name":"bash","arguments":{"command":"ls /tmp"}}</tool_call>'},
        {"role": "tool", "name": "bash", "content": "a.tmp b.tmp"},
        {"role": "assistant", "content": "<think>done</think>removed them"},
    ]


# --- split_think -------------------------------------------------------------

def test_split_think_explicit_block():
    r, v = atif.split_think("<think>reasoning</think>answer")
    assert r == "reasoning" and v == "answer"


def test_split_think_template_opened_case():
    """Ornith's template emits the opening tag, so generation starts INSIDE <think> and the
    text carries a leading </think> with no opener."""
    r, v = atif.split_think("thought so far</think>the answer")
    assert r == "thought so far" and v == "the answer"


def test_split_think_no_block():
    r, v = atif.split_think("plain answer")
    assert r == "" and v == "plain answer"


# --- steps_from_messages -----------------------------------------------------

def test_tool_results_attach_to_the_calling_step_not_their_own():
    steps = atif.steps_from_messages(_messages(), "ornith", [])
    assert [s["source"] for s in steps] == ["system", "user", "agent", "agent"]
    call_step = steps[2]
    assert call_step["tool_calls"][0]["function_name"] == "bash"
    assert call_step["tool_calls"][0]["arguments"] == {"command": "ls /tmp"}
    # the observation rides on the SAME step as the call it answers
    res = call_step["observation"]["results"][0]
    assert res["source_call_id"] == call_step["tool_calls"][0]["tool_call_id"]
    assert res["content"] == "a.tmp b.tmp"


def test_reasoning_is_split_out_of_the_message():
    steps = atif.steps_from_messages(_messages(), "ornith", [])
    assert steps[2]["reasoning_content"] == "plan it"
    assert "<think>" not in steps[2]["message"]
    assert steps[2]["message"].strip().startswith("on it")


def test_surplus_tool_results_get_a_null_source_call_id():
    """An edit nudge / validation reject appends a `tool` message with no matching call.
    ATIF allows source_call_id=None; inventing an id would fail harbor's validator."""
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant",
         "content": '<tool_call>{"name":"edit","arguments":{"path":"a"}}</tool_call>'},
        {"role": "tool", "name": "edit", "content": "ok"},
        {"role": "tool", "name": "edit", "content": "[nudge: re-read the file]"},
    ]
    results = atif.steps_from_messages(msgs, "m", [])[1]["observation"]["results"]
    assert results[0]["source_call_id"] is not None
    assert results[1]["source_call_id"] is None


def test_metrics_add_cached_tokens_back_into_prompt_tokens():
    """chad's GenStats.prompt_tokens excludes the cached prefix; ATIF's includes it. A warm
    KV cache must not read as a shrinking prompt."""
    stats = [atif.StepStat(prompt_tokens=100, cached_tokens=900, generated_tokens=50)]
    steps = atif.steps_from_messages(_messages(), "ornith", stats)
    m = steps[2]["metrics"]
    assert (m["prompt_tokens"], m["completion_tokens"], m["cached_tokens"]) == (1000, 50, 900)
    assert set(m) <= METRICS_KEYS, set(m) - METRICS_KEYS
    assert "metrics" not in steps[3]      # only one stat recorded


def test_no_unknown_keys_leak_into_steps():
    for s in atif.steps_from_messages(_messages(), "ornith", []):
        assert set(s) <= STEP_KEYS, set(s) - STEP_KEYS
        assert s["source"] in VALID_SOURCES


# --- recorder / document -----------------------------------------------------

def test_step_ids_are_sequential_from_one_across_segments():
    """Relaunches make a fresh Agent with empty messages. Each claims a segment;
    the document must still number steps 1..N globally or harbor's validator rejects it."""
    rec = atif.TrajectoryRecorder("/dev/null")
    a, b = rec.new_segment(), rec.new_segment()
    rec.set_segment(a, atif.steps_from_messages(_messages(), "m", []))
    rec.set_segment(b, atif.steps_from_messages(_messages(), "m", []))
    doc = rec.to_dict()
    assert [s["step_id"] for s in doc["steps"]] == list(range(1, 9))


def test_timestamps_are_kept_across_rebuilds():
    """The segment is rebuilt (and re-emitted) after every step; without first-seen
    timestamps every rewrite dragged all prior steps forward to the dump time, and a
    submitted trajectory showed ~all steps at the final flush, which reads as synthetic
    under a leaderboard integrity review. A step keeps the time its message was FIRST
    seen; only genuinely new messages get fresh ones."""
    clock = iter(f"2026-07-19T00:00:{i:02d}+00:00" for i in range(60)).__next__
    rec = atif.TrajectoryRecorder("/dev/null")
    seg = rec.new_segment()
    msgs = _messages()
    rec.set_segment(seg, atif.steps_from_messages(msgs[:3], "m", [], clock=clock))
    original_ts = [s["timestamp"] for s in rec.to_dict()["steps"]]
    rec.set_segment(seg, atif.steps_from_messages(msgs, "m", [], clock=clock))   # 2 more messages
    steps = rec.to_dict()["steps"]
    assert [s["timestamp"] for s in steps[:len(original_ts)]] == original_ts
    assert all("timestamp" in s for s in steps[len(original_ts):])
    assert steps[-1]["timestamp"] not in original_ts    # the new step is stamped now


def test_timestamps_follow_the_message_through_a_compaction_deletion():
    """Compaction deletes messages out of the middle of the transcript. Carrying prior
    timestamps forward by POSITION then handed every surviving step some earlier
    message's stamp — the stamp has to follow the message itself."""
    clock = iter(f"2026-07-19T00:00:{i:02d}+00:00" for i in range(60)).__next__
    rec = atif.TrajectoryRecorder("/dev/null")
    seg = rec.new_segment()
    msgs = _messages()
    rec.set_segment(seg, atif.steps_from_messages(msgs, "m", [], clock=clock))
    before = {s["message"]: s["timestamp"] for s in rec.to_dict()["steps"]}

    del msgs[1]                       # compaction sheds the oldest user turn
    rec.set_segment(seg, atif.steps_from_messages(msgs, "m", [], clock=clock))
    after = {s["message"]: s["timestamp"] for s in rec.to_dict()["steps"]}

    assert len(after) == len(before) - 1
    assert all(after[k] == before[k] for k in after), (before, after)
    # and the dropped message is not retained by the timestamp table (a tool message
    # is an observation on the step before it, never a step of its own)
    assert len(atif._STAMPS) == sum(1 for m in msgs if m["role"] != "tool")


def test_final_metrics_sum_across_steps():
    rec = atif.TrajectoryRecorder("/dev/null")
    seg = rec.new_segment()
    stats = [atif.StepStat(prompt_tokens=10, cached_tokens=5, generated_tokens=7,
                           think_tokens=4, gen_s=1.5),
             atif.StepStat(prompt_tokens=20, cached_tokens=1, generated_tokens=3,
                           think_tokens=2, gen_s=0.25)]
    rec.set_segment(seg, atif.steps_from_messages(_messages(), "m", stats))
    fm = rec.to_dict()["final_metrics"]
    assert fm["total_prompt_tokens"] == (10 + 5) + (20 + 1)
    assert fm["total_completion_tokens"] == 10
    assert fm["total_cached_tokens"] == 6
    assert fm["total_steps"] == 4
    assert fm["extra"] == {
        "total_think_tokens": 6, "total_gen_s": 1.75, "total_prefill_s": 0.0,
        "tok_per_s": round(10 / 1.75, 1), "cache_hit_rate": round(6 / 36, 3),
        "compactions": 0, "finishes": {"eos": 2}}


def test_dump_is_atomic_and_reparses(tmp_path):
    p = tmp_path / "nested" / "trajectory.json"
    rec = atif.TrajectoryRecorder(str(p))
    rec.model_name = "ornith-35b"
    rec.set_segment(rec.new_segment(), atif.steps_from_messages(_messages(), "ornith", []))
    rec.dump()
    doc = json.loads(p.read_text())
    assert doc["schema_version"] == "ATIF-v1.7"
    assert doc["agent"]["name"] == "chad" and doc["agent"]["model_name"] == "ornith-35b"
    assert len(list(p.parent.glob("*.tmp"))) == 0, "temp file left behind"


def test_dump_skips_an_empty_document(tmp_path):
    """ATIF requires steps: min_length=1. Writing a stepless doc would produce a file that
    fails validation — worse than no file, because it looks like a real trajectory."""
    p = tmp_path / "t.json"
    atif.TrajectoryRecorder(str(p)).dump()
    assert not p.exists()


def test_dump_never_raises_on_a_bad_path():
    """Telemetry must not be able to kill a benchmark trial."""
    rec = atif.TrajectoryRecorder("/nonexistent-root/nope/t.json")
    rec.set_segment(rec.new_segment(), atif.steps_from_messages(_messages(), "m", []))
    rec.dump()      # logs a warning, does not raise


# --- env gating --------------------------------------------------------------

def test_recorder_is_off_without_the_env_var(monkeypatch):
    monkeypatch.delenv("CHAD_TRAJECTORY_JSON", raising=False)
    assert atif.recorder() is None


def test_recorder_is_a_process_singleton(monkeypatch, tmp_path):
    """Relaunched Agents must share one document, not overwrite each other's file."""
    monkeypatch.setenv("CHAD_TRAJECTORY_JSON", str(tmp_path / "t.json"))
    assert atif.recorder() is atif.recorder() is not None


# --- what a step cost, and the step in flight ----------------------------------


def test_step_cost_rides_in_metrics_extra():
    """The three schema counts cannot tell a step that reasoned from one that decoded a
    file; the reasoning slice, the seconds and the finish reason go where the schema lets
    them — `metrics.extra` — so the document still validates."""
    stats = [atif.StepStat(generated_tokens=900, think_tokens=850, prefill_s=1.234,
                           gen_s=40.5, forwards=300, draft_proposed=1200,
                           draft_accepted=600, finish="interrupted")]
    extra = atif.steps_from_messages(_messages(), "m", stats)[2]["metrics"]["extra"]
    assert extra == {"think_tokens": 850, "prefill_s": 1.23, "gen_s": 40.5,
                     "tok_per_s": 22.2, "forwards": 300,
                     "draft_proposed": 1200, "draft_accepted": 600, "finish": "interrupted"}


def test_salvaged_is_recorded_only_when_it_happened():
    calm, saved = atif.StepStat(), atif.StepStat(salvaged=True)
    steps = atif.steps_from_messages(_messages(), "m", [calm, saved])
    assert "salvaged" not in steps[2]["metrics"]["extra"]
    assert steps[3]["metrics"]["extra"]["salvaged"] is True


def test_pending_step_files_an_unclosed_think_as_reasoning():
    """The template opened `<think>`, so the partial text carries no tag at all —
    `split_think` alone would publish a runaway think as the visible message."""
    step = atif.pending_step("so f(2,2,2,1,1) is", True, "m", 512, "t0")
    assert step["reasoning_content"] == "so f(2,2,2,1,1) is" and step["message"] == ""
    assert step["extra"] == {"in_flight": True, "generated_tokens": 512}
    assert set(step) <= STEP_KEYS

    closed = atif.pending_step("plan</think>writing it", True, "m", 600, "t0")
    assert closed["reasoning_content"] == "plan" and closed["message"] == "writing it"

    plain = atif.pending_step("no think here", False, None, 3, "t0")
    assert plain["message"] == "no think here" and "reasoning_content" not in plain


def test_pending_step_trails_its_segment_and_clears():
    rec = atif.TrajectoryRecorder("/dev/null")
    a, b = rec.new_segment(), rec.new_segment()
    rec.set_segment(a, atif.steps_from_messages(_messages(), "m", []))
    rec.set_segment(b, atif.steps_from_messages(_messages()[:2], "m", []))
    rec.set_pending(a, atif.pending_step("thinking", True, "m", 256, "t0"))

    steps = rec.to_dict()["steps"]
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    assert steps[4]["extra"]["in_flight"] is True          # after segment a's 4, before b
    assert "metrics" not in steps[4]

    rec.set_pending(a, None)
    assert all("extra" not in s for s in rec.to_dict()["steps"])


def test_start_installs_a_fresh_recorder_per_trial(tmp_path):
    first = atif.start(str(tmp_path / "a.json"))
    assert atif.recorder() is first
    second = atif.start(str(tmp_path / "b.json"))
    assert atif.recorder() is second and second is not first
    assert second.session_id != first.session_id


def test_draft_acceptance_is_split_by_phase_and_keeps_the_histogram():
    """One blended accept ratio hides both failure shapes: a head that misses the FIRST
    token (a wall at 0) and a width too narrow for the text (a pile at k). The histogram
    shows which, per phase — reasoning and a tool call accept very differently."""
    stat = atif.StepStat(draft_hist_think={4: [6, 2, 1, 0, 1]},      # 10 rounds, 8 accepted
                         draft_hist_act={4: [0, 0, 0, 0, 5], 2: [1, 0, 1]})
    draft = atif.steps_from_messages(_messages(), "m", [stat])[2]["metrics"]["extra"]["draft"]
    assert draft["think"] == {"rounds": 10, "proposed": 40, "accepted": 8,
                              "accepted_per_round": 0.8, "first_token_miss": 0.6,
                              "by_width": {"4": [6, 2, 1, 0, 1]}}
    assert draft["act"]["accepted_per_round"] == round(22 / 7, 2)
    assert list(draft["act"]["by_width"]) == ["2", "4"]
    json.dumps(draft)                      # string keys: the document must serialize


def test_a_step_that_drafted_nothing_carries_no_draft_block():
    extra = atif.steps_from_messages(_messages(), "m", [atif.StepStat()])[2]["metrics"]["extra"]
    assert "draft" not in extra


def test_trial_totals_roll_up_cache_draft_and_compaction():
    """A score table reads one row per trial; the rates that explain a slow trial have to
    be in it, pooled over steps by their real denominators (not a mean of step ratios)."""
    rec = atif.TrajectoryRecorder("/dev/null")
    seg = rec.new_segment()
    stats = [atif.StepStat(prompt_tokens=100, cached_tokens=900, generated_tokens=40,
                           gen_s=2.0, draft_hist_think={4: [5, 0, 0, 0, 5]}),
             atif.StepStat(prompt_tokens=3000, cached_tokens=0, generated_tokens=10,
                           gen_s=0.5, compacted=True, finish="interrupted",
                           draft_hist_act={2: [0, 0, 4]})]
    rec.set_segment(seg, atif.steps_from_messages(_messages(), "m", stats))
    extra = rec.to_dict()["final_metrics"]["extra"]
    assert extra["cache_hit_rate"] == round(900 / 4000, 3)
    assert extra["tok_per_s"] == 20.0
    assert extra["compactions"] == 1
    assert extra["finishes"] == {"eos": 1, "interrupted": 1}
    assert extra["draft_think"] == {"rounds": 10, "accept_rate": 0.5,
                                    "accepted_per_round": 2.0}
    assert extra["draft_act"] == {"rounds": 4, "accept_rate": 1.0, "accepted_per_round": 2.0}
