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
    stats = [{"prompt_tokens": 100, "cached_tokens": 900, "generated_tokens": 50}]
    steps = atif.steps_from_messages(_messages(), "ornith", stats)
    m = steps[2]["metrics"]
    assert m == {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 900}
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


def test_set_segment_keeps_first_seen_timestamps(monkeypatch):
    """The segment is rebuilt (and re-stamped `_now()`) after every step; without the
    positional timestamp merge, every rewrite dragged all prior steps' timestamps
    forward to the dump time — a submitted trajectory then showed ~all steps at the
    final flush, which reads as synthetic under a leaderboard integrity review. Step i
    must keep the timestamp it was FIRST dumped with; only new steps get fresh ones."""
    ticks = iter(f"2026-07-19T00:00:{i:02d}+00:00" for i in range(60))
    monkeypatch.setattr(atif, "_now", lambda: next(ticks))
    rec = atif.TrajectoryRecorder("/dev/null")
    seg = rec.new_segment()
    rec.set_segment(seg, atif.steps_from_messages(_messages()[:3], "m", []))
    original_ts = [s["timestamp"] for s in rec.to_dict()["steps"]]
    rebuilt = atif.steps_from_messages(_messages(), "m", [])  # 2 more messages appended
    assert [s["timestamp"] for s in rebuilt[:len(original_ts)]] != original_ts  # re-stamped
    rec.set_segment(seg, rebuilt)
    steps = rec.to_dict()["steps"]
    assert [s["timestamp"] for s in steps[:len(original_ts)]] == original_ts
    assert all("timestamp" in s for s in steps[len(original_ts):])


def test_final_metrics_sum_across_steps():
    rec = atif.TrajectoryRecorder("/dev/null")
    seg = rec.new_segment()
    stats = [{"prompt_tokens": 10, "cached_tokens": 5, "generated_tokens": 7},
             {"prompt_tokens": 20, "cached_tokens": 1, "generated_tokens": 3}]
    rec.set_segment(seg, atif.steps_from_messages(_messages(), "m", stats))
    fm = rec.to_dict()["final_metrics"]
    assert fm["total_prompt_tokens"] == (10 + 5) + (20 + 1)
    assert fm["total_completion_tokens"] == 10
    assert fm["total_cached_tokens"] == 6
    assert fm["total_steps"] == 4


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
