"""What ended a trial: the `stop` object, from the trajectory, in the row, and in the stats."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "polyglot"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trace as kit_trace  # noqa: E402

import run  # noqa: E402
import stats  # noqa: E402
import stops  # noqa: E402
import workspace  # noqa: E402
from catalog import Task  # noqa: E402
from harness import Solved  # noqa: E402
from harness.chad_inprocess import ChadInProcess, Loaded  # noqa: E402

from chad.base_engine import BackendError  # noqa: E402
from test_agent_e2e import ScriptedEngine  # noqa: E402
from test_polyglot_kit import _exercise  # noqa: E402


def _step(finish="eos", gen_s=10.0, prefill_s=1.0, calls=(), result="", message="",
          reasoning="thinking", in_flight=False):
    step = {"source": "agent", "step_id": 0, "message": message, "reasoning_content": reasoning,
            "tool_calls": [{"function_name": name, "arguments": {}} for name in calls]}
    if calls and result:
        step["observation"] = {"results": [{"source_call_id": None, "content": result}]}
    if in_flight:
        step["extra"] = {"in_flight": True, "generated_tokens": 5000}
    else:
        step["metrics"] = {"prompt_tokens": 10, "cached_tokens": 0, "completion_tokens": 50,
                           "extra": {"gen_s": gen_s, "prefill_s": prefill_s, "finish": finish}}
    return step


def _doc(*steps, agent="chad"):
    numbered = [{"source": "user"}, *steps]
    for n, step in enumerate(numbered, 1):
        step["step_id"] = n
    return {"agent": {"name": agent}, "steps": numbered}


@pytest.mark.parametrize("last, kind", [
    (_step(in_flight=True), "wall-think"),
    (_step(in_flight=True, message="writing the file"), "wall-act"),
    (_step(finish="interrupted"), "wall-think"),
    (_step(finish="interrupted", calls=["write"]), "wall-act"),
    (_step(calls=["bash"], result="[interrupted by user; no output before it was killed]"),
     "wall-tool"),
    (_step(calls=["bash"], result="ok 0.1s"), "wall-prefill"),
    (_step(), "wall-prefill"),
])
def test_a_capped_trial_says_where_the_wall_fell(last, kind):
    stop = stops.classify(_doc(_step(calls=["bash"], result="x"), last), True, 1200.0)
    assert (stop["kind"], stop["step"]) == (kind, 3)


def test_the_wall_splits_into_decode_prefill_and_the_rest():
    doc = _doc(_step(gen_s=100.0, prefill_s=5.0, calls=["bash"], result="x"),
               _step(gen_s=200.0, prefill_s=5.0, calls=["bash"],
                     result="[interrupted by user; no output before it was killed]"))
    stop = stops.classify(doc, True, 1200.0)
    assert stop["where"] == {"decode_s": 300.0, "prefill_s": 10.0, "tool_s": 890.0}
    # A step in flight at the kill has no measured decode time: the remainder is its.
    doc = _doc(_step(gen_s=100.0, prefill_s=5.0, calls=["bash"], result="x"), _step(in_flight=True))
    assert stops.classify(doc, True, 1200.0)["where"] == {
        "decode_s": 1195.0, "prefill_s": 5.0, "tool_s": 0.0}


def test_an_uncapped_trial_ends_as_the_agent_says():
    doc = _doc(_step(calls=["done"], result="ok"))
    assert stops.classify(doc, False, 30.0)["kind"] == "done"
    assert stops.classify(doc, False, 30.0, "step_cap")["kind"] == "step-cap"
    assert stops.classify(doc, False, 30.0, "done_rejected")["kind"] == "done-rejected"
    # An engine failure is infra whatever the flags say, and a CLI arm's non-zero exit is
    # an error.
    assert stops.classify(doc, True, 30.0, "infra")["kind"] == "infra"
    assert stops.classify(doc, False, 30.0, "error")["kind"] == "error"
    assert stops.classify(None, False, 30.0)["kind"] == "done"


def test_a_foreign_harness_cannot_say_where_and_gets_the_wall_alone():
    doc = _doc(_step(gen_s=100.0, in_flight=False), agent="goose")
    stop = stops.classify(doc, True, 1200.0)
    assert stop == {"kind": "unknown", "step": 2,
                    "where": {"decode_s": 0.0, "prefill_s": 0.0, "tool_s": 1200.0}}
    assert stops.classify(doc, False, 90.0)["kind"] == "done"


def test_the_row_carries_the_stop_beside_capped():
    task = Task(name="go/x", language="go", slug="x", source="/nowhere", solution=("x.go",),
                tests=("x_test.go",), example=(".meta/example.go",))
    verdict = workspace.Verdict(False, False, 2.0, "FAIL")
    doc = _doc(_step(in_flight=True))
    record = run.row(task, "arm", 1, Solved(1200.0, True, {"steps": 2}, stop_kind=None),
                     verdict, doc, "t.json", "")
    assert list(record)[4:8] == ["passed", "capped", "stop", "wall_s"]
    assert record["stop"]["kind"] == "wall-think"
    # The arm's own stop word rides Solved; a CLI arm's non-zero exit is an error.
    assert run.row(task, "arm", 1, Solved(9.0, False, {}, stop_kind="budget"), verdict, doc,
                   "t.json", "")["stop"]["kind"] == "budget"
    assert run.row(task, "arm", 1, Solved(9.0, False, {"exit_code": 2}), verdict, None,
                   "t.json", "")["stop"]["kind"] == "error"


def test_an_infra_row_is_redone_on_resume(tmp_path):
    rows = tmp_path / "trials.jsonl"
    rows.write_text(json.dumps({"task": "go/a", "rep": 1, "stop": {"kind": "wall-think"}}) + "\n"
                    + json.dumps({"task": "go/b", "rep": 1, "stop": {"kind": "infra"}}) + "\n"
                    + json.dumps({"task": "go/c", "rep": 1}) + "\n")
    assert run.done_keys(str(rows)) == {("go/a", 1), ("go/c", 1)}


class _DyingEngine(ScriptedEngine):
    def generate(self, *args, **kwargs):
        raise BackendError("llama-server error: connection refused", transient=True)


def test_an_engine_failure_marks_the_row_infra_and_the_block_goes_on(tmp_path):
    task = _exercise(tmp_path / "up")
    arm = ChadInProcess("scripted", thinking=False, max_steps=10,
                        load=lambda _m, _t: Loaded(_DyingEngine([]), 24000, 0.0),
                        root=str(tmp_path / "kit"))
    runs = tmp_path / "_runs"
    assert run.Block("arm", arm, wall_cap=120, runs=str(runs)).run([task], reps=1) == 0
    record = json.loads((runs / "arm" / "trials.jsonl").read_text())
    assert record["stop"]["kind"] == "infra" and record["passed"] is False
    assert run.done_keys(str(runs / "arm" / "trials.jsonl")) == set()


def _trial(task, passed, kind, wall=90.0):
    row = {"task": task, "language": task.split("/")[0], "passed": passed, "wall_s": wall,
           "gen_tokens": 100, "capped": kind.startswith("wall")}
    if kind:
        row["stop"] = {"kind": kind}
    return row


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_compare_leaves_infra_rows_out_and_says_so(tmp_path):
    a = _write(tmp_path / "a.jsonl", [_trial("go/a", True, "done"), _trial("go/b", False, "infra"),
                                      _trial("go/c", True, "done")])
    b = _write(tmp_path / "b.jsonl", [_trial("go/a", True, "done"), _trial("go/b", True, "done"),
                                      _trial("go/c", False, "wall-think")])
    c = stats.compare(stats.load_trials(a), stats.load_trials(b))
    # go/b is in A only as an infra row, so it is not a shared task and cannot flip up.
    assert (c.tasks, c.up, c.down, c.infra) == (2, 0, 1, (1, 0))
    assert "infra rows left out: A 1, B 0" in stats.compare_report(c)
    report = stats.score_report(stats.load_trials(b))
    assert "stops: done 2, wall-think 1" in report


def test_the_scorecard_counts_stops_per_arm(tmp_path):
    for label, rows in (("ref", [_trial("go/a", True, "done"), _trial("go/b", False, "wall-tool")]),
                        ("old", [_trial("go/a", True, ""), _trial("go/b", True, "")])):
        (tmp_path / label).mkdir()
        _write(tmp_path / label / "trials.jsonl", rows)
    table = stats.scorecard_report(str(tmp_path / "ref"), [str(tmp_path / "ref"), str(tmp_path / "old")])
    assert "| stops |" in table.splitlines()[0]
    assert "| done 1 wall-tool 1 | reference |" in table.splitlines()[2]
    assert table.splitlines()[3].endswith("|  | 1 / 0 (1.00) |")


def test_reclassify_backfills_published_rows_in_place(tmp_path):
    run_dir = tmp_path / "arm"
    (run_dir / "trajectories" / "go").mkdir(parents=True)
    (run_dir / "trajectories" / "go" / "a.rep1.json").write_text(json.dumps(_doc(_step(in_flight=True))))
    (run_dir / "trajectories" / "go" / "b.rep1.json").write_text(json.dumps(_doc(
        _step(calls=["bash"], result="[interrupted by user; killed]"))))
    rows = [{"task": "go/a", "language": "go", "passed": False, "capped": True, "wall_s": 1200.0,
             "trajectory": "trajectories/go/a.rep1.json", "test_tail": "x"},
            {"task": "go/b", "language": "go", "passed": False, "capped": True, "wall_s": 1200.0,
             "trajectory": "trajectories/go/b.rep1.json", "test_tail": "y"},
            {"task": "go/c", "language": "go", "passed": True, "capped": False, "wall_s": 50.0,
             "trajectory": "trajectories/go/missing.json", "test_tail": ""}]
    _write(run_dir / "trials.jsonl", rows)
    assert kit_trace.reclassify(str(run_dir)) == {"wall-think": 1, "wall-tool": 1, "done": 1}
    back = [json.loads(ln) for ln in (run_dir / "trials.jsonl").read_text().splitlines()]
    assert [r["stop"]["kind"] for r in back] == ["wall-think", "wall-tool", "done"]
    assert list(back[0]) == ["task", "language", "passed", "capped", "stop", "wall_s",
                             "trajectory", "test_tail"]
    assert back[2]["test_tail"] == "" and back[1]["stop"]["where"]["tool_s"] == 1189.0
