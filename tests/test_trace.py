"""The autopsy: a run's longest thinks, and what a ceiling would have done to them."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "polyglot"))

import trace as kit_trace  # noqa: E402

DRAFT = "```go\nfunc Main() {}\n```"
# Two complete drafts separated by prose, then an announcement and a third, open one: the
# shape of every fatal think read by hand.
REASONING = ("Let me think about the channel.\n" + DRAFT + "\nHmm, the order is wrong.\n"
             + DRAFT + "\nLet me write the file.\n```go\nfunc Main() {\n")


def _step(think, reasoning="", finish="eos", calls=(), in_flight=False):
    step = {"source": "agent", "reasoning_content": reasoning,
            "tool_calls": [{"function_name": name, "arguments": {}} for name in calls]}
    if in_flight:
        step["extra"] = {"in_flight": True, "generated_tokens": think}
    else:
        step["metrics"] = {"prompt_tokens": 100, "cached_tokens": 90, "completion_tokens": think,
                           "extra": {"think_tokens": think, "gen_s": think / 10, "finish": finish}}
    return step


def _run(root, rows):
    """`rows` are (task, passed, capped, steps); writes a run with its rows and trajectories."""
    lines = []
    for task, passed, capped, steps in rows:
        lang, slug = task.split("/")
        path = os.path.join("trajectories", lang, f"{slug}.rep1.json")
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        numbered = [{"source": "user"}, *steps]
        for n, step in enumerate(numbered, 1):          # ATIF: sequential from 1
            step["step_id"] = n
        full.write_text(json.dumps({"steps": numbered}))
        lines.append(json.dumps({"task": task, "language": lang, "rep": 1, "passed": passed,
                                 "capped": capped, "wall_s": 1200.0 if capped else 90.0,
                                 "trajectory": path}))
    (root / "trials.jsonl").write_text("\n".join(lines) + "\n")
    return str(root)


def test_autopsy_reads_the_longest_think_and_what_follows_it(tmp_path):
    run = _run(tmp_path / "arm", [
        ("go/killed", False, True, [_step(50, "", calls=["bash"]),
                                    _step(9000, REASONING, in_flight=True)]),
        ("go/recovered", True, False, [_step(3000, REASONING, calls=["bash"]),
                                       _step(40, "ok", calls=["write"])]),
    ])
    by_task = {a.task: a for a in kit_trace.autopsy_run(run)}
    killed, recovered = by_task["go/killed"], by_task["go/recovered"]
    # The step in flight at the kill: its tokens are what it generated, it has no decode
    # time yet, and nothing came after it.
    assert (killed.think_tokens, killed.gen_s, killed.at_kill) == (9000, None, True)
    assert (killed.drafts, killed.wrote_later, killed.passed) == (2, False, False)
    # Offsets are chars / 3.6 without a tokenizer, measured from the start of the reasoning.
    assert killed.first_fence == round(REASONING.index("```") / kit_trace.CHARS_PER_TOKEN)
    assert killed.first_write == round(REASONING.index("Let me write") / kit_trace.CHARS_PER_TOKEN)
    # A finished long think that a later step followed with a write.
    assert (recovered.step_id, recovered.gen_s, recovered.at_kill) == (2, 300.0, False)
    assert recovered.wrote_later and recovered.passed


def test_autopsy_counts_tokens_with_the_tokenizer_it_is_given(tmp_path):
    run = _run(tmp_path / "arm", [("go/x", True, False, [_step(500, REASONING)])])
    words = kit_trace.autopsy_run(run, count=lambda text: len(text.split()))
    assert words[0].first_fence == len(REASONING[:REASONING.index("```")].split())


def test_autopsy_report_says_what_each_ceiling_cuts(tmp_path):
    run = _run(tmp_path / "arm", [
        ("go/a", False, True, [_step(9000, REASONING, finish="interrupted")]),
        ("go/b", False, True, [_step(1500, "no code here")]),
        ("go/c", True, False, [_step(3000, REASONING)]),          # fence at ~9 tokens
        ("go/d", True, False, [_step(3000, "x" * 20000 + DRAFT)]),  # fence at ~5556 tokens
        ("go/e", True, False, [_step(200, "")]),
    ])
    report = kit_trace.autopsy_report(kit_trace.autopsy_run(run), [1024, 2048, 4096], "chars/3.6")
    assert "2 failing trials, 3 passing; offsets in chars/3.6" in report
    lines = {ln.split(" | ")[0]: ln for ln in report.splitlines() if ln.startswith("| ")}
    # 1024 cuts both failures and touches two passes, one of them before its first draft.
    assert lines["| 1024"] == "| 1024 | 2 of 2 | 2 of 3 | 1 |"
    assert lines["| 2048"] == "| 2048 | 1 of 2 | 2 of 3 | 1 |"
    assert lines["| 4096"] == "| 4096 | 1 of 2 | 0 of 3 | 0 |"
    # Failures sort first, longest think first; a row names the trial and its result.
    assert report.splitlines()[2].startswith("| arm | go/a | 1 | FAIL capped | 1200 | 2 | 9000 | 900 |")


def test_autopsy_needs_a_row_for_every_trajectory_it_reports(tmp_path):
    run = _run(tmp_path / "arm", [("go/a", True, False, [_step(10, "")])])
    stray = tmp_path / "arm" / "trajectories" / "go" / "stray.rep1.json"
    stray.write_text(json.dumps({"steps": [_step(99999, REASONING)]}))
    assert [a.task for a in kit_trace.autopsy_run(run)] == ["go/a"]


@pytest.mark.parametrize("text, found", [
    ("Let me write the file now", True), ("I'll implement it", True),
    ("now implement the solution", True), ("the write barrier", False),
])
def test_write_phrase_is_an_announcement_not_a_mention(text, found):
    assert bool(kit_trace._WRITE_PHRASE.search(text)) is found
