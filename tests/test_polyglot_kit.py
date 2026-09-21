"""The polyglot eval kit's model-free parts: leak-proof workspaces, test restoration, and
the paired statistics. Nothing here needs the upstream checkout, a toolchain or a model —
exercises are built in tmp_path in upstream's layout."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks", "polyglot"))

import trace as kit_trace  # noqa: E402

import catalog  # noqa: E402
import run  # noqa: E402
import stats  # noqa: E402
import workspace  # noqa: E402


def _exercise(root, language="python", slug="two-fer", files=None, config=None):
    source = root / language / "exercises" / "practice" / slug
    defaults = {
        "two_fer.py": "def two_fer(name='you'):\n    pass\n",
        "two_fer_test.py": "from two_fer import two_fer\n\n"
                           "def test_it():\n    assert two_fer() == 'One for you, one for me.'\n",
        ".meta/example.py": "def two_fer(name='you'):\n    return f'One for {name}, one for me.'\n",
        ".docs/instructions.md": "Say who gets one.",
        ".docs/hints.md": "HINT: use an f-string",
        ".approaches/intro.md": "the answer, in prose",
    }
    for rel, text in {**defaults, **(files or {})}.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (source / ".meta" / "config.json").write_text(json.dumps({"files": config or {
        "solution": ["two_fer.py"], "test": ["two_fer_test.py"],
        "example": [".meta/example.py"]}}))
    return catalog._read_task(language, slug, str(source))


def test_workspace_leaves_the_answer_behind(tmp_path):
    task = _exercise(tmp_path / "up")
    ws = tmp_path / "ws"
    workspace.materialize(task, str(ws))
    present = {os.path.relpath(os.path.join(d, f), ws) for d, _, fs in os.walk(ws) for f in fs}
    assert present == {"two_fer.py", "two_fer_test.py", workspace.SCRIPT}


def test_prompt_carries_instructions_but_never_hints(tmp_path):
    task = _exercise(tmp_path / "up")
    prompt = run.prompt_for(task)
    assert "Say who gets one." in prompt and "HINT" not in prompt
    assert workspace.SCRIPT in prompt and "`two_fer.py`" in prompt


def test_verify_fails_the_stub_and_passes_gold(tmp_path):
    task = _exercise(tmp_path / "up")
    ws = str(tmp_path / "ws")
    workspace.materialize(task, ws)
    assert not workspace.verify(task, ws).passed
    workspace.apply_gold(task, ws)
    assert workspace.verify(task, ws).passed


def test_verify_restores_tampered_tests_and_script(tmp_path):
    task = _exercise(tmp_path / "up")
    ws = tmp_path / "ws"
    workspace.materialize(task, str(ws))
    (ws / "two_fer_test.py").write_text("def test_it():\n    assert True\n")
    (ws / workspace.SCRIPT).write_text("#!/bin/sh\nexit 0\n")
    assert not workspace.verify(task, str(ws)).passed
    assert "One for you" in (ws / "two_fer_test.py").read_text()


def test_skipped_tests_are_switched_on(tmp_path):
    js = _exercise(tmp_path / "up", "javascript", "bob", files={
        "bob.spec.js": "test('a', () => {});\nxtest('b', () => {});\nxdescribe('c', () => {});\n",
        "bob.js": "", ".meta/proof.ci.js": ""},
        config={"solution": ["bob.js"], "test": ["bob.spec.js"], "example": [".meta/proof.ci.js"]})
    ws = tmp_path / "js"
    workspace.materialize(js, str(ws))
    assert (ws / "bob.spec.js").read_text() == \
        "test('a', () => {});\ntest('b', () => {});\ndescribe('c', () => {});\n"

    java = _exercise(tmp_path / "up", "java", "bob", files={
        "src/test/java/BobTest.java": "    @Test\n    @Disabled(\"Remove to run test\")\n    void b() {}\n",
        "src/main/java/Bob.java": "", ".meta/src/reference/java/Bob.java": "class Bob {}"},
        config={"solution": ["src/main/java/Bob.java"], "test": ["src/test/java/BobTest.java"],
                "example": [".meta/src/reference/java/Bob.java"]})
    ws = tmp_path / "java"
    workspace.materialize(java, str(ws))
    assert (ws / "src/test/java/BobTest.java").read_text() == "    @Test\n    void b() {}\n"
    workspace.apply_gold(java, str(ws))
    assert (ws / "src/main/java/Bob.java").read_text() == "class Bob {}"


def test_resume_skips_recorded_trials(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text('{"task": "go/bob", "rep": 1}\n{"task": "go/bob", "rep": 2}\n')
    assert run.done_keys(str(path)) == {("go/bob", 1), ("go/bob", 2)}
    assert run.done_keys(str(tmp_path / "absent.jsonl")) == set()


def _trials(outcomes, tokens=100):
    return [stats.Trial(task, "python", passed, 10.0, tokens, False)
            for task, results in outcomes.items() for passed in results]


def test_sign_test_matches_hand_computed_values():
    assert stats.sign_test(0, 0) == 1.0
    assert stats.sign_test(5, 0) == pytest.approx(2 / 32)
    assert stats.sign_test(6, 0) == pytest.approx(2 / 64)
    assert stats.sign_test(7, 1) == pytest.approx(2 * (1 + 8) / 256)
    assert stats.sign_test(4, 4) == 1.0
    assert stats.flips_needed(0.05) == 6


def test_compare_pairs_by_task_and_ignores_unshared():
    a = _trials({"t1": [False], "t2": [True], "t3": [False], "only_a": [True]})
    b = _trials({"t1": [True], "t2": [True], "t3": [False], "only_b": [False]}, tokens=150)
    c = stats.compare(a, b)
    assert (c.tasks, c.up, c.down) == (3, 1, 0)
    assert c.up_tasks == ("t1",) and c.token_ratio == pytest.approx(1.5)
    assert c.rate_a == pytest.approx(1 / 3) and c.rate_b == pytest.approx(2 / 3)


def test_compare_uses_per_task_rates_across_reps():
    a = _trials({"t1": [True, False, False], "t2": [True, True, True]})
    b = _trials({"t1": [True, True, False], "t2": [True, True, False]})
    c = stats.compare(a, b)
    assert (c.up, c.down, c.p_value) == (1, 1, 1.0)


def test_pool_is_the_frontier_and_needs_reps():
    trials = _trials({"always": [True, True], "never": [False, False], "mixed": [True, False]})
    assert stats.pool(trials) == ["mixed"]
    assert stats.pool(trials, include_never=True) == ["mixed", "never"]
    with pytest.raises(ValueError, match="at least 2 reps"):
        stats.pool(_trials({"t": [True]}))


def _step(name, prompt, cached, command=""):
    return {"source": "agent", "step_id": 3,
            "tool_calls": [{"function_name": name,
                            "arguments": {"command": command} if command else {}}],
            "metrics": {"prompt_tokens": prompt, "cached_tokens": cached,
                        "completion_tokens": 100,
                        "extra": {"think_tokens": 40, "prefill_s": 1.0, "gen_s": 4.0}}}


def _write_run(root):
    for lang, steps in (("go", [_step("bash", 1000, 900, "grep -rn Score . | head"),
                                _step("bash", 2000, 1900, "go test ./... | tail -5")]),
                        ("rust", [_step("edit", 1000, 900)])):
        path = root / "trajectories" / lang / "x.rep1.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"steps": [{"source": "user"}, *steps]}))


def test_profile_pools_tool_mix_and_cache_share_across_a_run(tmp_path):
    _write_run(tmp_path)
    docs = [doc for _, doc in kit_trace.load_run(str(tmp_path))]
    report = kit_trace.profile(docs, [kit_trace.tool_counts(doc) for doc in docs])
    assert "2 trials, 3 tool calls, 3 model steps" in report
    assert "  bash                2  66.7%" in report
    assert "from cache 3700 (92.5%), prefilled 300" in report
    assert "120 of them thinking (40%)" in report


def test_tool_counts_attribute_a_pipeline_to_its_first_command():
    doc = {"steps": [_step("bash", 1, 0, "grep -rn x . | head"),
                     _step("bash", 1, 0, "pytest -q | tail -5"),
                     _step("bash", 1, 0, "cd src && cat main.go")]}
    assert kit_trace.tool_counts(doc) == {"bash": 3, "bash:search": 1, "bash:read": 1}


def test_counts_rows_profile_without_their_trajectories(tmp_path):
    _write_run(tmp_path)
    rows = tmp_path / "counts.jsonl"
    rows.write_text("".join(json.dumps({"trajectory": path, "calls": kit_trace.tool_counts(doc)}) + "\n"
                            for path, doc in kit_trace.load_run(str(tmp_path))))
    report = kit_trace.profile([], kit_trace.load_counts(str(rows)))
    assert report.splitlines()[0] == "2 trials, 3 tool calls"
    assert "prompt tokens" not in report
