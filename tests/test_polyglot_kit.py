"""The polyglot eval kit's model-free parts: leak-proof workspaces, test restoration, the
paired statistics, and publishing a run without its local paths. Nothing here needs the
upstream checkout, a toolchain, a model or the network — exercises and runs are built in
tmp_path in the layouts upstream and `run.py` use."""
import json
import os
import re
import subprocess
import sys
import tarfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "polyglot"))

import trace as kit_trace  # noqa: E402

import catalog  # noqa: E402
import fetch  # noqa: E402
import publish  # noqa: E402
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


def test_no_run_output_is_tracked():
    out = subprocess.run(["git", "ls-files", "benchmarks"], cwd=REPO, capture_output=True,
                         text=True, check=False)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    run_output = re.compile(r"(^|/)(_runs|_work|_kv|_data|_publish|trajectories)/|trials\.jsonl$")
    tracked = [p for p in out.stdout.splitlines()
               if run_output.search(p) and not p.startswith("benchmarks/stock/_runs/")]
    assert tracked == [], "run output belongs in a published bundle (publish.py), not in git"


ROOTS = publish.Roots(home="/Users/tester", repo="/Users/tester/repo/chad",
                      kit="/Users/tester/repo/chad/benchmarks/polyglot")
WS = ROOTS.kit + "/_work/arm/rep1/go/bob"


def _run_dir(root, message=f"cd {WS} && go test ./...", tail=""):
    """A finished run in `run.py`'s layout, embedding the paths a real one does."""
    run_dir = root / "arm"
    (run_dir / "trajectories" / "go").mkdir(parents=True)
    rows = [{"task": "go/bob", "language": "go", "rep": 1, "passed": True, "capped": False,
             "wall_s": 12.0, "gen_tokens": 100, "test_tail": ""},
            {"task": "go/two", "language": "go", "rep": 1, "passed": False, "capped": True,
             "wall_s": 1199.8, "gen_tokens": 900,
             "test_tail": tail or f"{WS}/bob_test.go:9: want 2 "
                                  f"(full output: {ROOTS.home}/.cache/chad/spill/1/bash-1.log)"}]
    (run_dir / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "meta.json").write_text(json.dumps({
        "label": "arm", "started": "2026-09-21T17:26:45", "chad_version": "2.2.0",
        "git_rev": "abc1234", "git_dirty": False, "model": ROOTS.repo + "/models/Qwen"}))
    (run_dir / "trajectories" / "go" / "bob.rep1.json").write_text(
        json.dumps({"steps": [{"source": "agent", "step_id": 1, "message": message}]}))
    return run_dir


def test_publish_rewrites_every_local_path(tmp_path):
    out = tmp_path / "out"
    b = publish.bundle(str(_run_dir(tmp_path)), str(out), ROOTS, with_trajectories=True)
    rows = (out / "trials.jsonl").read_text()
    assert "/Users/" not in rows
    assert "./bob_test.go:9" in rows and "~/.cache/chad/spill/1/bash-1.log" in rows
    assert json.loads((out / "meta.json").read_text())["model"] == "<repo>/models/Qwen"
    with tarfile.open(out / publish.TRAJECTORIES) as tar:
        assert tar.getnames() == ["trajectories/go/bob.rep1.json"]
        member = tar.extractfile("trajectories/go/bob.rep1.json")
        assert member is not None and b"cd . && go test" in member.read()
    assert b.runs_row == ("| arm | 2026-09-21 | 2.2.0 (abc1234) | chad | Qwen | 2 × 1 | "
                          "hf://datasets/nathansutton/chad-polyglot-runs/polyglot/arm | "
                          f"{b.sha256} |")


def test_a_cli_trial_is_rewritten_wherever_its_root_is():
    """A CLI arm's trials live in the system temp directory, not under the kit."""
    root = "/private/var/folders/ab/cd_ef/T/chad-polyglot"
    text = (f"cmake {root}/_work/pi-pool/rep2/cpp/bank-account/build && "
            f"cat {root}/_home/pi-pool/rep2/cpp/bank-account/.pi/agent/models.json; "
            "grep -rn x benchmarks/polyglot/_work/arm/rep1/go/bob")
    assert publish.redact(text, ROOTS) == (
        "cmake ./build && cat <home>/.pi/agent/models.json; "
        "grep -rn x benchmarks/polyglot/_work/arm/rep1/go/bob")
    assert not publish.problems(publish.redact(text, ROOTS), ROOTS)


def test_rows_alone_unless_trajectories_are_asked_for(tmp_path):
    publish.bundle(str(_run_dir(tmp_path)), str(tmp_path / "out"), ROOTS)
    assert sorted(os.listdir(tmp_path / "out")) == ["meta.json", "trials.jsonl"]


def test_a_prefix_is_only_rewritten_whole():
    assert publish.redact("/Users/testerson/notes.txt", ROOTS) == "/Users/testerson/notes.txt"
    assert publish.problems("/Users/testerson/notes.txt", ROOTS)
    assert not publish.problems("the earlier `ls -la /Users/.../bob/` failed", ROOTS)


@pytest.mark.parametrize("message, reason", [
    (f"cat {ROOTS.home}/.ssh/config", "paths outside the workspace: ~/.ssh/config"),
    ("ls /Users/someone-else/src", "an absolute home path survived"),
    ("export HF_TOKEN=hf_" + "x1" * 17, "a credential-shaped string"),
])
def test_publish_refuses_what_must_not_leave_the_machine(tmp_path, message, reason):
    with pytest.raises(publish.PublishError, match=re.escape(reason)) as refused:
        publish.bundle(str(_run_dir(tmp_path, message)), str(tmp_path / "out"), ROOTS,
                       with_trajectories=True)
    assert "trajectories/go/bob.rep1.json" in str(refused.value)
    assert not (tmp_path / "out").exists()


def _ledger(tmp_path, b, sha256=None):
    row = b.runs_row.replace("hf://datasets/nathansutton/chad-polyglot-runs/polyglot/arm",
                             f"file://{b.path}")
    if sha256:
        row = row.replace(b.sha256, sha256)
    path = tmp_path / "RUNS.md"
    path.write_text(f"# Published runs\n\n{publish.RUNS_HEADER}\n|---|---|---|---|---|---|---|---|\n"
                    f"{row}\n")
    return fetch.read_ledger(str(path))


def test_fetch_round_trips_a_bundle_to_the_same_score(tmp_path):
    run_dir = _run_dir(tmp_path / "local")
    b = publish.bundle(str(run_dir), str(tmp_path / "bundle"), ROOTS, with_trajectories=True)
    dest = fetch.fetch(_ledger(tmp_path, b)["arm"], str(tmp_path / "runs"))
    assert stats.score_report(stats.load_trials(os.path.join(dest, "trials.jsonl"))) == \
        stats.score_report(stats.load_trials(str(run_dir / "trials.jsonl")))
    assert os.path.exists(os.path.join(dest, "trajectories", "go", "bob.rep1.json"))
    with pytest.raises(fetch.FetchError, match="already exists"):
        fetch.fetch(_ledger(tmp_path, b)["arm"], str(tmp_path / "runs"))


def test_fetch_refuses_rows_that_are_not_the_ones_recorded(tmp_path):
    b = publish.bundle(str(_run_dir(tmp_path / "local")), str(tmp_path / "bundle"), ROOTS)
    entry = _ledger(tmp_path, b, sha256="0" * 64)["arm"]
    with pytest.raises(fetch.FetchError, match="sha256"):
        fetch.fetch(entry, str(tmp_path / "runs"))
    assert not (tmp_path / "runs" / "arm").exists()


def test_the_committed_ledger_has_the_columns_fetch_reads():
    with open(fetch.LEDGER, encoding="utf-8") as f:
        assert publish.RUNS_HEADER in f.read()
    assert "design-sample" in fetch.read_ledger(fetch.LEDGER)
