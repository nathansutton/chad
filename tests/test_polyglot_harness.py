"""The polyglot kit's harness contract: every CLI arm renders to a real argv, environment
and config files; a CLI trial runs confined to its workspace and throwaway home; the wall
cap takes the whole process group; the lock refuses a drifted version; and chad in
process still writes the rows it always has. No model, no server, no network: the CLI arm
under test is a fake agent in tmp_path, and chad runs on the scripted engine."""
import http.server
import json
import os
import sys
import threading
import time
import tomllib
import uuid

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "polyglot"))

import harnesses  # noqa: E402
import run  # noqa: E402
import workspace  # noqa: E402
from catalog import Task  # noqa: E402
from harness import Solved, Trial, cli  # noqa: E402
from harness.chad_inprocess import ChadInProcess, Loaded  # noqa: E402

from chad import seatbelt  # noqa: E402
from test_agent_e2e import ScriptedEngine, _tool_call  # noqa: E402
from test_polyglot_kit import _exercise  # noqa: E402

needs_sandbox = pytest.mark.skipif(not seatbelt.Seatbelt().probe(),
                                   reason="sandbox-exec cannot enforce from here")

ENDPOINT = cli.Endpoint("http://127.0.0.1:8080", "local-model", 32768, "/models/tok")
# The launching shell: a key and a CHAD_* knob that must never reach a foreign agent.
ENVIRON = {"PATH": "/usr/bin:/bin", "HOME": "/Users/tester", "LANG": "C.UTF-8",
           "OPENAI_API_KEY": "sk-" + "r" * 30, "CHAD_NO_DFLASH": "1"}
GOLD = "def two_fer(name='you'):\n    return f'One for {name}, one for me.'\n"

# The row `--harness chad` has always written, key for key and in order.
CHAD_ROW = ["task", "language", "label", "rep", "passed", "capped", "wall_s", "verify_s",
            "steps", "tool_dispatches", "gen_tokens", "think_tokens", "prefill_tokens",
            "peak_ctx", "gen_time_s", "levers_fired", "thermal", "metrics", "trajectory",
            "test_tail"]
CLI_ROW = ["task", "language", "label", "rep", "passed", "capped", "wall_s", "verify_s",
           "exit_code", "thermal", "metrics", "trajectory", "test_tail"]


def _trial(tmp_path, prompt="solve it", cap=1200):
    return Trial(str(tmp_path / "ws"), str(tmp_path / "home"), prompt, cap,
                     str(tmp_path / "t.json"), str(tmp_path / "out.log"))


def _arm(name, tmp_path):
    return cli.CliHarness(harnesses.SPECS[name], ENDPOINT, None, environ=ENVIRON,
                          root=str(tmp_path))


RUNNABLE = [s.name for s in harnesses.SPECS.values() if not s.unsupported]
_PARSE = {".json": json.loads, ".yaml": yaml.safe_load, ".yml": yaml.safe_load,
          ".toml": tomllib.loads}


@pytest.mark.parametrize("name", RUNNABLE)
def test_every_arm_renders_to_an_argv_an_env_and_parseable_config(tmp_path, name):
    arm = _arm(name, tmp_path)
    trial = _trial(tmp_path)
    argv, env = arm.argv(trial), arm.env(trial.workspace, trial.home)
    files = arm.render(trial.home)
    for rel, text in files.items():
        assert "${" not in text
        _PARSE[os.path.splitext(rel)[1]](text)
    assert argv[0] == harnesses.SPECS[name].binary and "solve it" in argv
    assert env["HOME"] == trial.home and env["PWD"] == trial.workspace
    assert env["TMPDIR"] == os.path.join(trial.home, "tmp")
    assert "CHAD_NO_DFLASH" not in env
    # The only API key an arm sees is the placeholder its own entry sets.
    assert env.get("OPENAI_API_KEY", cli.API_KEY) == cli.API_KEY


def test_argv_snapshots(tmp_path):
    trial = _trial(tmp_path)
    assert _arm("pi", tmp_path).argv(trial) == [
        "pi", "-p", "solve it", "--provider", "llama", "--model", "local-model", "-a"]
    assert _arm("chad-llama", tmp_path).argv(trial) == [
        "chad", "solve it", "--yolo", "--backend", "llama",
        "--base-url", "http://127.0.0.1:8080", "--tokenizer", "/models/tok"]
    assert _arm("cline", tmp_path).argv(trial)[-2:] == ["-t", "1170"]
    assert _arm("opencode", tmp_path).argv(trial)[5:7] == ["--dir", trial.workspace]


def test_config_snapshots(tmp_path):
    home = str(tmp_path / "home")
    opencode = json.loads(_arm("opencode", tmp_path).render(home)[".config/opencode/opencode.json"])
    llama = opencode["provider"]["llama"]
    assert llama["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert llama["models"]["local-model"]["limit"] == {"context": 32768, "output": 8192}
    crush = json.loads(_arm("crush", tmp_path).render(home)[".config/crush/crush.json"])
    assert crush["$schema"] == "https://charm.land/crush.json"
    assert crush["permissions"] == {"skip_requests": True}
    assert crush["providers"]["llama"]["models"][0]["context_window"] == 32768
    codex = tomllib.loads(_arm("codex", tmp_path).render(home)[".codex/config.toml"])
    assert codex["model_providers"]["llama"]["base_url"] == "http://127.0.0.1:8080/v1"
    goose = yaml.safe_load(_arm("goose", tmp_path).render(home)[".config/goose/config.yaml"])
    assert (goose["OPENAI_HOST"], goose["GOOSE_MODE"]) == ("http://127.0.0.1:8080", "auto")
    mini = json.loads(_arm("mini", tmp_path).render(home)["litellm_registry.json"])
    assert mini["openai/local-model"]["max_input_tokens"] == 32768
    assert _arm("mini", tmp_path).env(home, home)["LITELLM_MODEL_REGISTRY_PATH"] == \
        os.path.join(home, "litellm_registry.json")
    assert _arm("chad-llama", tmp_path).env(home, home)["CHAD_TRAJECTORY_JSON"] == \
        os.path.join(home, "trajectory.json")


def test_the_profile_allows_writes_to_the_trial_and_nowhere_personal(tmp_path):
    ws, home = tmp_path / "w", tmp_path / 'h"ome'
    text = cli.profile(str(ws), str(home))
    assert "(deny file-write*)" in text
    assert f'(subpath "{os.path.realpath(ws)}")' in text
    assert f'(subpath "{os.path.realpath(tmp_path)}/h\\"ome")' in text
    assert os.path.expanduser("~") + '"' not in text


def test_unsupported_arms_are_refused_with_their_reason(tmp_path):
    for spec in harnesses.SPECS.values():
        if spec.unsupported:
            assert not spec.argv
            with pytest.raises(cli.HarnessError, match="unsupported"):
                cli.CliHarness(spec, ENDPOINT, None, sandbox_ok=lambda: True).start()


def test_a_cli_arm_is_never_run_unconfined(tmp_path):
    arm = cli.CliHarness(harnesses.SPECS["pi"], ENDPOINT, None, sandbox_ok=lambda: False)
    with pytest.raises(cli.HarnessError, match="confined or not at all"):
        arm.start()


def test_the_lock_round_trips_and_pins_only_lockable_arms(tmp_path):
    pins = {"pi": cli.Pin("0.80.3", "ab" * 32, "npm i -g pi@0.80.3")}
    harnesses.write_lock(pins, str(tmp_path / "lock"))
    assert harnesses.load_lock(str(tmp_path / "lock")) == pins
    committed = harnesses.load_lock()
    assert committed, "harnesses.lock is empty"
    for name, pin in committed.items():
        spec = harnesses.SPECS[name]
        assert spec.locked and not spec.unsupported
        assert pin.version in pin.install and len(pin.sha256) == 64


def test_server_context_reads_the_window_llama_server_reports():
    class Props(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"default_generation_settings": {"n_ctx": 65536}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Props)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert cli.server_context(f"http://127.0.0.1:{server.server_port}") == 65536
    finally:
        server.shutdown()
        server.server_close()
    with pytest.raises(cli.HarnessError, match="no llama-server"):
        cli.server_context(f"http://127.0.0.1:{server.server_port}", timeout_s=0.5)


# -- a fake agent, end to end ------------------------------------------------------

FAKE = '''#!{python}
"""A stand-in coding agent: reads the config prepare() wrote, copies the reference
solution in, and tries the writes the sandbox has to allow and deny."""
import json, os, shutil, subprocess, sys, time
if sys.argv[1:] == ["--version"]:
    print("This is fake-agent version 1.2.3.")
    sys.exit(0)
with open(os.path.join(os.environ["HOME"], ".fake", "config.json")) as f:
    config = json.load(f)
print("home:", os.environ["HOME"])
print("env:", " ".join(sorted(os.environ)), flush=True)    # before any wall-cap kill
if config.get("survivor"):
    subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(1.5); "
                      "open(sys.argv[1], 'w').write('outlived the trial')", config["survivor"]])
    time.sleep(30)
shutil.copyfile(config["gold"], "two_fer.py")
with open(os.path.join(os.environ["HOME"], "state.txt"), "w") as f:
    f.write("the home is writable")
try:
    with open(config["outside"], "w") as f:
        f.write("escaped")
except OSError as e:
    print("outside write refused:", e.strerror)
sys.exit(subprocess.run(["sh", "run-tests.sh"]).returncode)
'''


def _fake(tmp_path, config, version="1.2.3"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "fake-agent"
    exe.write_text(FAKE.replace("{python}", sys.executable))
    exe.chmod(0o755)
    spec = cli.CliSpec(name="fake", binary="fake-agent", install="pip install fake==${version}",
                       argv=("${prompt}",), env={"FAKE_ORIGIN": "${origin}"},
                       files={".fake/config.json": json.dumps(config)})
    environ = {**ENVIRON, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
               "HOME": os.path.expanduser("~")}
    pin = cli.Pin(version, cli.entry_sha256(str(exe)), f"pip install fake=={version}")
    return cli.CliHarness(spec, ENDPOINT, pin, environ=environ, root=str(tmp_path / "cli"))


@needs_sandbox
def test_a_cli_arm_solves_a_trial_confined_to_its_workspace_and_home(tmp_path):
    task = _exercise(tmp_path / "up")
    outside = os.path.join(REPO, f".sandbox-probe-{uuid.uuid4().hex}")
    arm = _fake(tmp_path, {"gold": os.path.join(task.source, ".meta", "example.py"),
                           "outside": outside})
    block = run.Block("arm", arm, wall_cap=120, runs=str(tmp_path / "_runs"))
    try:
        meta = arm.start()
        record = block.solve(task, 1)
    finally:
        escaped = os.path.exists(outside)
        if escaped:
            os.remove(outside)
    assert not escaped
    assert list(record) == CLI_ROW
    assert (record["passed"], record["capped"], record["exit_code"]) == (True, False, 0)
    assert record["trajectory"] is None
    assert arm.version() == "1.2.3" and meta["harness_sha256"] == arm.pin.sha256

    trial = block.trial(task, 1)
    assert trial.workspace.startswith(str(tmp_path / "cli" / "_work"))
    with open(trial.output, encoding="utf-8") as f:
        log = f.read()
    assert f"home: {trial.home}\n" in log
    assert "outside write refused: Operation not permitted" in log
    names = set(next(ln for ln in log.splitlines() if ln.startswith("env:")).split()[1:])
    assert {"HOME", "TMPDIR", "XDG_CONFIG_HOME", "PWD", "CARGO_HOME", "FAKE_ORIGIN"} <= names
    assert not names & {"OPENAI_API_KEY", "CHAD_NO_DFLASH"}
    assert os.path.exists(os.path.join(trial.home, "state.txt"))


@needs_sandbox
def test_the_wall_cap_kills_the_whole_process_group(tmp_path):
    task = _exercise(tmp_path / "up")
    survivor = tmp_path / "survivor.txt"
    arm = _fake(tmp_path, {"survivor": str(survivor)})
    block = run.Block("arm", arm, wall_cap=1, runs=str(tmp_path / "_runs"))
    arm.start()
    record = block.solve(task, 1)
    time.sleep(1.0)                    # past the moment the helper would have written
    assert record["capped"] is True and record["exit_code"] < 0
    assert "home:" in record["output_tail"]
    assert not survivor.exists()


@needs_sandbox
def test_start_refuses_a_version_the_lock_does_not_pin(tmp_path):
    arm = _fake(tmp_path, {})
    arm.pin = cli.Pin("9.9.9", "0" * 64, "pip install fake==9.9.9")
    with pytest.raises(cli.HarnessError, match="harnesses.lock pins 9.9.9: pip install"):
        arm.start()
    arm.pin = None
    with pytest.raises(cli.HarnessError, match="no entry in harnesses.lock"):
        arm.start()


# -- chad in process, through the same loop ---------------------------------------

def test_chad_in_process_writes_the_row_it_always_has(tmp_path):
    task = _exercise(tmp_path / "up")
    label, runs = "arm", tmp_path / "_runs"
    ws = os.path.join(str(tmp_path / "kit"), "_work", label, "rep1", "python", "two-fer")
    script = [_tool_call("write", path=os.path.join(ws, "two_fer.py"), content=GOLD),
              _tool_call("bash", command="sh run-tests.sh"),
              _tool_call("done", summary="implemented two_fer")]
    arm = ChadInProcess("scripted", thinking=False, max_steps=10,
                        load=lambda _model, _thinking: Loaded(ScriptedEngine(script), 24000, 0.0),
                        root=str(tmp_path / "kit"))
    assert run.Block(label, arm, wall_cap=120, runs=str(runs)).run([task], reps=1) == 0

    with open(runs / label / "trials.jsonl", encoding="utf-8") as f:
        record = json.loads(f.read())
    assert list(record) == CHAD_ROW
    assert record["passed"] is True and record["tool_dispatches"] >= 2
    assert record["trajectory"] == "trajectories/python/two-fer.rep1.json"
    assert os.path.exists(runs / label / record["trajectory"])
    meta = json.loads((runs / label / "meta.json").read_text())
    assert (meta["harness"], meta["model"], meta["ctx_limit"]) == ("chad", "scripted", 24000)


def test_the_row_frame_is_the_same_for_every_arm():
    task = Task(name="go/x", language="go", slug="x", source="/nowhere", solution=("x.go",),
                tests=("x_test.go",), example=(".meta/example.go",))
    verdict = workspace.Verdict(False, False, 2.0, "FAIL")
    record = run.row(task, "arm", 2, Solved(3.04, True, {"exit_code": -9}), verdict,
                     {"final_metrics": {"total_steps": 4}}, "trajectories/go/x.rep2.json", "")
    assert list(record) == CLI_ROW
    assert record["metrics"] == {"total_steps": 4} and record["test_tail"] == "FAIL"
    assert (record["wall_s"], record["capped"], record["exit_code"]) == (3.0, True, -9)
    assert record["trajectory"] == "trajectories/go/x.rep2.json"
    assert run.row(task, "arm", 2, Solved(1.0, False, {}), verdict, None, "t.json", "")[
        "trajectory"] is None
