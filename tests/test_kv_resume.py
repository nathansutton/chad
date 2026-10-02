"""A resumed session restores its KV checkpoint instead of re-reading the transcript.

Only the message list used to survive a session, so `chad -c` re-prefilled everything it
restored: at the ~100 tok/s the shipped model reads, a 20k-token conversation was three
minutes of silence before the first new token. A session that ends cleanly now writes its
live cache next to the warm-prefix files and records where in the session file
(`meta["kv"]`, a `KVCheckpointRef`); the resumed Agent keeps the system prompt that
cache was built under and, on its first turn, asks the engine to restore it.

The engine side (what a checkpoint holds, which refs it refuses) is in
test_warm_prefix_tiers.py over real KV layers. This file drives the Agent on a scripted
engine, so it covers the wiring: which system prompt a resumed Agent renders, when the
restore is attempted, what happens on a miss, and what `save(kv=True)` records.
"""
import os

from chad import session
from chad.agent import Agent
from chad.base_engine import KVCheckpointRef
from chad.prompt import build_system_prompt
from test_agent_e2e import ScriptedEngine, _tool_call


class _ResumableEngine(ScriptedEngine):
    """A scripted engine with a cache dir, so the Agent's warm-start block runs, and a
    settable answer to `restore_kv`. A restore that succeeds installs the ids it was
    given, as the real engine does; `save_kv` hands back whatever ref the test set."""

    def __init__(self, script, cache_dir, restore_result=True, ref_to_save=None):
        super().__init__(script)
        self.cache_dir = cache_dir
        self.restore_result = restore_result
        self.ref_to_save = ref_to_save
        self.restore_calls = []
        self.warm_calls = []

    def restore_kv(self, ref, prefix_ids):
        self.restore_calls.append((ref, list(prefix_ids)))
        if self.restore_result:
            self._cached_ids = list(prefix_ids[: ref.tokens])
        return self.restore_result

    def save_kv(self):
        return self.ref_to_save

    def warm_prefix(self, prefix_ids, should_stop=None, head_ids=None):
        self.warm_calls.append(len(self._cached_ids))
        return "skip", 0


SAVED_PROMPT = "the system prompt this conversation ran under"
EARLIER = [{"role": "system", "content": SAVED_PROMPT},
           {"role": "user", "content": "what does session.py do?"},
           {"role": "assistant", "content": "It persists conversations per directory."}]


def _ref(tmp_path, tokens=5, present=True):
    path = tmp_path / "kv" / "sess-abc.safetensors"
    if present:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"not read by the scripted engine")
    return KVCheckpointRef(path=str(path), tokens=tokens, sha="feed")


def _question_turn(tmp_path):
    """A read-only question whose scripted turn ends on a plain answer."""
    target = tmp_path / "data.txt"
    target.write_text("42\n")
    return [_tool_call("read", path=str(target)), "The file contains 42."]


def _resumed(tmp_path, monkeypatch, ref, restore_result=True):
    monkeypatch.chdir(tmp_path)
    eng = _ResumableEngine(_question_turn(tmp_path), str(tmp_path / "kv"),
                           restore_result=restore_result)
    notes = []
    agent = Agent(eng, mode="yolo", thinking=False, resume=EARLIER, resume_kv=ref,
                  emit=lambda kind, text: notes.append((kind, text)))
    return agent, eng, notes


def test_a_resume_with_a_checkpoint_keeps_the_saved_prompt_and_restores_first(
        tmp_path, monkeypatch):
    ref = _ref(tmp_path)
    agent, eng, notes = _resumed(tmp_path, monkeypatch, ref)
    # The cache was built under the saved prompt; a rebuilt one would not be its prefix.
    assert agent.messages[0]["content"] == SAVED_PROMPT
    assert [m["content"] for m in agent.messages[1:]] == [m["content"] for m in EARLIER[1:]]

    agent.run_turn("what's in data.txt?")

    # Restored before the warm prefix was consulted, against a render of the transcript
    # that is at least as long as the checkpoint; the warm start then saw the populated
    # cache (and adopts the warm files) instead of prefilling the prefix again.
    assert len(eng.restore_calls) == 1
    got_ref, ids = eng.restore_calls[0]
    assert got_ref == ref and len(ids) >= ref.tokens
    assert eng.warm_calls == [ref.tokens]
    assert agent._resume_kv is None
    assert any("resumed warm: 5 tokens" in text for _kind, text in notes)
    assert agent.messages[0]["content"] == SAVED_PROMPT


def test_a_checkpoint_the_transcript_no_longer_matches_falls_back_to_a_fresh_prompt(
        tmp_path, monkeypatch):
    ref = _ref(tmp_path)
    agent, eng, notes = _resumed(tmp_path, monkeypatch, ref, restore_result=False)
    assert agent.messages[0]["content"] == SAVED_PROMPT   # kept until the engine decides

    agent.run_turn("what's in data.txt?")

    assert len(eng.restore_calls) == 1
    # Nothing was cached yet, so the swap back to today's prompt cost nothing.
    assert agent.messages[0]["content"] == build_system_prompt()
    assert agent.messages[0]["content"] != SAVED_PROMPT
    assert not any("resumed warm" in text for _kind, text in notes)
    assert eng.warm_calls == [0]      # the ordinary cold warm-start ran


def test_a_checkpoint_whose_file_is_gone_is_a_cold_resume_from_the_start(tmp_path, monkeypatch):
    ref = _ref(tmp_path, present=False)
    agent, eng, _notes = _resumed(tmp_path, monkeypatch, ref)
    assert agent._resume_kv is None
    assert agent.messages[0]["content"] == build_system_prompt()
    agent.run_turn("what's in data.txt?")
    assert eng.restore_calls == []


def test_a_resume_without_a_checkpoint_is_unchanged(tmp_path, monkeypatch):
    agent, eng, _notes = _resumed(tmp_path, monkeypatch, None)
    assert agent.messages[0]["content"] == build_system_prompt()
    assert [m["content"] for m in agent.messages[1:]] == [m["content"] for m in EARLIER[1:]]
    agent.run_turn("what's in data.txt?")
    assert eng.restore_calls == []


def test_the_opt_out_resumes_cold(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_NO_KV_RESUME", "1")
    ref = _ref(tmp_path)
    agent, eng, _notes = _resumed(tmp_path, monkeypatch, ref)
    assert agent._resume_kv is None
    assert agent.messages[0]["content"] == build_system_prompt()
    agent.run_turn("what's in data.txt?")
    assert eng.restore_calls == []


def test_save_with_kv_records_the_ref_and_a_plain_save_does_not(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = _ref(tmp_path, tokens=1234)
    eng = _ResumableEngine([], str(tmp_path / "kv"), ref_to_save=ref)
    agent = Agent(eng, mode="yolo", thinking=False, persist=True)
    agent.messages.append({"role": "user", "content": "hello"})

    agent.save(kv=True)
    assert session.kv_ref(session.load_session(os.getcwd())) == ref

    # Every turn saves the conversation; only the session's end writes the cache. A
    # later plain save of the same session overwrites its file without the ref.
    agent.save()
    assert session.kv_ref(session.load_session(os.getcwd())) is None

    # An engine with nothing to save (cold cache, remote backend) records no ref.
    eng.ref_to_save = None
    agent.save(kv=True)
    assert session.kv_ref(session.load_session(os.getcwd())) is None


def test_kv_ref_parses_only_a_well_formed_entry():
    good = {"path": "/kv/sess-1.safetensors", "tokens": 12, "sha": "ab"}
    assert session.kv_ref({"meta": {"kv": good}}) == KVCheckpointRef("/kv/sess-1.safetensors", 12, "ab")
    assert session.kv_ref({"meta": {}}) is None
    assert session.kv_ref({"messages": []}) is None
    assert session.kv_ref({"meta": {"kv": "sess-1"}}) is None
    assert session.kv_ref({"meta": {"kv": {**good, "tokens": 0}}}) is None
    assert session.kv_ref({"meta": {"kv": {**good, "tokens": "12"}}}) is None
    assert session.kv_ref({"meta": {"kv": {**good, "path": ""}}}) is None
    assert session.kv_ref({"meta": {"kv": {"tokens": 12, "sha": "ab"}}}) is None
