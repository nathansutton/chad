"""Tier-1 (no audio hardware, no model) tests for voice mode's WIRING.

tests/test_speech.py covers the pure text layer — what a reply sounds like, how
a misheard word is remapped, what silence gets trimmed. This file covers the
part that actually coordinates: the /speech enable/disable lifecycle, the
push-to-talk state machine, the three-thread handoff, and every error path a
user can reach. That is where the bugs live; pure functions were never going to
break.

Nothing here imports sounddevice. Importing it INITIALIZES PortAudio (a
CoreAudio client thread that can crash interpreter teardown when it was started
for no reason), which is exactly why speech.available() probes with find_spec
instead of importing. The TUI takes its speech module, recorder and speaker as
constructor arguments, so fakes of all three reach every branch without touching
a device, and the Recorder's own ring/take logic is unit-tested for real in
test_speech.py.

What is deliberately NOT here: audio in -> transcript out. That needs a mic and
a human ear, and lives in the manual pass.
"""
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad import speech  # noqa: E402
from chad.engine import Engine  # noqa: E402
from chad.tui import TUI  # noqa: E402

_JOIN = 5.0  # seconds to wait on a worker thread before calling it hung


def _fake_engine():
    eng = object.__new__(Engine)
    eng.model_id = "test/model"
    eng.kv_bytes_per_token = 20_000.0
    return eng


class _FakeRecorder:
    """Stands in for speech.Recorder: same surface, no PortAudio. `audio` is
    what stop() hands back, so a test can decide what was "said"."""

    PRE_ROLL_S = speech.Recorder.PRE_ROLL_S

    def __init__(self, audio="AUDIO", open_raises=None, start_raises=None):
        self.recording = False
        self.stream_open = False
        self.take_full = False
        self.closed = False
        self.cancelled = False
        self.audio = audio
        self._open_raises = open_raises
        self._start_raises = start_raises

    def open_stream(self):
        if self._open_raises:
            raise self._open_raises
        self.stream_open = True

    def start(self):
        if self._start_raises:
            raise self._start_raises
        self.recording = True

    def stop(self):
        self.recording = False
        return self.audio

    def cancel(self):
        self.recording = False
        self.cancelled = True

    def close(self):
        self.closed = True
        self.stream_open = False
        self.recording = False


class _FakeSpeaker:
    def __init__(self):
        self.said = []
        self.stops = 0

    def speak(self, text):
        self.said.append(text)

    def stop(self):
        self.stops += 1


class _FakeSpeech:
    """Stands in for the chad.speech module as the TUI calls it: every probe
    answers what the test configured, transcribe() runs the test's function, and
    release_model() counts how often the weights were handed back."""

    def __init__(self, *, available=(True, ""), tts=(True, ""), stt=(True, ""),
                 remaps_error=None, cached=True, transcribe=None,
                 max_take_s=speech.MAX_TAKE_S):
        self.MAX_TAKE_S = max_take_s
        self.releases = 0
        self._available = available
        self._tts = tts
        self._stt = stt
        self._remaps_error = remaps_error
        self._cached = cached
        self._transcribe = transcribe if transcribe is not None else (lambda _audio: "")

    def available(self):
        return self._available

    def tts_status(self):
        return self._tts

    def stt_status(self):
        return self._stt

    def stt_model(self):
        return "test/stt-model"

    def load_remaps(self):
        if self._remaps_error is not None:
            raise self._remaps_error
        return {}

    def remap_path(self):
        return "/nonexistent/speech_words.json"

    def model_cached(self):
        return self._cached

    def release_model(self):
        self.releases += 1
        return True

    def transcribe(self, audio):
        return self._transcribe(audio)


def _tui(*, fake_speech=None, recorder=None, speaker=None):
    """A TUI wired to fakes, with voice mode not yet enabled."""
    rec = recorder if recorder is not None else _FakeRecorder()
    spk = speaker if speaker is not None else _FakeSpeaker()
    t = TUI(_fake_engine(), ctx_limit=24000,
            speech=fake_speech if fake_speech is not None else _FakeSpeech(),
            recorder=rec, speaker=spk)
    return t, rec, spk


def _out(t):
    return "".join(t._pending)


def _status(t):
    return "".join(txt for _s, txt in t._status_fragments())


# -- /speech enable: the four things that can go wrong before you ever talk ---

def test_enable_refuses_and_explains_when_deps_missing():
    fake = _FakeSpeech(available=(False, "speech needs sounddevice — install with X"))
    t, rec, _ = _tui(fake_speech=fake)
    t._toggle_speech()
    assert t.speech_on is False           # refused, not half-on
    assert not rec.stream_open            # and the mic was never opened
    assert "install with X" in _out(t)    # verbatim reason, so it names the fix


def test_enable_surfaces_mic_denial_instead_of_recording_silence():
    # TCC refusal, unplugged device: must fail HERE, loudly, not later as an
    # empty take the user mistakes for "it didn't hear me".
    rec = _FakeRecorder(open_raises=RuntimeError("PortAudioError: device unavailable"))
    t, rec, _ = _tui(recorder=rec)
    t._toggle_speech()
    assert t.speech_on is False
    assert "mic unavailable" in _out(t)
    assert "device unavailable" in _out(t)


def test_enable_reports_bad_voice_and_bad_quant_without_blocking():
    # Both are invisible otherwise: `say` exits nonzero and Speaker swallows it,
    # and stt_quant_bits() only raises from inside transcribe() — i.e. after you
    # have already spoken. Neither should block dictation.
    fake = _FakeSpeech(tts=(False, "CHAD_VOICE='Samanta' is not an installed voice"),
                       stt=(False, "CHAD_STT_QUANT must be 8 (default), 4, or none"))
    t, rec, _ = _tui(fake_speech=fake)
    t._toggle_speech()
    assert t.speech_on is True            # warnings, not a refusal
    assert "not an installed voice" in _out(t)
    assert "CHAD_STT_QUANT must be" in _out(t)


def test_enable_reports_malformed_word_table():
    t, rec, _ = _tui(fake_speech=_FakeSpeech(remaps_error=ValueError("must be a JSON object")))
    t._toggle_speech()
    assert t.speech_on is True            # remaps are optional; voice mode still works
    assert "word remaps IGNORED" in _out(t)


def test_enable_warns_when_weights_are_not_cached():
    t, rec, _ = _tui(fake_speech=_FakeSpeech(cached=False))
    t._toggle_speech()
    assert "aren't cached" in _out(t)     # so a 2.5GB fetch doesn't read as a hang


def test_enable_opens_the_mic_and_says_so_in_the_status_line():
    t, rec, _ = _tui()
    assert "mic open" not in _status(t)
    t._toggle_speech()
    assert t.speech_on is True
    assert rec.stream_open
    # An open mic you cannot see is a bug, not a feature.
    assert "mic open" in _status(t)


# -- /speech disable ---------------------------------------------------------

def test_disable_releases_mic_speaker_and_weights():
    fake = _FakeSpeech()
    t, rec, spk = _tui(fake_speech=fake)
    t._toggle_speech()
    t._toggle_speech()
    assert t.speech_on is False
    assert rec.closed                     # mic fully released, warm ring included
    assert spk.stops >= 1                 # a reply mid-sentence is cut
    assert fake.releases == 1             # ~790MB handed back
    assert "mic open" not in _status(t)


def test_disable_does_not_free_weights_while_a_decode_is_in_flight():
    # The worker holds a live reference to the model inside generate(); freeing
    # it underneath would be a use-after-free at the Metal layer.
    fake = _FakeSpeech()
    t, rec, spk = _tui(fake_speech=fake)
    t._toggle_speech()
    t._speech_phase = "transcribing"
    t._toggle_speech()
    assert t.speech_on is False
    assert rec.closed                     # the mic still goes
    assert fake.releases == 0             # the weights do NOT


# -- push-to-talk: the state machine -----------------------------------------

def test_ctrl_t_starts_a_take_and_hushes_our_own_tts():
    t, rec, spk = _tui()
    t._toggle_speech()
    t._toggle_recording()
    assert rec.recording
    assert t._speech_phase == "recording"
    assert spk.stops >= 1                 # never transcribe our own spoken reply
    assert "● rec" in _status(t)


def test_ctrl_t_twice_transcribes_into_the_input_box_unsent():
    t, rec, spk = _tui(fake_speech=_FakeSpeech(transcribe=lambda a: "fix the failing test"))
    t._toggle_speech()
    t._toggle_recording()                 # start
    t._toggle_recording()                 # stop -> spawns the worker
    _join_stt()
    # The worker hands the text back through the event loop, which isn't running
    # in a unit test — so assert the worker got there and the phase reset.
    assert t._speech_phase == ""
    assert not rec.recording


def test_start_failure_is_reported_and_leaves_no_half_take():
    rec = _FakeRecorder(start_raises=RuntimeError("device unplugged"))
    t, rec, _ = _tui(recorder=rec)
    t._toggle_speech()
    t._toggle_recording()
    assert t._speech_phase == ""          # not stuck showing "● rec"
    assert "mic unavailable" in _out(t)


def test_second_ctrl_t_is_ignored_while_still_decoding():
    t, rec, _ = _tui()
    t._toggle_speech()
    t._speech_phase = "transcribing"
    t._toggle_recording()
    assert rec.recording is False         # one decode at a time; no new take
    assert t._speech_phase == "transcribing"


def test_transcription_failure_resets_the_phase_and_tells_the_user():
    # A stuck spinner is worse than an error: the user waits forever.
    fake = _FakeSpeech(transcribe=lambda a: (_ for _ in ()).throw(RuntimeError("metal oom")))
    t, rec, _ = _tui(fake_speech=fake)
    t._toggle_speech()
    t._toggle_recording()
    t._toggle_recording()
    _join_stt()
    assert t._speech_phase == ""
    assert "transcription failed" in _out(t)
    assert "metal oom" in _out(t)


def test_empty_transcript_says_heard_nothing():
    t, rec, _ = _tui(fake_speech=_FakeSpeech(transcribe=lambda a: ""))
    t._toggle_speech()
    t._toggle_recording()
    t._toggle_recording()
    _join_stt()
    assert "heard nothing" in _out(t)


def test_transcript_is_discarded_when_speech_was_turned_off_mid_decode():
    # The first-use window is long (a 2.5GB download), so this is a real race,
    # not a theoretical one. Text must not appear in a mode the user left.
    gate = threading.Event()

    def slow(_audio):
        gate.wait(_JOIN)
        return "late transcript"

    t, rec, _ = _tui(fake_speech=_FakeSpeech(transcribe=slow))
    t._toggle_speech()
    t._toggle_recording()
    t._toggle_recording()                 # worker now blocked in slow()
    t._toggle_speech()                    # user gives up and turns speech off
    gate.set()
    _join_stt()
    assert "transcript discarded" in _out(t)
    assert "late transcript" not in t.input.buffer.text


# -- the take cap ------------------------------------------------------------

def test_status_line_announces_a_capped_take():
    t, rec, _ = _tui(fake_speech=_FakeSpeech(max_take_s=90))
    t._toggle_speech()
    t._toggle_recording()
    assert "max length" not in _status(t)
    rec.take_full = True
    s = _status(t)
    assert "max length (90s)" in s        # the cap the speech module enforces
    assert "● rec" in s                   # still recording — the mic stays honest


# -- spoken replies ----------------------------------------------------------

def test_reply_is_spoken_with_markup_and_code_stripped():
    t, rec, spk = _tui()
    t._toggle_speech()
    t.agent = type("A", (), {"messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "**Fixed** it.\n```py\nx=1\n```\nTests pass."},
    ]})()
    t._speak_reply()
    assert len(spk.said) == 1
    said = spk.said[0]
    assert "Fixed it." in said and "Tests pass." in said
    assert "*" not in said and "x=1" not in said


def test_empty_final_reply_is_not_spoken():
    t, rec, spk = _tui()
    t._toggle_speech()
    t.agent = type("A", (), {"messages": [{"role": "assistant", "content": ""}]})()
    t._speak_reply()
    assert spk.said == []                 # no process for an empty utterance


def test_no_assistant_message_is_not_spoken():
    t, rec, spk = _tui()
    t._toggle_speech()
    t.agent = type("A", (), {"messages": [{"role": "user", "content": "hi"}]})()
    t._speak_reply()
    assert spk.said == []


def test_shutdown_releases_mic_and_silences_speech():
    t, rec, spk = _tui()
    t._toggle_speech()
    t.app.exit = lambda: None
    t._shutdown_app(None)
    assert rec.closed
    assert spk.stops >= 1


def _join_stt():
    """Wait for the chad-stt worker to finish (it is a daemon thread we do not hold)."""
    deadline = time.monotonic() + _JOIN
    while time.monotonic() < deadline:
        if not any(th.name == "chad-stt" and th.is_alive() for th in threading.enumerate()):
            return
        time.sleep(0.01)
    raise AssertionError("the chad-stt worker never finished")


# === speech.py orchestration gaps the pure-layer file left open ==============

def test_release_model_is_a_noop_when_nothing_is_loaded():
    # No test loads STT weights (real MLX, and a 2.5GB download on first use),
    # so the process-wide cache is empty here.
    assert speech._stt == {}
    assert speech.release_model() is False   # and does not import mlx to say so


def test_release_model_clears_the_cache(monkeypatch):
    fake_mx = type("mx", (), {"clear_cache": staticmethod(lambda: None)})
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setattr(speech, "_stt", {("m", 8): object()})
    assert speech.release_model() is True
    assert speech._stt == {}


def test_model_cached_true_for_a_local_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAD_STT_MODEL", str(tmp_path))
    assert speech.model_cached() is True


def test_model_cached_false_when_the_hub_lookup_fails(monkeypatch):
    monkeypatch.setenv("CHAD_STT_MODEL", "nobody/not-a-real-model-xyz")
    assert speech.model_cached() is False    # -> the TUI prints the download notice


def _fake_say(tmp_path, script, executable=True):
    """A stand-in `say` at a real path. tts_status and Speaker run their real
    subprocess calls against it."""
    path = tmp_path / "say"
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(0o755 if executable else 0o644)
    return str(path)


_VOICE_LIST = ("Samantha           en_US    # Hi, my name is Samantha.\n"
               "Eddy (English (UK)) en_GB   # Hello, my name is Eddy.\n")


def test_tts_status_reports_a_missing_say_binary(tmp_path):
    ok, reason = speech.tts_status(say_bin=str(tmp_path / "say"))
    assert ok is False
    assert "replies won't be spoken" in reason


def test_tts_status_accepts_an_installed_voice_and_rejects_a_typo(monkeypatch, tmp_path):
    # Lists voices only when asked the way tts_status asks (`say -v ?`).
    say = _fake_say(tmp_path, '[ "$1" = "-v" ] && [ "$2" = "?" ] || exit 64\n'
                              "cat <<'EOF'\n" + _VOICE_LIST + "EOF\n")
    monkeypatch.setenv("CHAD_VOICE", "samantha")          # case-insensitive
    assert speech.tts_status(say_bin=say) == (True, "")
    monkeypatch.setenv("CHAD_VOICE", "Eddy (English (UK))")  # spaces in the name
    assert speech.tts_status(say_bin=say) == (True, "")
    monkeypatch.setenv("CHAD_VOICE", "Samanta")           # the typo
    ok, reason = speech.tts_status(say_bin=say)
    assert ok is False
    assert "Samanta" in reason and "Samantha" in reason    # names the near miss


def test_tts_status_stays_quiet_when_voices_cannot_be_listed(monkeypatch, tmp_path):
    # Don't cry wolf on a probe failure — let speak() try. The binary exists but
    # cannot be executed, so listing the voices raises OSError.
    say = _fake_say(tmp_path, "exit 0\n", executable=False)
    monkeypatch.setenv("CHAD_VOICE", "Whoever")
    assert speech.tts_status(say_bin=say) == (True, "")


def test_stt_status_catches_a_bad_quant_width(monkeypatch):
    monkeypatch.setenv("CHAD_STT_QUANT", "2")
    ok, reason = speech.stt_status()
    assert ok is False
    assert "CHAD_STT_QUANT" in reason
    monkeypatch.setenv("CHAD_STT_QUANT", "8")
    assert speech.stt_status() == (True, "")


def test_speaker_preempts_the_previous_utterance(tmp_path):
    # A reply about turn N read aloud while you are reading turn N+1 is worse
    # than silence, so a new speak() kills the old process.
    sp = speech.Speaker(say_bin=_fake_say(tmp_path, "exec sleep 30\n"))
    sp.speak("first")
    first = sp._proc
    sp.speak("second")
    second = sp._proc
    assert first.wait(timeout=_JOIN) == -signal.SIGKILL   # the first one was killed
    assert second.poll() is None                           # the second still speaks
    sp.stop()
    assert second.wait(timeout=_JOIN) == -signal.SIGKILL
    assert sp._proc is None


def test_speaker_survives_a_missing_say_binary(tmp_path):
    sp = speech.Speaker(say_bin=str(tmp_path / "say"))
    sp.speak("hello")              # best-effort: must not raise into the turn
    assert sp._proc is None


# -- the take cap, at the Recorder level -------------------------------------

def test_recorder_stops_growing_the_take_at_the_cap():
    import numpy as np
    rec = speech.Recorder()
    rec.start()
    sr = speech.SAMPLE_RATE
    cap = int(speech.MAX_TAKE_S * sr)
    second = np.ones(sr, dtype=np.float32)     # one buffer, appended by reference
    for _ in range(int(speech.MAX_TAKE_S) + 5):  # five seconds past the cap
        rec._on_audio(second)
    assert rec.take_full is True
    audio = rec.stop()
    assert cap <= len(audio) < cap + sr        # capped, chunk-granular
    assert rec.take_full is False              # reset for the next take


def test_transcribe_truncates_an_over_long_array(monkeypatch, tmp_path):
    # The ceiling guard pairs with the recorder cap: transcribe() is also
    # reachable with an array nothing bounded.
    import numpy as np
    monkeypatch.setenv("CHAD_SPEECH_WORDS", str(tmp_path / "none.json"))
    sr = speech.SAMPLE_RATE
    cap = int(speech.MAX_TAKE_S * sr)
    audio = np.zeros(cap + 10 * sr, dtype=np.float32)
    audio[0] = audio[-1] = 0.5                 # sound at both ends: trimming keeps it all
    handed_over = []

    def decode(take):
        handed_over.append(len(take))
        return "heard"

    assert speech.transcribe(audio, decode=decode) == "heard"
    assert handed_over == [cap]                # 10s past the cap handed in, the cap handed over
