"""Tier-1 (no audio deps) tests for voice mode's pure text layer.

speech.py's audio halves (sounddevice mic capture, mlx-whisper decode, `say`
playback) need hardware and the optional extra, so they're exercised manually.
What IS testable everywhere is the text contract: `spoken_text` (what a reply
sounds like), `say_argv` (how the utterance reaches `say`), and `available()`
degrading to an install hint instead of an ImportError.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad import speech  # noqa: E402
from chad.speech import (  # noqa: E402
    Recorder,
    apply_remaps,
    collapse_repeats,
    load_remaps,
    say_argv,
    spoken_text,
    trim_silence,
)


def test_spoken_text_drops_code_blocks():
    # A spoken diff is noise — the code is already on screen.
    t = spoken_text("Fixed the bug.\n```python\ndef f():\n    return 1\n```\nTests pass.")
    assert "def f" not in t
    assert "```" not in t
    assert "Fixed the bug." in t and "Tests pass." in t


def test_spoken_text_drops_unclosed_fence():
    # A truncated turn can end mid-fence; everything after ``` is still code.
    t = spoken_text("Done.\n```\nraw dump that never closes")
    assert t == "Done."


def test_spoken_text_strips_markdown_noise():
    t = spoken_text("## Result\n- **bold** item\n1. `inline_code` here\n> quoted")
    assert "#" not in t and "*" not in t and "`" not in t and ">" not in t
    assert "bold item" in t
    assert "inline_code here" in t  # inline code keeps its text, loses the ticks


def test_spoken_text_collapses_whitespace():
    assert spoken_text("a\n\n\nb   c") == "a b c"


def test_spoken_text_truncates_at_sentence():
    long = "This is a sentence. " * 100
    t = spoken_text(long, max_chars=100)
    assert len(t) <= 100
    assert t.endswith(".")  # cut on a sentence boundary, not mid-word


def test_spoken_text_truncates_unpunctuated_ramble():
    t = spoken_text("word " * 200, max_chars=100)
    assert len(t) <= 101  # +1 for the ellipsis
    assert t.endswith("…")
    assert not t[:-1].endswith(" ")  # no dangling half-word before the ellipsis


def test_spoken_text_empty_and_code_only():
    assert spoken_text("") == ""
    assert spoken_text("```\nonly code\n```") == ""


def test_say_argv_shape(monkeypatch):
    monkeypatch.delenv("CHAD_VOICE", raising=False)
    monkeypatch.delenv("CHAD_SPEECH_RATE", raising=False)
    assert say_argv("hello") == ["/usr/bin/say", "--", "hello"]
    # `--` so a transcript starting with a dash can't be parsed as a flag.
    assert say_argv("-v evil")[-2:] == ["--", "-v evil"]


def test_say_argv_env_tuning(monkeypatch):
    monkeypatch.setenv("CHAD_VOICE", "Samantha")
    monkeypatch.setenv("CHAD_SPEECH_RATE", "220")
    argv = say_argv("hi")
    assert argv[:5] == ["/usr/bin/say", "-v", "Samantha", "-r", "220"]


def test_stt_model_env_override(monkeypatch):
    monkeypatch.delenv("CHAD_WHISPER_MODEL", raising=False)
    assert speech.stt_model() == speech.DEFAULT_STT_MODEL
    monkeypatch.setenv("CHAD_WHISPER_MODEL", "mlx-community/whisper-small-mlx")
    assert speech.stt_model() == "mlx-community/whisper-small-mlx"


def test_available_reports_missing_deps_as_hint():
    # Whatever this machine has installed, the contract holds: either speech is
    # available, or the reason is a human install hint — never an exception.
    ok, reason = speech.available()
    assert ok in (True, False)
    if not ok:
        assert "uv sync --extra speech" in reason


def test_collapse_repeats_kills_hallucination_loop():
    # The real failure this guards: Whisper looping on a silent tail.
    loop = "look at all the failing test cases and the " + "hour and " * 60
    t = collapse_repeats(loop.strip())
    assert t.count("hour") <= 4  # a couple survive, the flood doesn't
    assert t.startswith("look at all the failing test cases")


def test_collapse_repeats_leaves_normal_speech_alone():
    for s in ("does this transcription really work",
              "run it again and again please",       # ×2 repeat is legit speech
              "no no no that is wrong"):              # ×3 stays under the cap
        assert collapse_repeats(s) == s


def test_collapse_repeats_multiword_phrase():
    t = collapse_repeats("fix it " * 20 + "now")
    assert t == "fix it fix it fix it now"


def test_trim_silence_drops_quiet_tail_and_lead():
    import numpy as np
    sr = speech.SAMPLE_RATE
    audio = np.zeros(6 * sr, dtype=np.float32)
    audio[2 * sr:3 * sr] = 0.3  # one second of "speech" amid five of silence
    out = trim_silence(audio)
    # keeps the loud second plus the 0.25s pad each side, sheds the rest
    assert sr <= len(out) <= int(1.6 * sr)


def test_trim_silence_all_quiet_is_empty():
    import numpy as np
    hiss = (np.ones(3 * speech.SAMPLE_RATE) * 0.003).astype(np.float32)
    assert len(trim_silence(hiss)) == 0    # never reaches the model
    assert len(trim_silence(hiss[:0])) == 0


def test_vendored_whisper_stays_torch_free():
    # The whole point of vendoring (see chad/whisper/__init__.py): a future
    # re-sync from upstream must not smuggle the torch/numba/scipy imports back.
    import re
    wdir = os.path.join(os.path.dirname(os.path.abspath(speech.__file__)), "whisper")
    banned = re.compile(r"^\s*(import|from)\s+(torch|numba|scipy)\b", re.MULTILINE)
    for fn in os.listdir(wdir):
        if fn.endswith(".py"):
            with open(os.path.join(wdir, fn)) as f:
                assert not banned.search(f.read()), f"{fn} imports a banned heavy dep"


def test_vendored_whisper_imports_and_ships_assets():
    import pytest
    pytest.importorskip("mlx.core")
    pytest.importorskip("tiktoken")
    import importlib
    whisper = importlib.import_module("chad.whisper")
    assert callable(whisper.transcribe)
    assert "torch" not in sys.modules  # importing it must not drag torch in
    # The wheel-shipped data files the tokenizer/mel pipeline resolve by __file__.
    assets = os.path.join(os.path.dirname(os.path.abspath(whisper.__file__)), "assets")
    for fn in ("mel_filters.npz", "multilingual.tiktoken", "gpt2.tiktoken"):
        assert os.path.exists(os.path.join(assets, fn)), fn


def _chunk(value, n):
    import numpy as np
    return (np.ones(n) * value).astype(np.float32)


def test_recorder_preroll_seeds_the_take():
    # Warm ring: idle audio arrives, then start() — the take must begin with
    # ~PRE_ROLL_S of audio from BEFORE the keypress (the anti-clipping fix).
    rec = Recorder()
    sr = speech.SAMPLE_RATE
    for i in range(10):  # 1s of idle audio in 0.1s chunks, values 0..9
        rec._on_audio(_chunk(i, sr // 10))
    rec.start()
    rec._on_audio(_chunk(99, sr // 10))  # spoken after the keypress
    audio = rec.stop()
    want_pre = int(Recorder.PRE_ROLL_S * sr)
    # pre-roll present (chunk-granular: at least the target, under the ring cap)
    assert want_pre <= len(audio) - sr // 10 <= int(Recorder.RING_S * sr)
    assert audio[0] >= 5           # earliest idle chunks (0..4) fell outside the pre-roll
    assert audio[-1] == 99         # live speech lands after the pre-roll
    assert not rec.recording       # stop() ends the take, ring resumes


def test_recorder_ring_is_bounded_and_cancel_discards():
    rec = Recorder()
    sr = speech.SAMPLE_RATE
    for i in range(50):  # 5s of idle audio must not accumulate past RING_S
        rec._on_audio(_chunk(i, sr // 10))
    assert rec._ring_samples <= int(Recorder.RING_S * sr) + sr // 10
    rec.start()
    rec._on_audio(_chunk(1.0, sr))
    rec.cancel()                   # esc: throw the take away
    assert not rec.recording
    assert len(rec.stop()) == 0    # nothing survives a cancel


def test_apply_remaps_word_boundary_case_insensitive():
    remaps = {"pie test": "pytest", "you vee": "uv", "chad": "chad"}
    assert apply_remaps("run Pie Test now", remaps) == "run pytest now"
    assert apply_remaps("occupie test", remaps) == "occupie test"  # no mid-word hit
    assert apply_remaps("YOU VEE sync", remaps) == "uv sync"


def test_apply_remaps_longest_match_first_and_literal_replacement():
    remaps = {"pie test": "pytest", "pie test dot py": "pytest.py"}
    assert apply_remaps("run pie test dot py", remaps) == "run pytest.py"
    # user data is literal on both sides: regex metachars and backslashes inert
    assert apply_remaps("c plus plus", {"c plus plus": r"C++ \1"}) == r"C++ \1"
    assert apply_remaps("a.b", {"a.b": "x"}) == "x"
    assert apply_remaps("acb", {"a.b": "x"}) == "acb"  # '.' not a wildcard


def test_load_remaps_missing_env_and_malformed(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_SPEECH_WORDS", str(tmp_path / "none.json"))
    assert load_remaps() == {}
    p = tmp_path / "words.json"
    p.write_text('{"pie test": "pytest"}')
    monkeypatch.setenv("CHAD_SPEECH_WORDS", str(p))
    assert load_remaps() == {"pie test": "pytest"}
    p.write_text('["not", "a", "dict"]')
    import pytest
    with pytest.raises(ValueError):
        load_remaps()  # /speech surfaces this at enable time


def test_speaker_noop_on_empty_text():
    # Empty utterance never spawns a process (and never raises without `say`).
    sp = speech.Speaker()
    sp.speak("   ")
    assert sp._proc is None
    sp.stop()
