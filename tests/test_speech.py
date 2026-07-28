"""Tier-1 (no audio deps) tests for voice mode's pure text layer.

speech.py's audio halves (sounddevice mic capture, Parakeet-on-MLX decode, `say`
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


def test_stt_quant_bits_parsing(monkeypatch):
    import pytest
    monkeypatch.delenv("CHAD_STT_QUANT", raising=False)
    assert speech.stt_quant_bits() == 8          # quantized by default (A/B'd)
    for off in ("none", "0", "16", "off", "BF16"):
        monkeypatch.setenv("CHAD_STT_QUANT", off)
        assert speech.stt_quant_bits() is None
    monkeypatch.setenv("CHAD_STT_QUANT", "4")
    assert speech.stt_quant_bits() == 4
    monkeypatch.setenv("CHAD_STT_QUANT", "2")    # unsupported width: loud, not silent
    with pytest.raises(ValueError):
        speech.stt_quant_bits()


def test_stt_model_env_override(monkeypatch):
    monkeypatch.delenv("CHAD_STT_MODEL", raising=False)
    assert speech.stt_model() == speech.DEFAULT_STT_MODEL
    assert "parakeet" in speech.DEFAULT_STT_MODEL
    monkeypatch.setenv("CHAD_STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
    assert speech.stt_model() == "mlx-community/parakeet-tdt-0.6b-v2"


def test_available_reports_missing_deps_as_hint():
    # Whatever this machine has installed, the contract holds: either speech is
    # available, or the reason is a human install hint — never an exception.
    ok, reason = speech.available()
    assert ok in (True, False)
    if not ok:
        assert "uv sync --extra speech" in reason


def test_collapse_repeats_kills_hallucination_loop():
    # The real failure this guards: a decoder looping on a silent tail.
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


def test_vendored_parakeet_stays_heavy_dep_free():
    # The whole point of vendoring (see chad/parakeet/__init__.py): a future
    # re-sync from upstream must not smuggle back torch, librosa's numba/scipy
    # tree, or the dacite/typer conveniences we replaced.
    import re
    pdir = os.path.join(os.path.dirname(os.path.abspath(speech.__file__)), "parakeet")
    banned = re.compile(
        r"^\s*(import|from)\s+(torch|numba|scipy|librosa|dacite|typer|tiktoken)\b",
        re.MULTILINE)
    for fn in os.listdir(pdir):
        if fn.endswith(".py"):
            with open(os.path.join(pdir, fn)) as f:
                assert not banned.search(f.read()), f"{fn} imports a banned heavy dep"


def test_vendored_parakeet_imports_clean_and_carries_license():
    import pytest
    pytest.importorskip("mlx.core")
    import importlib
    parakeet = importlib.import_module("chad.parakeet")
    assert callable(parakeet.from_pretrained) and callable(parakeet.get_logmel)
    for mod in ("torch", "librosa", "dacite", "numba", "scipy"):
        assert mod not in sys.modules, f"importing chad.parakeet loaded {mod}"
    # Apache-2.0 vendored code: the license text must travel with it (wheel too).
    lic = os.path.join(os.path.dirname(os.path.abspath(parakeet.__file__)), "LICENSE")
    assert os.path.exists(lic)


def test_mel_filters_match_librosa_goldens():
    # audio.py's librosa-free filterbank, pinned to values computed with real
    # librosa (slaney scale + norm) at the parakeet v3 preprocessor params.
    # If a re-sync or "simplification" changes the math, this catches it.
    import pytest
    pytest.importorskip("mlx.core")
    from chad.parakeet.audio import mel_filters
    fb = mel_filters(16000, 512, 128)
    assert fb.shape == (128, 257)
    for i, j, want in [(0, 1, 0.02837754), (0, 2, 0.0), (64, 100, 0.0),
                       (127, 255, 0.00087053), (127, 256, 0.0)]:
        assert abs(float(fb[i, j]) - want) < 1e-6, (i, j)
    for row, want in enumerate([0.02837754, 0.02837754, 0.02877802]):
        assert abs(float(fb.sum(axis=1)[row]) - want) < 1e-6


def test_vendored_parakeet_config_builder():
    # utils.from_dict replaced dacite: nested dataclasses built recursively,
    # extra keys ignored, defaults respected, `X | None` unwrapped.
    import pytest
    pytest.importorskip("mlx.core")
    from chad.parakeet.audio import PreprocessArgs
    from chad.parakeet.utils import from_dict
    cfg = from_dict(PreprocessArgs, {
        "sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
        "window_stride": 0.01, "window": "hann", "features": 128, "n_fft": 512,
        "dither": 1e-5, "_target_": "ignored.extra.Key", "log": True,
    })
    assert cfg.sample_rate == 16000 and cfg.features == 128
    assert cfg.pad_to == 0 and cfg.preemph == 0.97  # defaults filled in
    assert cfg.win_length == 400 and cfg.hop_length == 160


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
