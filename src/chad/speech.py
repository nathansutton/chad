"""All-local speech I/O for the TUI: push-to-talk dictation + spoken replies.

Nothing here leaves the machine. STT is Parakeet TDT on MLX — chad's own
vendored copy (chad.parakeet), same stack as the coding model, weights in the
shared Hugging Face cache; TTS is macOS `say`, which synthesizes on-device.
The one audio dep (sounddevice) is optional: the module imports clean without
it, and `available()` reports what's missing so the TUI can print an install
hint instead of crashing.

The heavy imports (sounddevice pulls PortAudio, chad.parakeet loads MLX) are
deferred to first use — `/speech` off must cost nothing, and touching MLX at
import time would slow the banner for every non-speech session.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections import deque

# Parakeet TDT v3 (0.6B, multilingual): the engine Hex ships as its default,
# and for the same reasons — a class faster than Whisper per clip, better
# dictation accuracy, and a TDT decoder without Whisper's repetition-loop
# hallucination mode. One-time ~2.5 GB download (the repo stores fp32;
# from_pretrained casts to bf16, ~1.2 GB resident); the tokenizer vocabulary
# rides inside the repo's config.json, so no vocab files ship with chad.
# Env-overridable, not a flag: a per-machine fit, not a per-session choice.
DEFAULT_STT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16_000  # what the ASR models are trained on; record at it

# The encoder ships as `rel_pos` with att_context_size [-1,-1] — full global
# attention, so its score matrix is T×T and a take costs QUADRATICALLY in its
# own length, on the same GPU the coding model is resident on. Switching to the
# banded local kernel makes that matrix (T, 2w+1) — linear — and costs nothing
# in accuracy here: w=256 encoder frames is ±20.5s of audio at 12.5 frames/s,
# so a dictation take sees its whole self anyway. Measured on a 9.7s clip, the
# transcript is byte-identical to full attention (and 8-bit local is
# byte-identical to bf16 local); at 291s the encoder is still flat where full
# attention is still climbing.
LOCAL_ATTN_CONTEXT = (256, 256)

# A take is bounded so a mic left open (a call, a meeting, a walk away) can't
# turn into a multi-minute decode nobody asked for. With local attention this
# is a sanity backstop, not the thing standing between you and an OOM — hence
# generous. Recording does not stop; the take simply stops growing, so
# everything said up to the cap is still transcribed rather than lost.
MAX_TAKE_S = 600


def stt_model() -> str:
    return os.environ.get("CHAD_STT_MODEL", DEFAULT_STT_MODEL)


def stt_quant_bits():
    """Load-time quantization width for the STT weights; None = keep bf16.

    Default 8: A/B'd against bf16 on a 24-clip multi-voice battery — 21/24
    transcripts byte-identical, the rest case/spacing-level (twice the 8-bit
    output was the *more* correct one), 786 MB resident vs 1296, and ~40%
    faster decode (the decoder is memory-bound). CHAD_STT_QUANT=4 halves
    memory again (514 MB, equally clean on the battery) but synthetic clean
    audio can't rule out degradation on noisy mics — opt-in, not default."""
    raw = os.environ.get("CHAD_STT_QUANT", "8").strip().lower()
    if raw in ("", "0", "16", "none", "off", "bf16"):
        return None
    if raw in ("4", "8"):
        return int(raw)
    raise ValueError(f"CHAD_STT_QUANT must be 8 (default), 4, or none — got {raw!r}")


def model_cached() -> bool:
    """Whether the STT weights are already in the HF cache (or a local path) —
    so the TUI can warn that the first ctrl-t triggers a one-time download
    instead of letting it look like a hang."""
    if os.path.isdir(stt_model()):
        return True
    try:
        from huggingface_hub import hf_hub_download
        for f in ("config.json", "model.safetensors"):  # what from_pretrained loads
            hf_hub_download(stt_model(), f, local_files_only=True)
        return True
    except Exception:
        return False


def available():
    """(ok, reason). ok=False carries the one-line reason speech can't start —
    surfaced verbatim by /speech, so it names the fix, not just the failure."""
    from importlib.util import find_spec
    # find_spec, not import: importing sounddevice INITIALIZES PortAudio (a
    # CoreAudio client thread that can crash interpreter teardown when it was
    # started for no reason, e.g. by a mere availability probe in tests). A
    # present-but-broken install still fails loudly later at open_stream(),
    # which /speech catches and reports.
    missing = []
    try:
        if find_spec("sounddevice") is None:
            missing.append("sounddevice")
    except (ImportError, ValueError):
        missing.append("sounddevice")
    if missing:
        return False, (f"speech needs {' + '.join(missing)} — install with "
                       f"`uv sync --extra speech` (all local: Parakeet-on-MLX STT, "
                       f"macOS `say` TTS)")
    return True, ""


SAY_BIN = "/usr/bin/say"


def tts_status():
    """(ok, reason) for the SPOKEN-REPLY half, checked at /speech enable.

    Deliberately not part of `available()`: a machine without `say` can still
    dictate, so this warns instead of blocking. It exists because every TTS
    failure is otherwise invisible — Speaker.speak swallows OSError, and a
    misspelled CHAD_VOICE doesn't even raise that (Popen succeeds, `say` exits
    nonzero), so voice mode would promise spoken replies and stay silent
    forever with nothing to pull on.
    """
    if not os.path.exists(SAY_BIN):
        return False, f"{SAY_BIN} not found — dictation works, replies won't be spoken"
    voice = os.environ.get("CHAD_VOICE")
    if not voice:
        return True, ""
    try:
        out = subprocess.run([SAY_BIN, "-v", "?"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return True, ""  # can't enumerate: don't cry wolf, let speak() try
    # Each line is "Name   lang  # sample sentence"; the name may contain spaces
    # ("Eddy (English (UK))"), so match the whole pre-language prefix.
    names = [ln.split("#")[0].rsplit(None, 1)[0].strip()
             for ln in out.splitlines() if ln.strip() and "#" in ln]
    if not any(n.lower() == voice.lower() for n in names):
        # difflib, not a substring test: the typos people actually make are
        # deletions and transpositions ("Samanta"), which no substring catches.
        import difflib
        near = difflib.get_close_matches(voice, names, n=3, cutoff=0.7)
        hint = f" (did you mean {', '.join(near)}?)" if near else ""
        return False, (f"CHAD_VOICE={voice!r} is not an installed voice{hint} — "
                       f"replies won't be spoken; `say -v '?'` lists them")
    return True, ""


def stt_status():
    """(ok, reason) for the env knobs transcribe() reads. stt_quant_bits raises
    on an unsupported width, but it is only ever called mid-take — so without
    this a typo surfaces as a failed transcription AFTER you have already
    spoken. Checked at /speech enable, where the fix is still cheap."""
    try:
        stt_quant_bits()
    except ValueError as e:
        return False, str(e)
    return True, ""


# ---------------------------------------------------------------------------
# Text -> speakable text. Pure functions, unit-tested without audio deps.
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```.*?(```|\Z)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")
_MD_NOISE_RE = re.compile(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s?)", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")


def spoken_text(text: str, max_chars: int = 600) -> str:
    """Reduce assistant prose to something worth hearing.

    Code blocks are dropped, not read: a diff spoken aloud is noise, and the
    real content is already on screen. Markdown markup is stripped because
    `say` reads it literally ("asterisk asterisk"). Long answers are cut at a
    sentence boundary — TTS at ~15 chars/sec turns a 3k-char answer into over
    three minutes of monologue that outlives the user's attention.
    """
    t = _FENCE_RE.sub(" ", text)
    t = _INLINE_CODE_RE.sub(r"\1", t)
    t = _EMPHASIS_RE.sub(r"\2", t)
    t = _MD_NOISE_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        cut = t[:max_chars]
        # Prefer ending on a sentence; a mid-word cut sounds like a dropped call.
        dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        t = cut[:dot + 1] if dot > max_chars // 2 else cut.rsplit(" ", 1)[0] + "…"
    return t


def collapse_repeats(text: str, max_run: int = 3) -> str:
    """Collapse a phrase stuck on repeat ("and hour and hour and hour…" ×40)
    down to at most `max_run` occurrences. A within-window repetition loop is
    the classic decoder failure on imperfect audio — Whisper's signature one,
    and the reason this guard was written; Parakeet's TDT decoder doesn't have
    that mode, but a flooded input box is bad enough to keep the insurance."""
    words = text.split()
    for size in range(1, 9):  # phrase lengths 1..8 words
        out, i, n = [], 0, len(words)
        while i < n:
            phrase = words[i:i + size]
            run = 1
            while words[i + run * size:i + (run + 1) * size] == phrase:
                run += 1
            out.extend(phrase * min(run, max_run))
            i += run * size
        words = out
    return " ".join(words)


def remap_path() -> str:
    return os.environ.get("CHAD_SPEECH_WORDS",
                          os.path.expanduser("~/.chad/speech_words.json"))


def load_remaps() -> dict:
    """The user's personal word table: a JSON object of {heard: meant}, e.g.
    {"pie test": "pytest", "you vee": "uv"}. No ASR model will ever spell a
    codebase's identifiers unaided — this fixes the misheard-term problem at
    the text layer (another idea taken from Hex's word remappings). Missing
    file => {}. A malformed file raises ValueError so /speech can say so at
    enable time; transcribe() itself never lets it break a dictation."""
    path = remap_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("speech words file must be a JSON object of "
                         '{"heard phrase": "replacement"}')
    return {str(k): str(v) for k, v in data.items()}


def apply_remaps(text: str, remaps: dict) -> str:
    """Case-insensitive whole-word replacement, longest match first so
    "pie test dot py" wins over a bare "pie test" entry. Replacements are
    literal (no regex/backreference expansion from user data)."""
    for match, repl in sorted(remaps.items(), key=lambda kv: -len(kv[0])):
        m = match.strip()
        if not m:
            continue

        def _literal(_mo: re.Match, _r: str = repl) -> str:
            return _r
        text = re.sub(rf"(?<!\w){re.escape(m)}(?!\w)",
                      _literal, text, flags=re.IGNORECASE)
    return text


def trim_silence(audio, threshold: float = 0.01, pad_s: float = 0.25):
    """Cut leading/trailing silence from a push-to-talk capture. The tail — the
    seconds between finishing a sentence and finding ctrl-t again — is dead
    weight the encoder would otherwise pay for at 12.5 frames per second, and
    the kind of near-silence a decoder is most likely to emit spurious tokens
    on; it never sees it. All-quiet audio (peak below `threshold`: an accidental
    tap, a muted mic) trims to empty, which the caller reports as "heard
    nothing" without invoking the model."""
    import numpy as np
    if len(audio) == 0:
        return audio
    amp = np.abs(audio)
    peak = float(amp.max())
    if peak < threshold:
        return audio[:0]
    idx = np.flatnonzero(amp >= max(threshold, 0.05 * peak))
    pad = int(pad_s * SAMPLE_RATE)
    return audio[max(0, int(idx[0]) - pad):int(idx[-1]) + pad]


def say_argv(text: str) -> list:
    """argv for macOS `say`. Text passed as a single argument (no shell), voice
    and rate env-tunable. `--` guards a transcript that starts with a dash."""
    argv = ["/usr/bin/say"]
    voice = os.environ.get("CHAD_VOICE")
    if voice:
        argv += ["-v", voice]
    rate = os.environ.get("CHAD_SPEECH_RATE")
    if rate:
        argv += ["-r", rate]
    return argv + ["--", text]


# ---------------------------------------------------------------------------
# TTS: macOS `say` in a subprocess. On-device synthesis, zero Python deps.
# ---------------------------------------------------------------------------

class Speaker:
    """One utterance at a time; a new speak() preempts the old one. Replies can
    arrive faster than they can be read aloud, and stale speech about turn N
    while reading turn N+1 is worse than silence."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def speak(self, text: str):
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._kill_locked()
            try:
                self._proc = subprocess.Popen(
                    say_argv(text),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:  # not macOS / say missing — speech stays best-effort
                self._proc = None

    def stop(self):
        with self._lock:
            self._kill_locked()

    def _kill_locked(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        self._proc = None


# ---------------------------------------------------------------------------
# STT: microphone capture (sounddevice) -> Parakeet-on-MLX transcription.
# ---------------------------------------------------------------------------

class Recorder:
    """Warm-capture push-to-talk (an idea taken from Hex, kitlangton/Hex, MIT):
    while speech mode is on the input stream stays OPEN, feeding a ~1s ring
    buffer that is continuously discarded; start() seeds the take with the last
    0.45s from that ring. Opening a CoreAudio device takes 100–300ms — opened
    on the keypress, that clips the first syllable ("ix the failing test"); the
    pre-roll means audio from *before* the keypress is already in hand. The
    trade is an honestly-open mic, which the caller must surface in the UI.

    Frames stay in memory — minutes of speech are a few MB, nothing on disk.
    `_on_audio` is the whole state machine and takes plain numpy chunks, so the
    ring/pre-roll logic is unit-tested without any audio hardware.

    Two buffers, one of which is always being thrown away:

        open_stream()                                       close()
             │                                                 ▲
             ▼                                                 │
        ┌─────────── IDLE (mic open, not recording) ───────────┴──┐
        │  _on_audio ──► _ring.append(chunk)                      │
        │                while _ring_samples > RING_S: popleft    │  ← bounded
        └───────────────────────────┬─────────────────────────────┘
                       start()      │      ▲  stop() / cancel()
          pop PRE_ROLL_S off the    │      │  _frames handed off (or dropped),
          RIGHT of _ring, reverse,  │      │  _ring cleared, back to IDLE
          seed _frames with it      ▼      │
        ┌──────────── RECORDING ───────────┴──────────────────────┐
        │  _on_audio ──► _frames.append(chunk)                    │
        │                _take_samples += len(chunk)              │
        │                     │                                   │
        │                     ├── under MAX_TAKE_S ──► keep going │
        │                     └── at MAX_TAKE_S ────► FULL        │  ← bounded
        └───────────────────────────┬─────────────────────────────┘
                                    ▼
        ┌────────────── RECORDING, TAKE FULL ─────────────────────┐
        │  _on_audio ──► dropped. `take_full` is True so the UI    │
        │  can say so. stop() still returns everything up to the   │
        │  cap — a long take is truncated, never lost.             │
        └──────────────────────────────────────────────────────────┘

    The take is capped because nothing else caps it: a mic left open feeds
    _frames forever, and the whole thing is handed to the encoder in one shot.
    """

    PRE_ROLL_S = 0.45   # Hex's default; enough for a clipped first word, cheap
    RING_S = 1.0        # ring capacity; only PRE_ROLL_S of it is ever kept

    def __init__(self):
        self._stream = None
        self._lock = threading.Lock()
        self._frames = []       # the active take (list of 1-D float32 chunks)
        self._take_samples = 0  # length of _frames, so the cap is O(1) to check
        self._take_full = False  # hit MAX_TAKE_S: still recording, no longer growing
        self._ring = deque()    # trailing idle audio, pre-roll source
        self._ring_samples = 0
        self._recording = False

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def stream_open(self) -> bool:
        return self._stream is not None

    @property
    def take_full(self) -> bool:
        """The take hit MAX_TAKE_S and stopped growing — the UI says so."""
        return self._take_full

    def _on_audio(self, chunk):
        """One mono chunk from the capture callback (audio thread)."""
        with self._lock:
            if self._recording:
                if self._take_samples >= int(MAX_TAKE_S * SAMPLE_RATE):
                    # Drop, don't stop: the user keeps whatever they already
                    # said, and the mic stays honest about being open.
                    self._take_full = True
                    return
                self._frames.append(chunk)
                self._take_samples += len(chunk)
                return
            self._ring.append(chunk)
            self._ring_samples += len(chunk)
            cap = int(self.RING_S * SAMPLE_RATE)
            while self._ring and self._ring_samples - len(self._ring[0]) >= cap:
                self._ring_samples -= len(self._ring.popleft())

    def open_stream(self):
        """Open the mic and start the warm ring. Raises (PortAudioError) if the
        mic is unavailable — e.g. TCC permission denied — so the caller can
        surface it at /speech time instead of recording silence later."""
        if self._stream is not None:
            return
        import sounddevice as sd

        def _cb(indata, frames, time_info, status):  # audio thread — copy only
            self._on_audio(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_cb)
        self._stream.start()

    def close(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            self._frames = []
            self._take_samples = 0
            self._take_full = False
            self._ring.clear()
            self._ring_samples = 0
            self._recording = False

    def start(self):
        """Begin a take. Pure state transition — does NOT open hardware: the
        stream is opened once at /speech enable (open_stream), which is also
        what makes the unit tests hardware-free. With no stream open the take
        simply captures nothing and stop() returns empty ("heard nothing")."""
        with self._lock:
            # Seed the take with the ring's tail: whole chunks until we have
            # PRE_ROLL_S worth (slightly over at chunk granularity is fine —
            # trim_silence cuts any leading quiet).
            want = int(self.PRE_ROLL_S * SAMPLE_RATE)
            seed, got = [], 0
            while self._ring and got < want:
                c = self._ring.pop()
                seed.append(c)
                got += len(c)
            seed.reverse()
            self._frames = seed
            self._take_samples = got   # pre-roll counts toward the cap
            self._take_full = False
            self._ring.clear()
            self._ring_samples = 0
            self._recording = True

    def stop(self):
        """End the take and return it (mono float32 @ 16 kHz). The stream stays
        open — the warm ring resumes for the next take."""
        import numpy as np
        with self._lock:
            self._recording = False
            frames, self._frames = self._frames, []
            self._take_samples = 0
            self._take_full = False
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames)

    def cancel(self):
        """Discard the take without transcribing (esc while recording)."""
        with self._lock:
            self._recording = False
            self._frames = []
            self._take_samples = 0
            self._take_full = False


_stt = {}  # {(model_id, bits): loaded chad.parakeet model} — load once per process


def release_model() -> bool:
    """Drop the cached STT weights and hand the GPU memory back. Returns whether
    anything was actually released.

    Called when voice mode is turned off, so a session that dictated once
    doesn't carry ~790 MB for the rest of its life on a machine whose coding
    model already owns most of RAM. The caller MUST NOT call this while a
    transcription is in flight — the worker thread holds a live reference to
    the model inside generate()."""
    if not _stt:
        return False
    _stt.clear()
    import mlx.core as mx
    mx.clear_cache()
    return True


def transcribe(audio) -> str:
    """Parakeet over a raw 16 kHz array. First call downloads + loads the model
    (cached in-process afterwards); runs on the same GPU as the coding model,
    which is idle between turns — exactly when dictation happens."""
    audio = trim_silence(audio)
    if len(audio) < SAMPLE_RATE // 4:  # <0.25s of actual sound: a tap, not speech
        return ""
    # Ceiling guard, paired with Recorder's cap: transcribe() is also reachable
    # with an array the recorder never bounded (tests, a future caller), and the
    # encoder should never be handed an unbounded one.
    audio = audio[:int(MAX_TAKE_S * SAMPLE_RATE)]
    import mlx.core as mx
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    from . import parakeet
    # The first-use hf download draws tqdm bars straight to the tty, over the
    # prompt_toolkit region — silence them for the duration (the TUI shows its
    # own download notice); restore so chad's own model download keeps its bar.
    bars_off = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        key = (stt_model(), stt_quant_bits())
        model = _stt.get(key)
        if model is None:
            model = parakeet.from_pretrained(stt_model())
            # Before quantizing, not after: set_attention_model rebuilds the
            # attention modules and copies weights in with load_weights(), which
            # rejects a quantized layer's scales/biases as parameters the fresh
            # nn.Linear doesn't have.
            model.encoder.set_attention_model("rel_pos_local_attn",
                                              LOCAL_ATTN_CONTEXT)
            bits = stt_quant_bits()
            if bits:
                # Quantize at load from the fp32 download — no separate quant
                # repo to host, and the group-64 divisibility predicate leaves
                # the few odd-shaped layers (and all convs) in bf16.
                import mlx.nn as mlx_nn
                mlx_nn.quantize(
                    model, group_size=64, bits=bits,
                    class_predicate=lambda p, m: isinstance(
                        m, (mlx_nn.Linear, mlx_nn.Embedding))
                    and m.weight.shape[-1] % 64 == 0)
            _stt[key] = model
        mel = parakeet.get_logmel(mx.array(audio), model.preprocessor_config)
        result = model.generate(mel)[0]
    finally:
        if not bars_off:
            enable_progress_bars()
    # collapse_repeats stays as cheap insurance: Parakeet's TDT decoder lacks
    # Whisper's repetition-loop failure mode, but a flooded input box is bad
    # enough to keep the guard anyway.
    text = collapse_repeats(result.text.strip())
    try:
        text = apply_remaps(text, load_remaps())
    except (ValueError, OSError):
        pass  # malformed word table — reported by /speech at enable, not mid-take
    return text
