"""Vendored Parakeet-on-MLX inference — chad's STT engine, MLX-only by design.

Copied from parakeet-mlx 0.5.2 (Apache-2.0, Copyright Senstella,
https://github.com/senstella/parakeet-mlx — license text in this directory's
LICENSE file). Parakeet TDT v3 (0.6B) replaces the earlier vendored Whisper:
same speed class as the coding model's own stack, better dictation accuracy,
and its TDT decoder doesn't have Whisper's repetition-loop hallucination mode.
Crucially, its tokenizer vocabulary ships inside the model repo's config.json —
no vocab data files in this repo or the wheel.

Vendored rather than depended on because the PyPI package drags librosa
(→ numba/scipy/scikit-learn) for a single mel-filterbank call, plus typer for
a CLI chad doesn't use. What changed, relative to upstream:
- audio.py: librosa.filters.mel replaced by a local numpy implementation,
  validated against librosa to <1e-8 (goldens in tests/test_speech.py);
  load_audio (ffmpeg file decode) removed — chad feeds mic arrays.
- utils.py: dacite.from_dict replaced by a local recursive dataclass builder.
- parakeet.py: transcribe(path) raises (it needed the ffmpeg loader); the
  array-level pipeline chad uses — get_logmel() + generate() — is untouched.
- cli.py dropped.
Every change is marked "VENDORED CHANGE"; the rest is byte-identical to
upstream, so a version bump is a re-copy plus re-applying those edits.

Runtime deps beyond chad's core tree (mlx, numpy, huggingface-hub): none.
"""
from .audio import get_logmel, mel_filters  # noqa: F401
from .utils import from_config, from_pretrained  # noqa: F401
