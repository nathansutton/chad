"""Vendored MLX Whisper inference — STT for voice mode with zero torch in scope.

Copied from mlx-whisper 0.4.3 (MIT, Copyright © 2023 Apple Inc.,
https://github.com/ml-explore/mlx-examples/tree/main/whisper). Vendored rather
than depended on because the PyPI package hard-requires torch (~2 GB) for its
checkpoint-conversion path — dead weight here: the mlx-community/whisper-* repos
chad loads are already MLX-format safetensors, so inference needs only mlx +
numpy + tiktoken + huggingface-hub, all of which chad's core dep tree carries
except tiktoken (added by the `speech` extra).

What was cut, relative to upstream:
- torch_whisper.py / cli.py / writers.py — conversion + CLI, never imported.
- timing.py and transcribe.py's word_timestamps path — word-level timestamps
  need numba + scipy; dictation only needs the text. transcribe() raises if
  word_timestamps=True is requested.
Everything kept is byte-identical to upstream apart from those cuts, so a
future version bump is a re-copy plus re-applying the transcribe.py edit
(marked with "VENDORED CHANGE").

assets/ ships in the wheel: mel_filters.npz (mel filterbank) and the two
tiktoken vocabs (multilingual + gpt2 for *.en models) — ~1.6 MB, the price of
transcription that works offline once the model weights are cached.
"""
from .transcribe import transcribe  # noqa: F401
