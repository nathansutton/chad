import functools
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


# VENDORED CHANGE: librosa's only role here was librosa.filters.mel at init.
# This is that filterbank (slaney mel scale + slaney norm, fmin=0, fmax=sr/2),
# self-contained in numpy — validated against librosa to <1e-8 max abs diff
# across parakeet/whisper-shaped configs (see tests/test_speech.py goldens).
def mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmax = sr / 2.0
    f_sp = 200.0 / 3                      # Hz per mel in the linear region
    min_log_hz = 1000.0                   # slaney: linear below 1 kHz, log above
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        mel = f / f_sp
        return np.where(f >= min_log_hz,
                        min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
                        mel)

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        return np.where(m >= min_log_mel,
                        min_log_hz * np.exp(logstep * (m - min_log_mel)),
                        f_sp * m)

    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mels + 2))
    fft_freqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    fdiff = np.diff(mel_f)
    ramps = mel_f.reshape(-1, 1) - fft_freqs.reshape(1, -1)
    lower = -ramps[:-2] / fdiff[:-1].reshape(-1, 1)
    upper = ramps[2:] / fdiff[1:].reshape(-1, 1)
    weights = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])  # slaney area norm
    weights *= enorm.reshape(-1, 1)
    return weights.astype(np.float32)


@dataclass
class PreprocessArgs:
    sample_rate: int
    normalize: str
    window_size: float
    window_stride: float
    window: str
    features: int
    n_fft: int
    dither: float
    pad_to: int = 0
    pad_value: float = 0
    preemph: float | None = 0.97
    mag_power: float = 2.0

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)

    def __post_init__(self):
        # only slow at first run, should be acceptable to most of users
        # VENDORED CHANGE: local mel_filters (above) instead of librosa.filters.mel.
        self._filterbanks = mx.array(
            mel_filters(self.sample_rate, self.n_fft, self.features),
            dtype=mx.float32,
        )


# VENDORED CHANGE: upstream's load_audio (ffmpeg file decode) was removed —
# chad feeds 16 kHz float32 arrays straight from the mic (speech.Recorder),
# never file paths.


# thanks to https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/audio.py
@functools.lru_cache(None)
def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])


@functools.lru_cache(None)
def hamming(size):
    return mx.array(np.hamming(size + 1)[:-1])


@functools.lru_cache(None)
def blackman(size):
    return mx.array(np.blackman(size + 1)[:-1])


@functools.lru_cache(None)
def bartlett(size):
    return mx.array(np.bartlett(size + 1)[:-1])


def stft(
    x, n_fft, hop_length=None, win_length=None, window=None, axis=-1, pad_mode="reflect"
):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = n_fft // 4
    if window is None:
        window = mx.ones(win_length)

    if win_length != n_fft:
        if win_length > n_fft:
            window = window[:n_fft]
        else:
            padding = [(0, n_fft - win_length)]
            window = mx.pad(window, padding)

    def _pad(x, padding, pad_mode="reflect"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = n_fft // 2
    x = _pad(x, padding, pad_mode)

    strides = [hop_length, 1]
    t = (x.size - win_length + hop_length) // hop_length
    shape = [t, n_fft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def get_logmel(x: mx.array, args: PreprocessArgs) -> mx.array:
    original_dtype = x.dtype

    if args.pad_to > 0:
        if x.shape[-1] < args.pad_to:
            pad_length = args.pad_to - x.shape[-1]
            x = mx.pad(x, ((0, pad_length),), constant_values=args.pad_value)

    if args.preemph is not None:
        x = mx.concat([x[:1], x[1:] - args.preemph * x[:-1]], axis=0)

    window = (
        hanning(args.win_length).astype(x.dtype)
        if args.window == "hann" or args.window == "hanning"
        else hamming(args.win_length).astype(x.dtype)
        if args.window == "hamming"
        else blackman(args.win_length).astype(x.dtype)
        if args.window == "blackman"
        else bartlett(args.win_length).astype(x.dtype)
        if args.window == "bartlett"
        else None
    )
    x = stft(x, args.n_fft, args.hop_length, args.win_length, window)
    abs = mx.abs(mx.view(x, original_dtype))
    x = abs[..., ::2] + abs[..., 1::2]

    if args.mag_power != 1.0:
        x = mx.power(x, args.mag_power)

    x = mx.matmul(args._filterbanks.astype(x.dtype), x.T)
    x = mx.log(x + 1e-5)

    if args.normalize == "per_feature":
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.std(x, axis=1, keepdims=True)
        normalized_mel = (x - mean) / (std + 1e-5)
    else:
        mean = mx.mean(x)
        std = mx.std(x)
        normalized_mel = (x - mean) / (std + 1e-5)

    normalized_mel = normalized_mel.T
    normalized_mel = mx.expand_dims(normalized_mel, axis=0)

    return normalized_mel.astype(original_dtype)
