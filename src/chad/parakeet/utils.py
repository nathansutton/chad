import json
import types
import typing
from dataclasses import fields, is_dataclass
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten


# VENDORED CHANGE: replaces dacite.from_dict — the one thing dacite did here.
# The Parakeet config dataclasses are plain nested dataclasses with primitive /
# list / `X | None` fields, so structural recursion covers them: extra dict
# keys are ignored, missing keys fall back to field defaults, and any nested
# dataclass (direct or behind `| None`) is built recursively.
def from_dict(cls, data: dict):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        t = hints.get(f.name, f.type)
        if typing.get_origin(t) in (typing.Union, types.UnionType):
            members = [a for a in typing.get_args(t) if a is not type(None)]
            if len(members) == 1:
                t = members[0]
        if is_dataclass(t) and isinstance(val, dict):
            val = from_dict(t, val)
        kwargs[f.name] = val
    return cls(**kwargs)

from .parakeet import (
    BaseParakeet,
    ParakeetCTC,
    ParakeetCTCArgs,
    ParakeetRNNT,
    ParakeetRNNTArgs,
    ParakeetTDT,
    ParakeetTDTArgs,
    ParakeetTDTCTC,
    ParakeetTDTCTCArgs,
)


def from_config(config: dict) -> BaseParakeet:
    """Loads model from config (randomized weight)"""
    if (
        config.get("target")
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is not None
    ):
        cfg = from_dict(ParakeetTDTArgs, config)
        model = ParakeetTDT(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is not None
    ):
        cfg = from_dict(ParakeetTDTCTCArgs, config)
        model = ParakeetTDTCTC(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        and config.get("model_defaults", {}).get("tdt_durations") is None
    ):
        cfg = from_dict(ParakeetRNNTArgs, config)
        model = ParakeetRNNT(cfg)
    elif (
        config.get("target")
        == "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE"
    ):
        cfg = from_dict(ParakeetCTCArgs, config)
        model = ParakeetCTC(cfg)
    else:
        raise ValueError("Model is not supported yet!")

    model.eval()  # prevents layernorm not computing correctly on inference!

    return model


def from_pretrained(
    hf_id_or_path: str,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    cache_dir: str | Path | None = None,
) -> BaseParakeet:
    """Loads model from Hugging Face or local directory"""
    try:
        config = json.load(
            open(
                hf_hub_download(hf_id_or_path, "config.json", cache_dir=cache_dir), "r"
            )
        )
        weight = hf_hub_download(
            hf_id_or_path, "model.safetensors", cache_dir=cache_dir
        )
    except Exception:
        config = json.load(open(Path(hf_id_or_path) / "config.json", "r"))
        weight = str(Path(hf_id_or_path) / "model.safetensors")

    model = from_config(config)
    model.load_weights(weight)

    # cast dtype
    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model
