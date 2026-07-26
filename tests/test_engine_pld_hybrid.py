"""Bit-exactness of the hybrid (recurrent-snapshot) speculative rollback.

Ornith's cache is not trimmable: its Gated-DeltaNet layers hold recurrent state that
`trim_prompt_cache` cannot rewind. The engine rolls a rejected draft back a different
way — snapshot the recurrent layers by reference, native-trim the attention KV, re-feed
the accepted prefix — and that path had no automated coverage: the existing tier-2 case
in `test_engine.py` needs a downloaded hybrid checkpoint and self-skips everywhere else,
which in practice meant everywhere. The cache path the shipped model actually uses was
the one nothing tested.

Two things make coverage possible without weights:

1. **A tiny synthetic qwen3_5, left UNQUANTIZED.** Quantization is what forces the
   tier-2 test to skip: a coarse logit grid produces exact argmax ties (top1 == top2)
   whose winner is decided by sub-ULP noise from how tokens are batched into forward
   passes, so two legitimate greedy decodes of the same weights can disagree. Off the
   grid, bit-equality is meaningful again.

2. **Injected drafts.** A random-weight model never repeats an n-gram, so real prompt
   lookup proposes nothing and the rollback is never entered (measured: 0 drafts over
   40 tokens). The invariant under test is "the cache is restored exactly, whatever the
   drafter proposed", so the draft quality is fixed deliberately instead of hoped for:
   all-correct (pure accept), partly-correct (partial rollback), all-wrong (full
   rollback). Each must reproduce plain greedy decoding token for token.

Every case asserts the rollback actually ran — an accept/reject count of zero means the
test proved nothing, which is how this path went uncovered in the first place.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models import cache as cache_utils  # noqa: E402

from chad import engine as eng_mod  # noqa: E402
from chad.engine import Engine  # noqa: E402
from test_mlx_fastpath import TINY_CFG  # noqa: E402

PROMPT = [5, 9, 11, 22, 33, 44, 55, 66, 77, 88, 101, 102, 7]
N_TOKENS = 40
BOGUS = 3          # a token the model will not greedily emit, so a draft is rejected


@pytest.fixture(scope="module")
def tiny():
    """The tiny hybrid WITHOUT `nn.quantize` — see the module docstring."""
    import copy

    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    cfg = copy.deepcopy(TINY_CFG)
    cfg["text_config"]["head_dim"] = 64
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(cfg))
    model.eval()
    return model


class _Detok:
    """The four-method slice of mlx_lm's detokenizer that the PLD path touches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.text = ""
        self.last_segment = ""

    def add_token(self, t):
        self.last_segment = f" {t}"
        self.text += self.last_segment

    def finalize(self):
        pass


class _Tok:
    eos_token_id = None
    eos_token_ids = []

    def __init__(self):
        self.detokenizer = _Detok()


def _engine(model):
    """An Engine around an already-built model: `load()` wants a weights directory."""
    eng = object.__new__(Engine)
    eng.model = model
    eng.tok = _Tok()
    eng.kv_bits = None                 # PLD requires an unquantized cache
    eng.temp = 0.0                     # ...and greedy decoding
    eng.min_p = eng.top_p = 0.0
    eng.prompt_lookup = True
    eng.pld_num_draft = 8
    eng.pld_ngram = 3
    eng.enable_pld_hybrid = True       # off by default (slower on real work), on to test
    eng.cache_dir = None
    eng.max_context = None
    eng.effective_ctx = 4096
    eng.model_id = "tiny-hybrid"
    eng._model_path = ""
    eng._cached_ids = []
    eng._rewind_snap = None
    eng._cache_stack = []
    eng._warm_prefix_ids = None
    eng.kv_bytes_per_token = 0.0
    eng._is_moe = False
    eng._n_layers = 0
    eng._reset_cache()
    return eng


def _greedy(model, prompt_ids, n):
    """Plain one-token-at-a-time argmax on a fresh cache — the reference PLD must equal."""
    cache = cache_utils.make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    out = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(n - 1):
        logits = model(mx.array([out[-1]])[None], cache=cache)
        out.append(int(mx.argmax(logits[0, -1]).item()))
    return out


def _run(mode, model, reference):
    """Decode with drafts of a fixed quality; return (tokens, stats)."""
    prompt_len = len(PROMPT)

    def fake_draft(context, num_draft, ngram_max=3, ngram_min=1):
        # Respect num_draft: the engine's own adaptive width (cold probe vs latched-wide)
        # is part of the path under test, not something to bypass.
        if num_draft <= 0:
            return []
        generated = len(context) - prompt_len
        actual = reference[generated: generated + num_draft]
        if len(actual) < num_draft:
            return []
        if mode == "correct":
            return list(actual)
        if mode == "partial":
            return list(actual[:2]) + [BOGUS] * (num_draft - 2)
        return [BOGUS] * num_draft

    real = eng_mod.prompt_lookup_draft
    eng_mod.prompt_lookup_draft = fake_draft
    try:
        eng = _engine(model)
        text, stats = eng._generate_prompt_lookup(PROMPT, N_TOKENS, None, None)
    finally:
        eng_mod.prompt_lookup_draft = real
    return [int(t) for t in text.split()], stats


def test_cache_is_the_hybrid_shape(tiny):
    """Guard the premise: if the tiny model ever stopped producing a non-trimmable
    mixed ArraysCache/KVCache, every test below would still pass while exercising the
    ordinary trimmable path instead of the rollback."""
    eng = _engine(tiny)
    assert not eng._trimmable, "expected a non-trimmable (recurrent) cache"
    assert eng._pld_hybrid, "hybrid rollback would never engage"
    kinds = {type(c).__name__ for c in eng._cache}
    assert "ArraysCache" in kinds and "KVCache" in kinds, kinds


def test_all_drafts_rejected_is_bit_exact(tiny):
    """Full rollback: restore recurrent state, trim the KV, re-feed only the pending
    token. The token after a total rejection is where an off-by-one shows up first."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("wrong", tiny, ref)
    rejected = stats.draft_proposed - stats.draft_accepted
    assert rejected > 0, "no draft was rejected — the rollback never ran"
    assert stats.draft_accepted == 0, stats.draft_accepted
    assert got == ref, f"diverged at {next(i for i, (a, b) in enumerate(zip(ref, got)) if a != b)}"


def test_partially_accepted_drafts_are_bit_exact(tiny):
    """The subtle case: the cache must end up holding the pending token PLUS exactly the
    accepted prefix — not the whole draft, not none of it."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("partial", tiny, ref)
    assert stats.draft_accepted > 0, "nothing accepted — partial rollback never ran"
    assert stats.draft_proposed > stats.draft_accepted, "nothing rejected"
    assert got == ref, f"diverged at {next(i for i, (a, b) in enumerate(zip(ref, got)) if a != b)}"


def test_fully_accepted_drafts_are_bit_exact(tiny):
    """No rollback at all: the accepted run plus its bonus token must commit cleanly, and
    the multi-token verify forward must leave the same state as N single-token steps."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    got, stats = _run("correct", tiny, ref)
    assert stats.draft_accepted > 0, "nothing accepted"
    assert stats.draft_accepted == stats.draft_proposed, "expected no rejections"
    assert got == ref, f"diverged at {next(i for i, (a, b) in enumerate(zip(ref, got)) if a != b)}"


def test_accepting_drafts_costs_fewer_forwards(tiny):
    """PLD's whole reason to exist. Accepted drafts must cover the same 40 tokens in
    strictly fewer model calls than rejected ones, or the hybrid path is pure overhead
    and `enable_pld_hybrid` should stay off for correctness reasons rather than speed."""
    ref = _greedy(tiny, PROMPT, N_TOKENS)
    calls = {}
    # Patch the CLASS, not the instance: `model(...)` resolves __call__ on the type, so
    # an instance attribute is silently ignored and every count comes back 0.
    cls = type(tiny)
    original = cls.__call__
    for mode in ("correct", "wrong"):
        n = [0]

        def counted(self, *a, __orig=original, __n=n, **k):
            __n[0] += 1
            return __orig(self, *a, **k)

        cls.__call__ = counted
        try:
            got, _ = _run(mode, tiny, ref)
        finally:
            cls.__call__ = original
        assert got == ref
        calls[mode] = n[0]
    assert calls["correct"] > 0, "forward counter never fired"
    assert calls["correct"] < calls["wrong"], calls
