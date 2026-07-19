"""Per-model profiles: the line between core harness and model-specific accommodation.

chad has always had exactly one prompt, so every accommodation for Ornith's quirks sat
in the same text as the general agent contract, indistinguishable from it. That is fine
for a product that ships one model. It is not fine for harness-bench, whose same-engine
comparison rests on chad not being secretly shaped around the model it ships with — and
it is not fine for a leaderboard entry, where a reviewer is entitled to ask which half of
the score is harness and which half is a patch for this model's habits.

So: a change belongs in **core** when it fires on a real condition any agent would hit
(a full-page read, a zero-match grep), and in a **profile** when it encodes what one
model needs. The discipline is pushing each change as far toward core as it honestly
goes, because that is the version that still pays after this model is gone.

A profile block is strictly ADDITIVE and never load-bearing. The `<tool_call>` emission
contract stays in the base prompt even though it is Ornith-shaped, because a run with
`--profile generic` must still be able to call tools — an ablation that bricks the agent
measures nothing. What lives here is the surplus: nudges that exist because of a
pathology we watched this model exhibit.

Resolution order: `CHAD_PROFILE` env → match against the model id → `ornith` when the
model is unknown (chad ships Ornith; an unlabelled local run is an Ornith run).
"""

import os
from dataclasses import dataclass, field

from . import levers

# Extracted VERBATIM from prompt._BASE_PROMPT. It is here rather than there because it
# exists for one reason: Ornith emits tool calls whose `content` field is the literal
# string "<tool_response>", which sends the turn into a garbled-tool-call death spiral.
# That is a property of this model's post-training, not of coding agents in general.
_ORNITH_BLOCK = (
    "\n\n# Tool-call hygiene\n"
    "- Tool arguments must be literal values, never template tags. When you `write` or "
    "`edit`, the `content`/`new` field is the actual file text — never the string "
    '"<tool_response>" or "<tool_call>".'
)


@dataclass(frozen=True)
class Profile:
    name: str
    # Appended to the system prompt above the volatile cache boundary, so it stays inside
    # the reusable KV prefix (it is constant for a given model, like the base prompt).
    prompt_block: str = ""
    # Levers this profile turns off by default. A profile may not turn a lever ON that
    # CHAD_DISABLE turned off: the env is the ablation driver's channel and must win, or
    # leave-one-out silently tests nothing.
    disables: frozenset[str] = field(default_factory=frozenset)


ORNITH = Profile(name="ornith", prompt_block=_ORNITH_BLOCK)
GENERIC = Profile(name="generic")

PROFILES: dict[str, Profile] = {p.name: p for p in (ORNITH, GENERIC)}


def resolve(model_id: str | None = None) -> Profile:
    """The profile for this run. An explicit CHAD_PROFILE always wins so a benchmark can
    force `generic` against an Ornith endpoint and measure what the accommodations buy."""
    forced = os.environ.get("CHAD_PROFILE")
    if forced:
        if forced not in PROFILES:
            raise ValueError(
                f"unknown CHAD_PROFILE={forced!r}; known: {sorted(PROFILES)}")
        return PROFILES[forced]
    if model_id and "ornith" in model_id.lower():
        return ORNITH
    if model_id:
        return GENERIC
    return ORNITH


def prompt_block(model_id: str | None = None) -> str:
    """The active profile's additive prompt text, or "" when the profile_prompt lever is
    off. Gated here rather than at the call site so every prompt builder inherits it."""
    if not levers.enabled("profile_prompt"):
        return ""
    return resolve(model_id).prompt_block
