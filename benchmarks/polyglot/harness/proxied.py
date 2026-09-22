"""A CLI arm behind its block's proxy: each trial's requests go to a log of their own, and
that log becomes the trial's ATIF trajectory when the harness writes none itself.

Nothing the harness sees changes — same argv, same home, same sandbox; the `Endpoint` it
was built with already points at the proxy. What the wrapper adds is what only the proxy
knows: the tokens the server generated, how many generation requests the trial made and
how many of them were side requests beside the agent loop, and how the sampler check
came out. `start()` also records the block's sampler audit, then empties every slot of
the server, so each arm's block starts on a cold prefix cache whatever ran before it.

chad-llama writes a trajectory of its own; the proxy's reading of the same trial is kept
beside its output (`<slug>.rep<k>.proxy.json`) for the step-23 cross-check.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping

import proxy_atif
from catalog import JsonValue, is_object
from proxy import Proxy, audit, erase_slots

from harness import AtifDoc, Solved, Trial
from harness.cli import CliHarness


def _write(path: str, doc: Mapping[str, JsonValue]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


class Proxied:
    def __init__(self, inner: CliHarness, proxy: Proxy,
                 count_tokens: Callable[[str], int] | None = None):
        self.inner, self.proxy, self.count_tokens = inner, proxy, count_tokens
        self.name, self.root = inner.name, inner.root
        self._doc: AtifDoc | None = None

    def version(self) -> str:
        return self.inner.version()

    def start(self) -> Mapping[str, JsonValue]:
        meta = dict(self.inner.start())
        meta["sampler_forced"] = dict(self.proxy.forced)
        meta["sampler_audit"] = dict(audit(self.proxy, self.inner.endpoint.model))
        meta["slots_erased"] = erase_slots(self.proxy.upstream)
        meta["think_tokens"] = "counted with the served tokenizer" if self.count_tokens \
            else "not counted"
        return meta

    def prepare(self, home: str) -> None:
        self.inner.prepare(home)

    def solve(self, trial: Trial) -> Solved:
        stem = os.path.splitext(trial.output)[0]
        requests = stem + ".requests.jsonl"
        if os.path.exists(requests):
            os.remove(requests)        # a trial re-run after a crash starts its log over
        self.proxy.route(requests)
        try:
            solved = self.inner.solve(trial)
        finally:
            self.proxy.route("")
        records = proxy_atif.load(requests) if os.path.exists(requests) else []
        native = self.inner.trajectory()
        self._doc = native
        fields: dict[str, JsonValue] = dict(solved.fields)
        if records:
            doc = proxy_atif.convert(records, self.name, self.version(),
                                     self.inner.endpoint.model, self.count_tokens)
            if native is None:
                _write(trial.trajectory, doc)
                self._doc = doc
            else:
                _write(stem + ".proxy.json", doc)
            final = doc.get("final_metrics")
            totals = final if is_object(final) else {}
            extra = totals.get("extra")
            extras = extra if is_object(extra) else {}
            side = extras.get("side")
            fields.update({
                "gen_tokens": totals.get("total_completion_tokens"),
                "requests": extras.get("requests"),
                "side_requests": side.get("requests") if is_object(side) else 0,
                "sampler": extras.get("sampler")})
        else:
            fields.update({"gen_tokens": 0, "requests": 0, "side_requests": 0, "sampler": {}})
        return Solved(solved.wall_s, solved.capped, fields)

    def trajectory(self) -> AtifDoc | None:
        return self._doc
