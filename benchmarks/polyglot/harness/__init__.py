"""The one contract every agent under test meets, so a block of trials is the same loop
whoever is solving them.

An arm is chad in this process (`chad_inprocess`) or any coding agent with a headless
command line (`cli`, driven by an entry in `harnesses.py`). The runner keeps everything a
comparison depends on — the task, its workspace, the prompt, the wall cap, verification
and the row — and asks the harness only for what differs between agents:

* `start()`: once per block, load or resolve the agent, and return what `meta.json`
  should record about it;
* `prepare()`: write the agent's configuration into the trial's throwaway home;
* `solve()`: run the agent in the workspace until it stops or the wall cap does;
* `trajectory()`: the trial's ATIF document, when the agent writes one itself. `None`
  means the arm has no native trajectory.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from catalog import JsonValue

AtifDoc = Mapping[str, JsonValue]


@dataclass(frozen=True)
class Trial:
    """One task, one rep, as the runner laid it out. Paths are absolute."""

    workspace: str
    home: str                    # a throwaway HOME: nothing exists there until prepare()
    prompt: str
    wall_cap_s: int
    trajectory: str              # where the trial's ATIF document belongs
    output: str                  # where the agent's own stdout and stderr go


@dataclass(frozen=True)
class Solved:
    wall_s: float
    capped: bool                 # stopped by the wall cap, not by the agent
    fields: Mapping[str, JsonValue]   # the arm's own row fields, in row order


class Harness(Protocol):
    name: str                    # the `--harness` value, and the `harness` of meta.json
    root: str                    # the directory whose `_work/` and `_home/` hold its trials

    def version(self) -> str: ...

    def start(self) -> Mapping[str, JsonValue]: ...

    def prepare(self, home: str) -> None: ...

    def solve(self, trial: Trial) -> Solved: ...

    def trajectory(self) -> AtifDoc | None: ...
