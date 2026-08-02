"""Every behavior the registry can switch off must also be OBSERVABLE when it acts.

Plan 123's ladder: a lever can exist (registry entry), be exercised (a `levers.fired()`
event in the trace), or improve an outcome (a measured contrast). The registry made rung
one enforceable; this gate makes rung two enforceable. Without it, an ablation arm that
shows "no effect" for a lever cannot distinguish exercised-and-useless (retire it) from
never-fired-on-these-tasks (untested — the verdict does not transfer), which is the
difference between deleting scaffolding and deleting evidence.

Mechanics: a lever declares `fires=EVENT` (default) or `fires=PASSIVE` (its whole effect
is static prompt content, where "fired" would only restate the config record). This gate
statically scans the source so that:

  1. every EVENT lever has at least one literal `levers.fired("<name>"` call site;
  2. every `fired()` literal names a registered lever (a typo'd name would otherwise
     count firings for a lever that does not exist — the same lie the UnknownLever
     check closes for CHAD_DISABLE);
  3. no PASSIVE lever has a fired() site (the annotation and the code must agree).

A new lever therefore cannot merge half-instrumented: add the registry entry, and either
call `fired()` at its action site or annotate `fires=PASSIVE`, or this gate is red.
"""

import re
from pathlib import Path

from chad import levers

SRC = Path(levers.__file__).parent
FIRED_LITERAL = re.compile(r'levers\.fired\(\s*"([a-z0-9_]+)"|(?<!\.)\bfired\(\s*"([a-z0-9_]+)"')


def _fired_names() -> dict[str, list[str]]:
    """lever name -> list of file:line sites where a fired() literal names it."""
    sites: dict[str, list[str]] = {}
    for py in sorted(SRC.glob("*.py")):
        if py.name == "levers.py":
            continue  # the definition, not a call site
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            for m in FIRED_LITERAL.finditer(line):
                name = m.group(1) or m.group(2)
                sites.setdefault(name, []).append(f"{py.name}:{lineno}")
    return sites


def test_every_event_lever_has_a_fired_site():
    sites = _fired_names()
    missing = sorted(n for n, lv in levers.LEVERS.items()
                     if lv.fires == levers.EVENT and n not in sites)
    assert not missing, (
        f"EVENT levers with no levers.fired() call site: {missing}. Either instrument "
        f"the action site or annotate the registry entry fires=PASSIVE.")


def test_every_fired_literal_is_registered():
    sites = _fired_names()
    unknown = sorted(n for n in sites if n not in levers.LEVERS)
    assert not unknown, (
        f"fired() literals naming unregistered levers (typo?): "
        f"{ {n: sites[n] for n in unknown} }")


def test_no_passive_lever_fires():
    sites = _fired_names()
    contradicted = sorted(n for n, lv in levers.LEVERS.items()
                          if lv.fires == levers.PASSIVE and n in sites)
    assert not contradicted, (
        f"levers annotated fires=PASSIVE but with fired() sites: "
        f"{ {n: sites[n] for n in contradicted} } — drop the annotation or the calls.")


def test_fired_emits_structured_event_and_counts(caplog, monkeypatch):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    name = next(iter(levers.LEVERS))
    before = levers.fire_counts().get(name, 0)
    with caplog.at_level("INFO", logger="chad"):
        levers.fired(name, step=7, why="test")
    assert levers.fire_counts()[name] == before + 1
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("LEVER "))
    import json
    evt = json.loads(line[len("LEVER "):])
    assert evt["lever"] == name and evt["step"] == 7 and evt["why"] == "test"
    assert "disabled" not in evt


def test_fired_flags_disabled_leak(caplog, monkeypatch):
    """A fire while the lever is disabled means an enabled() guard leaked — recorded,
    not crashed, and loud enough for an arm's analysis to reject the trial."""
    name = next(iter(levers.LEVERS))
    monkeypatch.setenv("CHAD_DISABLE", name)
    with caplog.at_level("INFO", logger="chad"):
        levers.fired(name)
    import json
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("LEVER "))
    assert json.loads(line[len("LEVER "):]).get("disabled") is True
