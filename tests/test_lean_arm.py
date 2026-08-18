"""The CHAD_LEAN arm: five tools, and a prompt that only ever names those five.

Lean is an ablation — hide everything the model would have to learn in-context and
keep only what it already knows from pretraining (the unix toolbox, one exact-match
editor). The failure mode that makes such an arm measure nothing is silent drift
between the two halves of it: prompt text that keeps advertising a tool the schema no
longer carries. Every occurrence found so far was a real one — the workspace map
pointed at `repo_map`/`view_symbol`, and the skills catalog told the model to open
bundled files "with the `read` tool" — and each costs the model a wasted call or a
skipped affordance while looking, in the trace, like the model's own mistake.

So the invariant here is mechanical rather than a list of known spots: every chad tool
name the model-facing text mentions must be a tool that arm actually exposes. The model
sees three such channels and the contract covers all three — the system prompt, the tool
RESULTS, and the tool SCHEMAS. The schema channel was the one this file originally
missed, and it is the worst place to miss: a description is the most-attended text about
its own tool, and `edit` — the lean arm's ONLY editor — shipped telling the model to
prefer `replace_lines` with line numbers "from read". The schema tests below are written
against every arm chad can be run in, not just lean, because the same hole silently
distorts a hide-one-dialect A/B: the surviving editor advertises the one just removed.
"""

import os
import re

import pytest

from chad import prompt, skills, tools


def _isolated_home(monkeypatch, tmp_path):
    """Keep the developer's own ~/.claude/skills out of the catalog under test."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" else p)


@pytest.fixture
def lean(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAD_LEAN", "1")
    monkeypatch.chdir(tmp_path)
    _isolated_home(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture
def full(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAD_LEAN", raising=False)
    monkeypatch.chdir(tmp_path)
    _isolated_home(monkeypatch, tmp_path)
    return tmp_path


def _exposed():
    return {s["function"]["name"] for s in tools.active_schemas()}


def _all_tool_names():
    """Every name chad can call. `done` and `task` are terminal/agent-level tools with no
    DISPATCH entry, so a check built from DISPATCH alone would be blind to them."""
    return set(tools.DISPATCH) | {s["function"]["name"] for s in tools.SCHEMAS}


def _named_tools(text):
    """chad tool names the prompt mentions in backticks — the form it uses to tell the
    model to call something. A bare word like "read" in prose is not a tool reference
    and must not be one here, or the invariant below would fight English."""
    return {m for m in re.findall(r"`([a-z_]+)`", text) if m in _all_tool_names()}


# Tool names that cannot be mistaken for ordinary English when a description says them
# bare. The rest ("read", "edit", "write", "done", "task", "hover", "definition",
# "overview") only count as references in a call shape — `name` or name( — because a
# description is prose and "read a type in one line" must not read as a tool call.
_UNAMBIGUOUS = {n for n in _all_tool_names() if "_" in n} | {"grep", "glob"}


def _referenced_tools(description):
    """Tool names `description` tells the model to call: any name in a call shape, plus
    the unambiguous ones written bare."""
    names = _all_tool_names()
    found = {m for m in re.findall(r"`([a-z_]+)`", description) if m in names}
    found |= {m for m in re.findall(r"\b([a-z_]+)\(", description) if m in names}
    found |= {n for n in names & _UNAMBIGUOUS
              if re.search(rf"\b{re.escape(n)}\b", description)}
    return found


# Every arm chad can be run in: the shipped default, the two ablations, and the
# edit-dialect isolations the edit benchmark drives. Each is (id, env).
_ARMS = [
    ("full", {}),
    ("lean", {"CHAD_LEAN": "1"}),
    ("no_symbols", {"CHAD_NO_SYMBOLS": "1"}),
    ("no_task", {"CHAD_NO_TASK": "1"}),
    ("dialect_line", {"CHAD_HIDE_TOOLS": "edit"}),
    ("dialect_exact", {"CHAD_HIDE_TOOLS": "replace_lines,insert_lines"}),
]


def test_lean_exposes_only_the_pretrained_surface(lean):
    exposed = _exposed()
    assert exposed >= {"bash", "edit", "write", "write_todos", "done"}
    # everything else chad can dispatch is hidden — dispatch itself is untouched
    assert exposed - {"activate_skill"} == tools._LEAN
    assert "repo_map" not in exposed and "read" not in exposed and "grep" not in exposed


def test_full_arm_keeps_the_whole_surface(full):
    exposed = _exposed()
    assert {"read", "grep", "repo_map", "view_symbol"} <= exposed


@pytest.mark.parametrize("arm", ["lean", "full"])
def test_prompt_never_names_a_tool_the_arm_hides(arm, monkeypatch, tmp_path, request):
    request.getfixturevalue(arm)
    # The <available_skills> rows are third-party text (a skill author's own
    # description), not chad's instructions to the model — only what chad writes is
    # under this contract.
    text = re.sub(r"<available_skills>.*</available_skills>", "",
                  prompt.build_system_prompt(), flags=re.S)
    missing = _named_tools(text) - _exposed()
    assert not missing, (
        f"the {arm} system prompt tells the model to call {sorted(missing)}, "
        f"which that arm does not expose")


@pytest.mark.parametrize("arm_id,env", _ARMS, ids=[a[0] for a in _ARMS])
def test_schemas_never_name_a_tool_the_arm_hides(arm_id, env, monkeypatch, tmp_path):
    """The prompt contract, on the schema channel. This is the one that shipped broken:
    under lean, `edit` is the only editor and its description told the model to prefer
    `replace_lines` with the line numbers "from read". It matters beyond lean — the
    dialect arms exist to measure one editor against the other, and until this test the
    surviving editor kept advertising the one the arm had just removed."""
    monkeypatch.chdir(tmp_path)
    _isolated_home(monkeypatch, tmp_path)
    for var in ("CHAD_LEAN", "CHAD_NO_SYMBOLS", "CHAD_NO_TASK", "CHAD_HIDE_TOOLS"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    schemas = tools.active_schemas()
    exposed = {s["function"]["name"] for s in schemas}
    for s in schemas:
        dangling = _referenced_tools(s["function"]["description"]) - exposed
        assert not dangling, (
            f"in the {arm_id} arm, the `{s['function']['name']}` schema tells the model "
            f"to call {sorted(dangling)}, which that arm does not expose")


def test_every_tool_stays_consistent_when_hidden_alone(monkeypatch, tmp_path):
    """The arms above are the ones we run; this one is the general statement, so a new
    cross-reference between two tool descriptions cannot land un-gated just because no
    named arm happens to separate them."""
    monkeypatch.chdir(tmp_path)
    _isolated_home(monkeypatch, tmp_path)
    for var in ("CHAD_LEAN", "CHAD_NO_SYMBOLS", "CHAD_NO_TASK"):
        monkeypatch.delenv(var, raising=False)
    for hidden_one in sorted(tools.DISPATCH):
        monkeypatch.setenv("CHAD_HIDE_TOOLS", hidden_one)
        schemas = tools.active_schemas()
        exposed = {s["function"]["name"] for s in schemas}
        for s in schemas:
            dangling = _referenced_tools(s["function"]["description"]) - exposed
            assert not dangling, (
                f"hiding `{hidden_one}` alone leaves the `{s['function']['name']}` "
                f"schema pointing at {sorted(dangling)}")


def test_xref_spans_still_occur_in_their_descriptions():
    """The gating table holds EXACT substrings of the descriptions it edits. Reword a
    description without updating the table and the span silently stops matching — the
    clause then survives into every arm, which is the original bug wearing a new hat."""
    by_name = {s["function"]["name"]: s["function"]["description"] for s in tools.SCHEMAS}
    for name, spans in tools._XREF_SPANS.items():
        assert name in by_name, f"_XREF_SPANS names `{name}`, which is not a tool schema"
        for span, needs, _repl in spans:
            assert span in by_name[name], (
                f"the `{name}` description no longer contains the gated span {span!r} "
                f"(gated on {sorted(needs)}) — update _XREF_SPANS with the new wording")


def test_full_surface_descriptions_are_untouched(full):
    """The gating machinery must be inert when nothing is hidden. The full surface is
    what every banked benchmark number was produced against; a description that drifts
    here is an unmeasured change to the shipped default."""
    rendered = {s["function"]["name"]: s["function"]["description"]
                for s in tools.active_schemas()}
    for s in tools.SCHEMAS:
        name = s["function"]["name"]
        assert rendered[name] == s["function"]["description"], (
            f"the full arm re-rendered `{name}`'s description")


def test_tool_results_never_prescribe_a_hidden_tool(lean):
    """The same contract as the prompt, on the result channel. A failed `edit` is the
    most consequential message in this arm — it is the only editor — and it shipped four
    times in a measured run telling the model to fall back on `replace_lines` with line
    numbers "from read", neither of which exists here."""
    f = lean / "m.ts"
    f.write_text("export function alpha() {\n\treturn 1;\n}\n")
    results = [
        tools.tool_edit(str(f), "'export function alpha() {", "x"),   # no match
        tools.tool_edit(str(f), "\treturn 1;", "\treturn 1;"),        # no-op
        tools.tool_edit(str(f), "nothing like this exists", "x"),     # no close line
        tools.tool_write(str(f), "export function alpha() {\n\treturn 2;\n}\n"),
    ]
    hidden = set(tools.DISPATCH) - _exposed()
    for text in results:
        named = {h for h in hidden if h in text}
        assert not named, f"result prescribes hidden tool(s) {sorted(named)}: {text[:200]}"


def test_skills_catalog_points_at_a_reader_that_exists(lean, monkeypatch):
    """A skill's bundled scripts/references are opened by whatever read path the arm
    has. Under lean that is bash; naming `read` there sends the model at a tool that
    isn't in its schema."""
    root = os.path.join(str(lean), ".agents", "skills", "alpha")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "SKILL.md"), "w") as f:
        f.write("---\nname: alpha\ndescription: Do alpha things.\n---\nBody.\n")

    block = skills.catalog_block(str(lean))
    assert "sed -n" in block and "`read` tool" not in block

    monkeypatch.delenv("CHAD_LEAN", raising=False)
    assert "`read` tool" in skills.catalog_block(str(lean))


def test_catalog_carries_every_skill_at_one_line_each(tmp_path, monkeypatch):
    """Name and description survive (the description IS the trigger text the model
    matches a task against); the per-skill XML framing and the location line do not.
    On a host with ~60 skills that framing was a third of the largest block in the
    system prompt, restating a path `activate()` reports itself."""
    _isolated_home(monkeypatch, tmp_path)
    root = os.path.join(str(tmp_path), ".agents", "skills")
    for name in ("alpha", "beta", "gamma"):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "SKILL.md"), "w") as f:
            f.write(f"---\nname: {name}\ndescription: Handles {name}.\n---\nBody.\n")

    block = skills.catalog_block(str(tmp_path))
    rows = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert rows == ["- alpha: Handles alpha.", "- beta: Handles beta.",
                    "- gamma: Handles gamma."]
    assert "<skill>" not in block and "SKILL.md" not in block


# --- the think ceiling is an arm decision, not a global one -------------------

def _agent_ceiling(monkeypatch, lean, env=None):
    """The ceiling a real Agent resolves for this arm. Built through Agent.__init__ on a
    scripted engine rather than re-deriving the rule here — a test that reimplements the
    logic it checks would pass even if agent.py never learned it."""
    from chad.agent import Agent
    from test_agent_e2e import ScriptedEngine
    monkeypatch.delenv("CHAD_THINK_CEILING", raising=False)
    if lean:
        monkeypatch.setenv("CHAD_LEAN", "1")
    else:
        monkeypatch.delenv("CHAD_LEAN", raising=False)
    if env is not None:
        monkeypatch.setenv("CHAD_THINK_CEILING", env)
    return Agent(ScriptedEngine([]), mode="yolo", thinking=False).think_ceiling


def test_lean_does_not_force_close_thinking(monkeypatch, tmp_path):
    """Force-closing </think> mid-generation is the most invasive thing the harness does
    to the token stream, and it is not a lever — it is a constructor default, so it
    appears in neither `chad levers` nor CHAD_DISABLE. The lean arm exists to measure the
    model plus a tool loop with the scaffolding removed, so inheriting it silently
    measured the scaffold. Measured on the shipped base before this changed: 20-38% of
    steps force-closed, against a value sized to bite a ~10% tail on another base."""
    from chad import guardrails
    monkeypatch.chdir(tmp_path)
    _isolated_home(monkeypatch, tmp_path)
    assert _agent_ceiling(monkeypatch, lean=True) == 0
    assert _agent_ceiling(monkeypatch, lean=False) == guardrails.THINK_CEILING_DEFAULT


def test_an_explicit_ceiling_still_wins_in_either_arm(monkeypatch, tmp_path):
    """The ceiling-on lean arm has to stay one env var away, or the A/B that would
    justify either default cannot be run."""
    monkeypatch.chdir(tmp_path)
    _isolated_home(monkeypatch, tmp_path)
    assert _agent_ceiling(monkeypatch, lean=True, env="384") == 384
    assert _agent_ceiling(monkeypatch, lean=False, env="0") == 0


# --- contracts the ablation must not drop by accident -------------------------

@pytest.mark.parametrize("arm", ["lean", "full"])
def test_both_arms_ask_for_a_plan_first(arm, monkeypatch, tmp_path, request):
    """Lean subtracts TOOLS; it must not silently subtract behavioral contracts that
    have nothing to do with the tool surface. `write_todos` is one of lean's five tools,
    but the plan-first instruction lived only in the full prompt — lean got a clause
    inside its tool list. Measured across a day of dogfood: the full arm called
    write_todos at step 0, while lean never planned in 4 of 7 runs and, in its 93-step
    runaway, wrote the plan at step 71 — after the work, just before a 20-step side
    quest. Same tool, different contract, different behavior."""
    request.getfixturevalue(arm)
    text = prompt.build_system_prompt()
    assert "write_todos" in _named_tools(text) or "write_todos" in text
    # the imperative, not just a mention in a tool list
    assert re.search(r"FIRST call `?write_todos`?|your FIRST call is `?write_todos`?", text), (
        f"the {arm} prompt never tells the model to plan FIRST")
    # a worked turn sequence that starts with the plan
    assert re.search(r"write_todos\s*→", text), (
        f"the {arm} prompt shows no worked turn sequence beginning with write_todos")
