"""Tests for Agent Skills support (src/chad/skills.py) + its harness integration.

Covers the four spec responsibilities — discovery, lenient parsing, tier-1 disclosure,
tier-2 activation — plus the cross-cutting concerns the integration guide calls out:
project-over-user precedence, enum-constrained tool registration, dedupe, and protecting
activated skill content from context compaction.
"""

import os
import textwrap

import pytest

from chad import compaction, skills, tools


def _write_skill(root, name, description="Use this when the user mentions widgets.",
                 body="# Do the thing\nStep 1. Step 2.", extra="", dirname=None,
                 name_field="__use_name__"):
    """Create <root>/<dirname or name>/SKILL.md and return its directory."""
    d = os.path.join(root, dirname or name)
    os.makedirs(d, exist_ok=True)
    nm = name if name_field == "__use_name__" else name_field
    fm = [f"name: {nm}"] if nm is not None else []
    fm.append(f"description: {description}")
    if extra:
        fm.append(extra)
    text = "---\n" + "\n".join(fm) + "\n---\n" + body
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(text)
    return d


@pytest.fixture(autouse=True)
def _clean_registry(tmp_path, monkeypatch):
    """Each test starts with no cached registry / activations, and an isolated empty
    HOME so the developer's real ~/.claude/skills never leaks into discovery."""
    empty_home = tmp_path / "_empty_home"
    empty_home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(empty_home) if p == "~" or p.startswith("~/") else p)
    skills.reset_session()
    yield
    skills.reset_session()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_valid_skill():
    text = textwrap.dedent("""\
        ---
        name: pdf-processing
        description: Extract text from PDFs. Use when handling PDF documents.
        license: Apache-2.0
        metadata:
          author: example-org
          version: "1.0"
        ---
        # PDF Processing
        Body instructions here.
    """)
    skill, reason = skills.parse_skill_text(text, "/x/pdf-processing/SKILL.md")
    assert reason is None
    assert skill.name == "pdf-processing"
    assert skill.description.startswith("Extract text")
    assert skill.body.startswith("# PDF Processing")
    assert skill.frontmatter["license"] == "Apache-2.0"
    assert skill.frontmatter["metadata"]["author"] == "example-org"
    assert skill.warnings == []  # name matches dir, description fine


def test_parse_missing_description_is_skipped():
    text = "---\nname: foo\n---\nbody"
    skill, reason = skills.parse_skill_text(text, "/x/foo/SKILL.md")
    assert skill is None
    assert "description" in reason


def test_parse_no_frontmatter_is_skipped():
    skill, reason = skills.parse_skill_text("just a markdown file\n", "/x/foo/SKILL.md")
    assert skill is None
    assert "frontmatter" in reason


def test_parse_unterminated_frontmatter_is_skipped():
    skill, reason = skills.parse_skill_text("---\nname: foo\ndescription: bar\n", "/x/foo/SKILL.md")
    assert skill is None


def test_parse_colon_in_value_fallback():
    # Unquoted colon-space in a value is invalid YAML many clients accept; we recover.
    text = "---\nname: foo\ndescription: Use this skill when: the user asks about PDFs\n---\nbody"
    skill, reason = skills.parse_skill_text(text, "/x/foo/SKILL.md")
    assert reason is None
    assert skill.description == "Use this skill when: the user asks about PDFs"


def test_parse_name_mismatch_warns_but_loads():
    text = "---\nname: other-name\ndescription: A real description here.\n---\nbody"
    skill, reason = skills.parse_skill_text(text, "/x/foo/SKILL.md")
    assert reason is None
    assert skill.name == "other-name"
    assert any("does not match" in w for w in skill.warnings)


def test_parse_missing_name_derives_from_dir():
    text = "---\ndescription: A real description here.\n---\nbody"
    skill, reason = skills.parse_skill_text(text, "/x/my-skill/SKILL.md")
    assert reason is None
    assert skill.name == "my-skill"
    assert any("using parent directory" in w for w in skill.warnings)


# ---------------------------------------------------------------------------
# Discovery + precedence
# ---------------------------------------------------------------------------

def test_discover_project_and_user_scopes(tmp_path):
    cwd = tmp_path / "proj"
    home = tmp_path / "home"
    _write_skill(str(cwd / ".agents" / "skills"), "alpha")
    _write_skill(str(home / ".agents" / "skills"), "beta")
    by_name, order, _ = skills.discover(str(cwd), str(home))
    assert set(by_name) == {"alpha", "beta"}
    assert "alpha" in order and "beta" in order


def test_project_overrides_user(tmp_path):
    cwd = tmp_path / "proj"
    home = tmp_path / "home"
    _write_skill(str(home / ".agents" / "skills"), "code-review",
                 description="USER copy of the skill.")
    _write_skill(str(cwd / ".agents" / "skills"), "code-review",
                 description="PROJECT copy of the skill.")
    by_name, order, warnings = skills.discover(str(cwd), str(home))
    assert by_name["code-review"].description == "PROJECT copy of the skill."
    assert order.count("code-review") == 1
    assert any("shadows user skill" in w for w in warnings)


def test_discover_claude_dir_too(tmp_path):
    cwd = tmp_path / "proj"
    home = tmp_path / "home"
    _write_skill(str(cwd / ".claude" / "skills"), "from-claude-dir")
    by_name, _, _ = skills.discover(str(cwd), str(home))
    assert "from-claude-dir" in by_name


def test_discover_ignores_non_skill_dirs(tmp_path):
    cwd = tmp_path / "proj"
    home = tmp_path / "home"
    sk = tmp_path / "proj" / ".agents" / "skills"
    _write_skill(str(sk), "real")
    # a stray directory with no SKILL.md must be ignored
    os.makedirs(str(sk / "not-a-skill"), exist_ok=True)
    with open(str(sk / "README.md"), "w") as f:
        f.write("not a skill")
    by_name, _, _ = skills.discover(str(cwd), str(home))
    assert set(by_name) == {"real"}


# ---------------------------------------------------------------------------
# Dispatch (slash menu) + registry
# ---------------------------------------------------------------------------

def test_no_skill_text_reaches_the_system_prompt(tmp_path, monkeypatch):
    """The load-bearing guarantee of the slash-command design: however many skills are
    installed, the system prompt does not grow by one token."""
    from chad import prompt as prompt_mod
    cwd = tmp_path / "proj"
    _write_skill(str(cwd / ".agents" / "skills"), "alpha",
                 description="Do alpha things. Use for alpha.")
    monkeypatch.chdir(cwd)
    skills.reset_session()
    built = prompt_mod.build_system_prompt()
    assert "alpha" not in built
    assert "Do alpha things" not in built
    assert "activate_skill" not in built


def test_slash_commands_lists_skills(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    _write_skill(str(cwd / ".agents" / "skills"), "alpha",
                 description="Do alpha things. Use for alpha.")
    monkeypatch.chdir(cwd)
    skills.reset_session()
    assert skills.slash_commands() == [("/alpha", "Do alpha things. Use for alpha.")]


def test_slash_commands_empty_when_no_skills(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skills.reset_session()
    assert skills.slash_commands() == []


def test_is_skill_command_parses_name_and_task(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    _write_skill(str(cwd / ".agents" / "skills"), "alpha")
    monkeypatch.chdir(cwd)
    skills.reset_session()
    assert skills.is_skill_command("/alpha") == ("alpha", "")
    assert skills.is_skill_command("/alpha fix the flaky test") == (
        "alpha", "fix the flaky test")
    assert skills.is_skill_command("/alphabet") is None   # exact leading token only
    assert skills.is_skill_command("/model") is None      # a builtin is never a skill
    assert skills.is_skill_command("alpha") is None       # needs the leading slash
    assert skills.is_skill_command("/alpha\nmore") is None  # single-line only


def test_registry_cached_and_rebuilt_on_cwd_change(tmp_path, monkeypatch):
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    home = tmp_path / "home"
    _write_skill(str(proj_a / ".agents" / "skills"), "skill-a")
    _write_skill(str(proj_b / ".agents" / "skills"), "skill-b")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)

    monkeypatch.chdir(proj_a)
    assert skills.skill_names() == ["skill-a"]
    reg1 = skills.get_registry()
    assert skills.get_registry() is reg1  # cached, same object

    monkeypatch.chdir(proj_b)
    assert skills.skill_names() == ["skill-b"]  # rebuilt for new cwd


# ---------------------------------------------------------------------------
# Loading (the skill body as a user turn)
# ---------------------------------------------------------------------------

@pytest.fixture
def _in_project(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    monkeypatch.chdir(cwd)  # HOME is already isolated+empty by the autouse fixture
    return cwd


def test_load_returns_wrapped_body_and_resources(_in_project):
    d = _write_skill(str(_in_project / ".agents" / "skills"), "pdf",
                     body="# PDF\nDo the steps.")
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    with open(os.path.join(d, "scripts", "extract.py"), "w") as f:
        f.write("print('hi')")
    out = skills.load("pdf")
    assert out.startswith('<skill name="pdf">')
    assert "Do the steps." in out
    assert "name:" not in out  # frontmatter stripped
    assert "Skill directory:" in out
    assert "scripts/extract.py" in out
    assert "</skill>" in out


def test_load_appends_the_users_task(_in_project):
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf")
    out = skills.load("pdf", "fix the flaky test")
    assert out.rstrip().endswith("fix the flaky test")


def test_load_dedupes(_in_project):
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf")
    first = skills.load("pdf")
    assert first.startswith("<skill ")
    second = skills.load("pdf", "and now this")
    assert "loaded earlier in this conversation" in second
    assert "<skill " not in second          # the body is not sent twice
    assert second.endswith("and now this")  # but the new task still gets through


def test_load_unknown_name(_in_project):
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf")
    out = skills.load("nope")
    assert "no skill named" in out
    assert "pdf" in out  # lists what's available


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------

def test_skills_add_no_tool_to_the_model(_in_project):
    """Installing skills must not change the tool surface: selection is the user's."""
    before = [s["function"]["name"] for s in tools.active_schemas()]
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf")
    skills.reset_session()
    after = [s["function"]["name"] for s in tools.active_schemas()]
    assert before == after
    assert "activate_skill" not in after
    assert "activate_skill" not in tools.DISPATCH


# ---------------------------------------------------------------------------
# Compaction protection
# ---------------------------------------------------------------------------

def test_skill_message_detected():
    assert skills.is_skill_message(
        {"role": "user", "content": '<skill name="x">...'})
    assert not skills.is_skill_message({"role": "tool", "name": "read", "content": "hi"})
    assert not skills.is_skill_message({"role": "user", "content": "hi"})
    assert not skills.is_skill_message({"role": "user"})  # content may be absent


def test_compaction_does_not_truncate_skill_content():
    big = "INSTRUCTIONS LINE\n" * 500
    skill_content = '<skill name="x">\n' + big + "</skill>"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do a thing"},
        {"role": "user", "content": skill_content},
        {"role": "tool", "name": "bash", "content": "noise\n" * 2000},
        {"role": "user", "content": "now the active task"},
        {"role": "tool", "name": "bash", "content": "recent\n" * 2000},
    ]
    # render() returns a token-count proxy (list sized to total content length) so the
    # compactor sees "over limit" and runs every pass; ctx_limit forces compaction.
    def render():
        return [0] * (sum(len(m["content"]) for m in messages) // 4)

    emit = lambda *a, **k: None
    compaction.compact_if_needed(messages, render, emit, ctx_limit=100, prompt_ids=render())
    survivor = next(m for m in messages if skills.is_skill_message(m))
    assert big in survivor["content"]  # skill instructions kept verbatim
    assert compaction._COLLAPSED not in survivor["content"]

