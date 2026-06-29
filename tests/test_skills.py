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
# Disclosure (catalog) + registry
# ---------------------------------------------------------------------------

def test_catalog_block_empty_when_no_skills(tmp_path):
    assert skills.catalog_block(str(tmp_path)) == ""


def test_catalog_block_lists_skills(tmp_path):
    cwd = tmp_path / "proj"
    _write_skill(str(cwd / ".agents" / "skills"), "alpha",
                 description="Do alpha things. Use for alpha.")
    block = skills.catalog_block(str(cwd))
    assert "<available_skills>" in block
    assert "<name>alpha</name>" in block
    assert "Do alpha things" in block
    assert "activate_skill" in block  # behavioral instruction present
    assert "/SKILL.md" in block       # location present


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
# Activation (tier 2)
# ---------------------------------------------------------------------------

@pytest.fixture
def _in_project(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    monkeypatch.chdir(cwd)  # HOME is already isolated+empty by the autouse fixture
    return cwd


def test_activate_returns_wrapped_body_and_resources(_in_project):
    d = _write_skill(str(_in_project / ".agents" / "skills"), "pdf",
                     body="# PDF\nDo the steps.")
    os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    with open(os.path.join(d, "scripts", "extract.py"), "w") as f:
        f.write("print('hi')")
    out = skills.activate("pdf")
    assert out.startswith('<skill_content name="pdf">')
    assert "Do the steps." in out
    assert "name:" not in out  # frontmatter stripped
    assert "Skill directory:" in out
    assert "scripts/extract.py" in out
    assert "</skill_content>" in out


def test_activate_dedupes(_in_project):
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf")
    first = skills.activate("pdf")
    assert first.startswith("<skill_content")
    second = skills.activate("pdf")
    assert "already active" in second


def test_activate_unknown_name(_in_project):
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf")
    out = skills.activate("nope")
    assert "no skill named" in out
    assert "pdf" in out  # lists what's available


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def test_activate_skill_tool_registered_only_with_skills(_in_project):
    # No skills yet -> tool absent.
    names = [s["function"]["name"] for s in tools.active_schemas()]
    assert "activate_skill" not in names
    # Add one and rebuild registry.
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf")
    skills.reset_session()
    schemas = tools.active_schemas()
    sk = next(s for s in schemas if s["function"]["name"] == "activate_skill")
    enum = sk["function"]["parameters"]["properties"]["name"]["enum"]
    assert enum == ["pdf"]  # constrained to real names


def test_activate_skill_dispatch(_in_project):
    _write_skill(str(_in_project / ".agents" / "skills"), "pdf", body="# PDF\nbody-text")
    out = tools.DISPATCH["activate_skill"]({"name": "pdf"})
    assert "body-text" in out


def test_activate_skill_not_mutating():
    assert "activate_skill" not in tools.MUTATING


# ---------------------------------------------------------------------------
# Compaction protection
# ---------------------------------------------------------------------------

def test_skill_message_detected():
    assert skills.is_skill_message(
        {"role": "tool", "name": "activate_skill", "content": "<skill_content name=\"x\">..."})
    assert skills.is_skill_message(
        {"role": "tool", "name": "read", "content": "<skill_content name=\"x\">leaked"})
    assert not skills.is_skill_message({"role": "tool", "name": "read", "content": "hi"})
    assert not skills.is_skill_message({"role": "user", "content": "hi"})


def test_compaction_does_not_truncate_skill_content():
    big = "INSTRUCTIONS LINE\n" * 500
    skill_content = '<skill_content name="x">\n' + big + "</skill_content>"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do a thing"},
        {"role": "tool", "name": "activate_skill", "content": skill_content},
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
