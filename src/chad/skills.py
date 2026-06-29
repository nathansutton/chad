"""Agent Skills support (https://agentskills.io) — discovery, parsing, disclosure, activation.

A *skill* is a directory containing a `SKILL.md` file: YAML frontmatter (at minimum
`name` + `description`) followed by a markdown body of instructions, optionally
bundling `scripts/`, `references/`, and `assets/`. This module implements the four
client-side responsibilities from the spec's "adding skills support" guide:

  1. Discover  — scan project- and user-level skill dirs for `SKILL.md` files.
  2. Parse     — leniently extract frontmatter + body (warn, don't crash, on cosmetic
                 issues; skip only when there's no usable description / unparseable YAML).
  3. Disclose  — `catalog_block()` lists name+description+location for the system prompt
                 (tier 1: ~50-100 tokens/skill, loaded at session start).
  4. Activate  — `activate(name)` returns the full body wrapped in <skill_content> with a
                 bundled-resource listing (tier 2: loaded only when the model asks).

Progressive disclosure means the base context stays small: the model sees the catalog
always, pulls a skill's instructions only when a task matches, and reads bundled
resources (scripts/refs) on demand via the normal `read` tool.

State (the cwd-keyed registry and the set of already-activated skills) lives at module
level — same pattern as `tools._TODOS` — and is cleared by `reset_session()` when a new
Agent / `/reset` starts so activations don't bleed across sessions.
"""

import os

from .diag import log, warn_footer
from .ignore import IGNORE_DIRS

# Bound the discovery walk so a pathological tree (a skills dir nested in a huge repo)
# can't stall startup. Mirrors the spec's "reasonable bounds" guidance.
_MAX_DEPTH = 4
_MAX_DIRS = 2000
# Directories never worth descending into during a skill scan (shared canonical set).
_SKIP_DIRS = set(IGNORE_DIRS)
# Cap the bundled-resource listing returned at activation — a large skill dir shouldn't
# flood the activation result; the model reads specific files on demand anyway.
_MAX_RESOURCES = 50


class Skill:
    """One discovered skill: the parsed metadata plus where it lives on disk."""

    __slots__ = ("name", "description", "location", "base_dir", "body",
                 "frontmatter", "warnings")

    def __init__(self, name, description, location, body, frontmatter, warnings):
        self.name = name
        self.description = description
        self.location = location              # absolute path to SKILL.md
        self.base_dir = os.path.dirname(location)
        self.body = body                      # markdown after the frontmatter
        self.frontmatter = frontmatter        # full parsed dict (license/compat/metadata…)
        self.warnings = warnings              # lenient-validation diagnostics


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _split_frontmatter(text: str):
    """Return (frontmatter_yaml, body) for a `---`-delimited file, or (None, text).

    The opening `---` must be the first line; the closing `---` is the next line that
    is exactly `---`. Everything between is YAML; everything after, trimmed, is the body.
    """
    # Tolerate a leading BOM / blank lines before the first delimiter.
    stripped = text.lstrip("﻿")
    lines = stripped.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = "".join(lines[1:i])
            body = "".join(lines[i + 1:]).strip()
            return fm, body
    return None, text  # opened but never closed -> treat as bodyless/invalid


def _quote_colon_values(fm: str) -> str:
    """Best-effort fix for the most common cross-client YAML breakage: an unquoted
    scalar whose value contains a colon-space (`description: Use when: foo`), which a
    strict YAML parser rejects. Wrap such top-level values in double quotes and retry.
    Only touches obvious `key: value` lines; leaves nested/structured YAML alone."""
    out = []
    for ln in fm.splitlines():
        stripped = ln.strip()
        # key at column 0 (no indent), value present, value contains a colon-space,
        # and isn't already quoted / a block scalar / a mapping opener.
        if (ln[:1] not in (" ", "\t", "#", "")
                and ":" in stripped):
            key, _, val = ln.partition(":")
            val_s = val.strip()
            if (val_s and ": " in val_s
                    and val_s[0] not in ("'", '"', "|", ">", "[", "{")):
                esc = val_s.replace("\\", "\\\\").replace('"', '\\"')
                out.append(f'{key}: "{esc}"')
                continue
        out.append(ln)
    return "\n".join(out)


def parse_skill_text(text: str, location: str):
    """Parse one SKILL.md's text. Returns (Skill, None) on success or (None, reason) when
    the skill must be skipped (no usable description, or unparseable YAML). Lenient: a bad
    `name` is repaired/derived and recorded as a warning rather than rejected."""
    import yaml

    fm_text, body = _split_frontmatter(text)
    if fm_text is None:
        return None, "no YAML frontmatter (missing or unterminated `---` block)"
    try:
        meta = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        try:
            meta = yaml.safe_load(_quote_colon_values(fm_text))
        except yaml.YAMLError as e:
            return None, f"unparseable YAML frontmatter: {e}"
    if not isinstance(meta, dict):
        return None, "frontmatter is not a key-value mapping"

    warnings = []
    desc = meta.get("description")
    desc = desc.strip() if isinstance(desc, str) else ""
    if not desc:
        # A description is essential for disclosure — without it the model can never
        # know when to activate the skill. Skip (the one hard failure besides bad YAML).
        return None, "missing or empty `description`"
    if len(desc) > 1024:
        warnings.append(f"description exceeds 1024 chars ({len(desc)})")

    dir_name = os.path.basename(os.path.dirname(location))
    name = meta.get("name")
    name = name.strip() if isinstance(name, str) else ""
    if not name:
        name = dir_name
        warnings.append("missing `name`; using parent directory name")
    if name != dir_name:
        warnings.append(f"name {name!r} does not match directory {dir_name!r}")
    if len(name) > 64:
        warnings.append(f"name exceeds 64 chars ({len(name)})")

    return Skill(name, desc, location, body, meta, warnings), None


def parse_skill_file(location: str):
    """Read and parse a SKILL.md at `location`. Returns (Skill, None) or (None, reason)."""
    try:
        with open(location, "r", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return None, f"cannot read: {e}"
    return parse_skill_text(text, os.path.abspath(location))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _scope_roots(cwd: str, home: str):
    """The directories to scan, lowest precedence first. Project-level skills override
    user-level ones (the universal convention), so projects come last and win on merge.
    Both the cross-client `.agents/skills` convention and the pragmatic `.claude/skills`
    location are scanned at each level."""
    return [
        ("user", os.path.join(home, ".claude", "skills")),
        ("user", os.path.join(home, ".agents", "skills")),
        ("project", os.path.join(cwd, ".claude", "skills")),
        ("project", os.path.join(cwd, ".agents", "skills")),
    ]


def _find_skill_dirs(root: str):
    """Yield directories under `root` that directly contain a `SKILL.md`, bounded by
    depth and a total-dir cap so a huge tree can't stall the scan."""
    if not os.path.isdir(root):
        return
    seen = 0
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        seen += 1
        if seen > _MAX_DIRS:
            log.warning("skills: hit %d-dir scan cap under %s; some skills may be missed",
                        _MAX_DIRS, root)
            return
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth >= _MAX_DEPTH:
            dirnames[:] = []
        # prune noise dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if "SKILL.md" in filenames:
            yield os.path.join(dirpath, "SKILL.md")


def discover(cwd: str = None, home: str = None):
    """Scan all scopes and return (skills_by_name, order, warnings).

    `skills_by_name` maps name -> Skill with project-over-user precedence applied;
    `order` is the de-duplicated catalog order (stable: discovery order); `warnings`
    collects per-skill diagnostics plus shadow/collision notes for surfacing in /skills.
    """
    cwd = cwd or os.getcwd()
    home = home or os.path.expanduser("~")
    by_name = {}
    order = []
    warnings = []
    for scope, root in _scope_roots(cwd, home):
        for location in sorted(_find_skill_dirs(root)):
            skill, reason = parse_skill_file(location)
            if skill is None:
                log.warning("skills: skipping %s — %s", location, reason)
                warnings.append(f"{location}: skipped ({reason})")
                continue
            for w in skill.warnings:
                warnings.append(f"{skill.name}: {w}")
            if skill.name in by_name:
                prev = by_name[skill.name]
                # Same name seen again. Project beats user; otherwise first-found wins.
                prev_scope = "project" if _is_project(prev.location, cwd) else "user"
                if scope == "project" and prev_scope == "user":
                    warnings.append(
                        f"{skill.name}: project skill shadows user skill at {prev.location}")
                    by_name[skill.name] = skill  # replace in place; keep catalog position
                    log.info("skills: %s (project) overrides user copy", skill.name)
                else:
                    warnings.append(
                        f"{skill.name}: duplicate at {location} shadowed by {prev.location}")
                    log.info("skills: %s duplicate at %s ignored", skill.name, location)
                continue
            by_name[skill.name] = skill
            order.append(skill.name)
    return by_name, order, warnings


def _is_project(location: str, cwd: str) -> bool:
    cwd_abs = os.path.abspath(cwd)
    return os.path.abspath(location).startswith(cwd_abs + os.sep)


# ---------------------------------------------------------------------------
# Registry (cwd-cached) + session activation state
# ---------------------------------------------------------------------------

class _Registry:
    def __init__(self, cwd):
        self.cwd = cwd
        self.by_name, self.order, self.warnings = discover(cwd)


_registry = None      # cached _Registry for the current cwd
_activated = set()    # names activated this session (for dedupe)


def get_registry() -> "_Registry":
    """The skill registry for the current cwd, discovered once and cached. Rebuilt
    automatically if the working directory changed since the last call."""
    global _registry
    cwd = os.getcwd()
    if _registry is None or _registry.cwd != cwd:
        _registry = _Registry(cwd)
        if _registry.order:
            log.info("skills: discovered %d skill(s): %s",
                     len(_registry.order), ", ".join(_registry.order))
    return _registry


def reset_session():
    """Clear per-session state (activations) and force re-discovery. Called when a new
    Agent / `/reset` starts so a prior session's activated skills don't leak forward."""
    global _registry, _activated
    _registry = None
    _activated = set()


def skill_names():
    """Names of all available skills (used to constrain the activate_skill tool enum)."""
    return list(get_registry().order)


def has_skills() -> bool:
    return bool(get_registry().order)


# ---------------------------------------------------------------------------
# Disclosure (tier 1) + activation (tier 2)
# ---------------------------------------------------------------------------

_CATALOG_INSTRUCTIONS = (
    "# Skills\n"
    "The following skills provide specialized instructions for specific tasks. Each "
    "lists what it does and when to use it. When a task matches a skill's description, "
    "call the `activate_skill` tool with the skill's `name` to load its full "
    "instructions BEFORE proceeding — do not guess the procedure from the name alone. "
    "A skill's instructions may reference bundled files (scripts/, references/, "
    "assets/); read those on demand with the `read` tool, resolving relative paths "
    "against the skill directory reported when you activate it.\n"
)


def catalog_block(cwd: str = None) -> str:
    """The tier-1 skills section for the system prompt: behavioral instructions plus a
    compact <available_skills> catalog (name + description + location). Returns "" when
    no skills are installed, so an empty block never confuses the model."""
    reg = get_registry() if cwd is None else _Registry(cwd)
    if not reg.order:
        return ""
    lines = ["\n\n" + _CATALOG_INSTRUCTIONS, "<available_skills>"]
    for name in reg.order:
        s = reg.by_name[name]
        lines.append("  <skill>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{_one_line(s.description)}</description>")
        lines.append(f"    <location>{s.location}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _one_line(text: str) -> str:
    """Collapse whitespace so a multi-line description stays on one catalog line."""
    return " ".join(text.split())


def _list_resources(base_dir: str, skill_md: str):
    """Relative paths of bundled files in the skill dir (excluding SKILL.md itself),
    capped. These are surfaced to the model but NOT read — it loads them on demand."""
    files = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            if os.path.abspath(full) == os.path.abspath(skill_md):
                continue
            rel = os.path.relpath(full, base_dir)
            files.append(rel)
            if len(files) >= _MAX_RESOURCES:
                truncated = True
                break
        if truncated:
            break
    return sorted(files), truncated


def activate(name: str) -> str:
    """Tier-2 activation: return the named skill's instructions wrapped in identifying
    tags, with its directory and a bundled-resource listing. Frontmatter is stripped
    (the body is the actionable part). Already-activated skills return a short note
    instead of re-injecting the same instructions."""
    reg = get_registry()
    skill = reg.by_name.get(name)
    if skill is None:
        avail = ", ".join(reg.order) or "none installed"
        return (f"[no skill named {name!r}. Available skills: {avail}. "
                f"Call activate_skill with one of those exact names.]")
    if name in _activated:
        return (f"[skill '{name}' is already active earlier in this conversation — its "
                f"instructions are still in effect; no need to re-activate.]")
    _activated.add(name)
    log.info("skills: activated %s (%s)", name, skill.location)

    resources, truncated = _list_resources(skill.base_dir, skill.location)
    parts = [f'<skill_content name="{name}">', skill.body, "",
             f"Skill directory: {skill.base_dir}",
             "Relative paths in this skill are relative to the skill directory; pass "
             "absolute paths to tools."]
    if resources:
        parts.append("<skill_resources>")
        parts += [f"  <file>{r}</file>" for r in resources]
        if truncated:
            parts.append(f"  <note>listing capped at {_MAX_RESOURCES} files; more exist</note>")
        parts.append("</skill_resources>")
    parts.append("</skill_content>")
    return "\n".join(parts)


# Marker that identifies an activated-skill tool result, so context compaction can
# protect it from truncation/dropping (skill instructions are durable behavioral
# guidance — silently losing them mid-session degrades the agent with no error).
SKILL_CONTENT_MARKER = "<skill_content "


def is_skill_message(msg: dict) -> bool:
    """True if a conversation message carries activated skill instructions."""
    return (msg.get("role") == "tool"
            and (msg.get("name") == "activate_skill"
                 or SKILL_CONTENT_MARKER in msg.get("content", "")))


def summary_lines():
    """Human-readable lines for the `/skills` command: one row per skill (name, an
    `*active*` marker if already loaded, and a trimmed description), plus a footer of
    any discovery warnings. Empty list message when no skills are installed."""
    reg = get_registry()
    if not reg.order:
        return ["no skills installed. Add a SKILL.md under .agents/skills/ (project) or "
                "~/.agents/skills/ (user). See https://agentskills.io"]
    out = []
    for name in reg.order:
        s = reg.by_name[name]
        desc = _one_line(s.description)
        if len(desc) > 100:
            desc = desc[:97] + "…"
        active = " *active*" if name in _activated else ""
        out.append(f"{name}{active} — {desc}")
    out += warn_footer(reg.warnings)
    return out
