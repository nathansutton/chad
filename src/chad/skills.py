"""Agent Skills (https://agentskills.io) — discovery, parsing, and user-driven loading.

A *skill* is a directory containing a `SKILL.md` file: YAML frontmatter (at minimum
`name` + `description`) followed by a markdown body of instructions, optionally
bundling `scripts/`, `references/`, and `assets/`. chad discovers and loads them:

  1. Discover — scan project- and user-level skill dirs for `SKILL.md` files.
  2. Parse    — leniently extract frontmatter + body (warn, don't crash, on cosmetic
                issues; skip only when there's no usable description / unparseable YAML).
  3. Dispatch — `slash_commands()` puts every skill in the TUI's `/` menu.
  4. Load     — `load(name, task)` returns the body as text to submit as a USER turn.

chad deliberately does NOT implement the spec's tier-1 disclosure, where a catalog of
every skill's description rides in the system prompt so the model can select one. That
catalog measured 4,751 tokens against 62 installed skills — 60% of the entire system
prompt, permanently, on a box whose usable window is ~50k. The completion menu shows
the same names to the person who already knows which one they want, for nothing. So
selection is the user's (`/ship`), and only what they ask for reaches the model.

State (the cwd-keyed registry and the set of loaded skills) lives at module level — same
pattern as `tools._TODOS` — and is cleared by `reset_session()` when a new Agent /
`/reset` starts so one session's loads don't bleed into the next.
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
# Cap the bundled-resource listing returned at load — a large skill dir shouldn't flood
# the loaded turn; the model reads the specific files it needs on demand anyway.
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
    `order` is the de-duplicated menu order (stable: discovery order); `warnings`
    collects per-skill diagnostics plus shadow/collision notes for surfacing in /skills.
    """
    # CHAD_NO_SKILLS=1 disables discovery entirely, so `/name` resolves to nothing and
    # the menu is bare. The prompt-confound this originally existed for — ~50 user
    # skills injected into every system prompt, differing from a clean environment and
    # shifting a small model's greedy trajectory — is now structurally impossible, since
    # discovery reaches only the completion menu. It stays as the switch for a host that
    # wants its personal skills unreachable from chad at all.
    if os.environ.get("CHAD_NO_SKILLS", "").strip().lower() in ("1", "true", "yes", "on"):
        return {}, [], []
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
_loaded = set()       # names loaded this session (a second copy is dead weight)


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
    """Clear per-session state (loads) and force re-discovery. Called when a new Agent /
    `/reset` starts so a prior session's loaded skills don't leak forward."""
    global _registry, _loaded
    _registry = None
    _loaded = set()


def skill_names():
    """Names of all available skills."""
    return list(get_registry().order)


def has_skills() -> bool:
    return bool(get_registry().order)


# ---------------------------------------------------------------------------
# Dispatch (slash commands) + loading (the skill body as a user turn)
# ---------------------------------------------------------------------------

def slash_commands():
    """`[(command, description)]` for the TUI's `/` menu — one row per installed skill.

    This is the whole of chad's skill *discovery* surface. Skills are invoked by the
    user typing `/name`; the model never selects one from a prompt catalog. The catalog
    this replaced cost 4,751 tokens of every system prompt (60% of it) so a small model
    could guess among dozens of descriptions — when the completion menu already puts the
    same names in front of the person who knows which one they want. This costs nothing
    in context because it never reaches the model.
    """
    reg = get_registry()
    return [(f"/{name}", _one_line(reg.by_name[name].description)[:70])
            for name in reg.order]


def is_skill_command(text: str):
    """Split a typed line into `(skill_name, trailing_args)` when it names a skill, else
    None. `/ship` -> ("ship", ""); `/investigate the flaky test` -> ("investigate", "the
    flaky test"). Matches the leading token exactly, so a builtin like `/model` can
    never be captured by a skill whose name merely starts the same way."""
    if not text.startswith("/") or "\n" in text:
        return None
    head, _, rest = text[1:].partition(" ")
    if head and head in get_registry().by_name:
        return head, rest.strip()
    return None


def _one_line(text: str) -> str:
    """Collapse whitespace so a multi-line description stays on one menu line."""
    return " ".join(text.split())


def _list_resources(base_dir: str, skill_md: str):
    """Relative paths of bundled files in the skill dir (excluding SKILL.md itself),
    capped. These are listed for the model but NOT read — it opens them on demand."""
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


def load(name: str, task: str = "") -> str:
    """The text to submit as a USER turn when someone types `/name`: the skill's body,
    its directory, and a listing of the files it bundles.

    A user turn, not a tool result, is the whole point of the slash-command move. A
    skill is guidance for one task, so it should live where the task lives — droppable
    by compaction when the task is done, instead of welded into the system prompt for
    the session. `task` is whatever the user typed after the command
    (`/investigate the flaky test`) and becomes the concrete ask the instructions apply to.

    Re-invoking a skill already loaded this session returns a short note instead of the
    body: the instructions are still in the transcript, and on a 24 GB box a second copy
    of a 40k-token skill is most of the window.
    """
    reg = get_registry()
    skill = reg.by_name.get(name)
    if skill is None:
        avail = ", ".join(reg.order) or "none installed"
        return f"[no skill named {name!r}. Available skills: {avail}.]"
    if name in _loaded:
        note = (f"[the '{name}' skill's instructions were loaded earlier in this "
                f"conversation and are still in effect.]")
        return f"{note}\n\n{task}" if task else note
    _loaded.add(name)
    log.info("skills: loaded %s (%s)", name, skill.location)

    resources, truncated = _list_resources(skill.base_dir, skill.location)
    parts = [f'<skill name="{name}">', skill.body, "",
             f"Skill directory: {skill.base_dir}",
             "Relative paths in this skill are relative to the skill directory; pass "
             "absolute paths to tools."]
    if resources:
        parts.append("<skill_resources>")
        parts += [f"  <file>{r}</file>" for r in resources]
        if truncated:
            parts.append(f"  <note>listing capped at {_MAX_RESOURCES} files; more exist</note>")
        parts.append("</skill_resources>")
    parts.append("</skill>")
    parts.append("")
    parts.append(f"[I invoked the '{name}' skill. Follow the instructions above for "
                 f"this task.]")
    if task:
        parts.append(task)
    return "\n".join(parts)


# Marker identifying the user turn that carries a skill's instructions, so context
# compaction can protect it from truncation/dropping (skill instructions are durable
# behavioral guidance — silently losing them mid-session degrades the agent with no
# error, and the user paid for them deliberately by typing the command).
SKILL_CONTENT_MARKER = "<skill name="


def is_skill_message(msg: dict) -> bool:
    """True if a conversation message carries a loaded skill's instructions."""
    return SKILL_CONTENT_MARKER in (msg.get("content") or "")


def summary_lines():
    """Human-readable lines for the `/skills` command: one row per skill (name, a
    `*loaded*` marker if already pulled in this session, and a trimmed description), plus
    a footer of any discovery warnings. Empty list message when no skills are installed."""
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
        active = " *loaded*" if name in _loaded else ""
        out.append(f"{name}{active} — {desc}")
    out += warn_footer(reg.warnings)
    return out
