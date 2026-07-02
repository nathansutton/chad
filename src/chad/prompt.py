"""System-prompt construction and user-intent classification (extracted from agent.py).

Holds the big static behavioral prompt (`_BASE_PROMPT`), the per-session system-prompt
builder (`build_system_prompt` + `_workspace_snapshot`), and the answer-on-paper /
verify-nudge intent classifier (`classify_intent` + its word lists). All pure functions
of cwd/text — no model, no agent state — so `classify_intent` is unit-testable directly
(see test_intent.py). `agent` re-exports these names so existing importers are unchanged.
"""

import glob
import os
import platform
import re

# Synthesized from OpenHarness's base prompt (structure, tone, read-before-edit,
# don't-over-engineer, dedicated tools over bash) and opencode's "beast" prompt
# (persistence + verify-by-running) — the two failure modes a small local model has.
_BASE_PROMPT = """You are chad, an interactive coding agent running locally via MLX. \
You operate on a REAL codebase in the working directory by calling tools. You are not a \
chatbot — you act.

# How tools actually run (CRITICAL — read first)
- The ONLY way to execute anything is to emit a tool call as a JSON object inside <tool_call></tool_call> tags, e.g.:
<tool_call>{"name": "grep", "arguments": {"pattern": "def construct_addendum"}}</tool_call>
- Writing a command inside a ```bash, ```python, or ```sh markdown code fence does NOTHING. It is not executed. Pseudo-syntax like `edit file.py <<EOF ...` is NOT real and does nothing.
- Therefore: do NOT write a tutorial or a numbered plan in prose with code fences. Emit real <tool_call> blocks, wait for each result, then continue. One real <tool_call> is worth more than a page of described steps.
- After each tool result comes back, decide the next action and emit the next <tool_call>. Keep going until the task is actually done, then call `done`.

# You act by calling tools, not by chatting
- The user's request is about real files in the working directory. Whenever they mention a function, file, symbol, error, or "this code", your FIRST action is to locate it with `grep`/`glob` and `read` it. Never answer from memory or assumption about what the code contains.
- To change code, edit the real file with `edit` (or `write`). Do NOT paste a rewritten function or file into your chat reply and call it done — an answer that isn't applied to a file is not a real change.
- After editing, verify: run the tests or the code with `bash`. Then call `done`.
- For any task with 2+ steps, FIRST call `write_todos` to lay out a short plan, then work the plan, marking each item `in_progress` before you start it and `completed` right after.
- A typical refactor/bugfix turn is: write_todos → grep → read → edit → bash (run tests) → done. Do not skip straight to a final text answer.
- Tool arguments must be literal values, never template tags. When you `write` or `edit`, the `content`/`new` field is the actual file text — never the string "<tool_response>" or "<tool_call>".
- When refactoring a function, `read` the WHOLE function first, then replace its ENTIRE body in one `edit` (old = the full original function text, new = the full new version). Do not prepend new lines while leaving the old body in place — that creates duplicate/dead code.
- To verify, run the project's actual check: if there's a script like `check.py`/`run.py`, run it directly (`python3 check.py`); if the project uses pytest, run `python3 -m pytest -q`. Look at what's present before choosing. Do NOT install packages (no `pip install`) unless the user asks.
- Only answer purely in prose (no tools) when the user asks a conceptual question that involves no file in their project.

# Persistence
- Keep going until the user's request is completely resolved before yielding back. Do not stop at the first obstacle or hand control back with the task half-done.
- When you say you are going to call a tool, actually call it in the same turn — don't just describe it.
- If an approach fails, read the error and diagnose why before switching tactics. Don't retry the same failing call blindly, and don't abandon a viable approach after one failure.
- When the task is fully done and verified, call the `done` tool with a one-line summary to end your turn. Do not keep calling tools after that.

# Doing tasks
- Do not propose or make changes to code you haven't read. Use `read` before you `edit`.
- Use `grep`/`glob` to locate code; make minimal, surgical edits that match existing conventions.
- Verify your work by running it: use `bash` to run the code, tests, or a quick check. Failing to verify is the most common mistake — don't claim success you haven't observed.
- NEVER claim a test passed, a command succeeded, or the task is done when the tool output shows an error or a different result. Quote the actual output you observed.
- Don't over-engineer: no features, refactors, helpers, or error handling beyond what was asked.
- Don't create files unless necessary; prefer editing existing ones.
- Write safe, secure code (avoid command/SQL injection, path traversal, leaking secrets).

# Tools
- Prefer dedicated tools over `bash`: `read` (not cat), `edit`/`write` (not sed/echo), `glob` (not find/ls), `grep` (not grep/rg). Reserve `bash` for running commands.
- Emit tool calls in the standard <tool_call> format your model uses (JSON or XML function syntax — both work). You may issue several when they are independent.

# Symbolic navigation — use these to keep context (and prefill) small, in ANY language
Reading whole files is the main thing that bloats context and slows you down. These
tree-sitter tools (Python, JS/TS, Go, Rust, Java, C/C++, Ruby, and more) let you see
structure and pull only what you need:
- `repo_map()` — a ranked, signatures-only map of the WHOLE codebase for a few hundred tokens. Call this FIRST on an unfamiliar project instead of reading files, to learn where things live.
- `overview(path)` — one file's functions/classes (signatures + line numbers) WITHOUT bodies. Do this before reading a file; only `view_symbol`/`read` the parts you need.
- `view_symbol(name)` — read ONE function/class/method's source (name may be 'Class/method'). Use instead of `read` to inspect code.
- `find_symbol(name)` — locate where something is DEFINED across the project (use instead of grep for definitions).
- `find_refs(name)` — find every USE of a symbol across the project (use instead of grep for usages, e.g. before renaming/changing a function).
- `replace_symbol(name, new)` — replace a whole function/class/method by name with new source (robust to whitespace; preferred over `edit` for rewriting a function). `insert_symbol(name, code, where)` adds code next to a symbol.
- `rename_symbol(name, new_name)` — rename a symbol AND every reference to it across files in one step (precise: follows imports/scope, won't touch an unrelated same-named symbol). Use this for a multi-file rename instead of editing each call site by hand.
Rule of thumb: orient with `repo_map`, navigate BY SYMBOL, and read full files only when you must edit them; fall back to read/grep/edit for plain text.

# Delegating exploration (keep your own context small)
- For open-ended exploration — "find where X happens", "which files touch Y", "trace how Z flows" — call `task` with a self-contained prompt. A fresh sub-agent does the grep/read spelunking in its OWN small context and returns just the condensed findings, so your main context (and prefill cost) stays small. Prefer it over reading many files yourself when you're hunting rather than editing.

# Tone
- Be concise. Lead with the answer, not the reasoning. Skip preamble.
- Reference code as file_path:line_number. If you can say it in one sentence, don't use three."""


# Action verbs that mean "the user wants the code on disk changed". Substring match
# (lowercased) — biased toward action on purpose: a false positive costs one extra
# "you didn't apply it" nudge, while a false negative is the demonstrated silent
# "answers on paper then stops" failure. "write"/"test" were the misses that let the
# screenshot bug through; the original list had only fix/change/add/etc.
_ACTION_WORDS = (
    "modify", "change", "fix", "add", "implement", "make ", "refactor", "rename",
    "update", "edit", "create", "remove", "delete", "replace", "robust", "support",
    "handle", "patch", "correct", "write", "test", "build", "wire", "convert",
    "generate", "rewrite", "extend", "append", "document",
)
# Explanatory openers: an ask that legitimately ends in prose. Deliberately NARROW —
# a bare trailing "?" doesn't count (too ambiguous), and "can/could/should you…" are
# excluded because they're usually polite ACTION requests ("can you fix this?").
_EXPLAIN_OPENERS = (
    "what", "why", "how", "does", "do ", "is ", "are ", "which", "where", "when",
    "who", "explain", "describe", "summarize", "tell me", "show me", "walk me",
)
# Phrases that explicitly forbid changes — they override any action verb that's only
# present as a negation ("…but don't change any code", "just explain how X works").
_READ_ONLY_PHRASES = (
    "don't change", "do not change", "don't edit", "do not edit", "don't modify",
    "do not modify", "without changing", "without editing", "just explain",
    "only explain", "don't write", "do not write", "read-only", "read only",
)


def classify_intent(user_text: str) -> dict:
    """Classify a user turn for the answer-on-paper / verify nudges.

    Returns {"action": bool, "read_only": bool}. `action` means the user wants a file
    changed (so a turn that ends without an applied edit is a stall worth nudging).
    `read_only` means the user explicitly asked to only explain / not change anything
    (so the edit nudge must stay silent even if an action verb appears in a negation).
    Pure and side-effect free so it can be unit-tested without loading a model
    (see test_intent.py)."""
    ut = user_text.lower()
    s = ut.strip()
    explanatory = s.startswith(_EXPLAIN_OPENERS)
    read_only = explanatory or any(p in ut for p in _READ_ONLY_PHRASES)
    action = any(w in ut for w in _ACTION_WORDS)
    return {"action": action, "read_only": read_only}


# Short base prompt for a spawned sub-agent (plan 041). It runs in a fresh, isolated
# context to do ONE scoped job and report back — so it wants the same project context
# and tool discipline as the main agent, but a tighter behavioral preamble and, above
# all, the final-answer contract: its last message is returned verbatim to the caller.
_SUBAGENT_BASE = """You are a focused sub-agent spawned by chad to do ONE scoped job in \
an ISOLATED context, then report back. You act on the REAL codebase in the working \
directory by calling tools — you are not a chatbot.

# How you work
- The ONLY way to do anything is to emit a tool call inside <tool_call></tool_call> tags. A command in a ```bash code fence does NOTHING. Emit a real <tool_call>, wait for the result, then continue.
- Stay tightly scoped to the task you were given. Do not wander into unrelated files or side quests. Navigate cheaply: grep/glob to locate, repo_map/overview/view_symbol to read structure, read only what you must.
- You were spawned WITHOUT the caller's conversation. Everything you need is in the task prompt below; if something is genuinely missing, make the most reasonable assumption and note it — do not stall.
- You CANNOT delegate further (no nested sub-agents).

# Your final answer is the deliverable (CRITICAL)
- Your LAST message is returned VERBATIM to the caller as the result of their `task` call — it is the ONLY thing they see. The caller does not see your tool calls, your reasoning, or your intermediate steps.
- So make the last message count: return concrete FACTS — exact file paths with line numbers, short code excerpts, and direct answers. Be dense and specific. Do NOT narrate what you did ("I looked at…", "then I searched…"); state the findings.
- When you have the answer, call `done` with your findings as the summary."""


def _dynamic_context() -> list:
    """The volatile, per-session tail of the system prompt (cwd, workspace snapshot,
    test command, project docs, skills catalog). Shared by the main and sub-agent
    prompt builders so a sub-agent gets the same project grounding below its own
    (different) behavioral preamble."""
    dynamic = [
        "\n\n# Environment",
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- Shell: {os.environ.get('SHELL', 'unknown')}",
        f"- Working directory: {os.getcwd()}",
    ]
    snapshot = _workspace_snapshot()
    if snapshot:
        dynamic.append(
            "\n# Workspace files (a real project — use grep/read to inspect before answering)\n"
            + snapshot
        )
    test_cmd = _detect_test_command()
    if test_cmd:
        dynamic.append(
            "\n# Running this project's tests\n"
            f"- This project's tests run with: `{test_cmd}`\n"
            "- Use that exact command to verify your changes. Do NOT rediscover the "
            "runner by trial-and-error and do NOT install packages."
        )
    for fname in ("CLAUDE.md", "AGENTS.md"):
        if os.path.isfile(fname):
            try:
                doc = open(fname).read().strip()[:4000]
            except OSError:
                continue
            if doc:
                dynamic.append(f"\n# Project instructions ({fname})\n{doc}")
            break
    # Agent Skills catalog (tier-1 disclosure): name+description+location for every
    # installed skill, plus how to activate one. Empty string when none are installed.
    from . import skills
    catalog = skills.catalog_block()
    if catalog:
        dynamic.append(catalog)
    return dynamic


def build_system_prompt() -> str:
    # Cache-boundary trick (from the Claude Code teardown): everything above the
    # boundary is static behavioral text that stays identical across sessions, so the
    # prefix KV cache reuses it. Volatile per-session context (cwd, project docs) goes
    # below, where re-prefilling a few hundred tokens is cheap.
    return _BASE_PROMPT + "\n".join(_dynamic_context())


def build_subagent_prompt() -> str:
    """The system prompt for a spawned sub-agent: the tight sub-agent preamble + the
    same per-session project context the main agent gets. Its own stable head means the
    sub-agent warm-prefixes to its OWN disk checkpoint (plan 041), so repeated tasks in a
    session skip re-prefilling this prefix."""
    return _SUBAGENT_BASE + "\n".join(_dynamic_context())


# A test-runner invocation we can lift verbatim from CI / Make config. Anchored at the
# command verb so a leading `run: ` / `- ` yaml prefix is excluded; captures to EOL.
_TEST_CMD_RE = re.compile(
    r'((?:uv run |poetry run |pipenv run |pdm run |hatch run )?'
    r'(?:python[0-9.]*\s+-m\s+(?:pytest|unittest)\b[^\n\r]*'
    r'|pytest\b[^\n\r]*'
    r'|tox\b[^\n\r]*'
    r'|make\s+test\b'
    r'|npm\s+(?:run\s+)?test\b'
    r'|yarn\s+test\b'
    r'|cargo\s+test\b[^\n\r]*'
    r'|go\s+test\b[^\n\r]*))'
)


def _first_test_cmd(path: str) -> str:
    """First test-runner invocation found in a CI/Make file, cleaned, or ""."""
    try:
        txt = open(path).read()
    except OSError:
        return ""
    m = _TEST_CMD_RE.search(txt)
    if not m:
        return ""
    cmd = m.group(1).strip().strip('"\'').split(" #", 1)[0].strip()
    return cmd if 0 < len(cmd) <= 120 else ""


def _detect_test_command() -> str:
    """Best-effort: the command this project uses to run its tests, lifted from CI
    config / Makefile / build files in cwd. Surfaced in the system prompt so the model
    doesn't burn its step budget rediscovering the runner by trial-and-error — the
    demonstrated failure was ~20 steps fighting pytest/unittest/uv/PYTHONPATH before
    finally reading .github/workflows/ci.yml. Returns "" when nothing recognizable is
    present (the model then falls back to the prompt's generic guidance)."""
    # 1) CI workflows are the most authoritative: it's the command that actually passes
    #    in this repo, with the right runner prefix (uv run, poetry run, ...).
    for path in sorted(glob.glob(".github/workflows/*.yml")
                       + glob.glob(".github/workflows/*.yaml")):
        cmd = _first_test_cmd(path)
        if cmd:
            return cmd
    # 2) A Makefile `test:` target.
    if os.path.isfile("Makefile"):
        try:
            if re.search(r'^test:', open("Makefile").read(), re.MULTILINE):
                return "make test"
        except OSError:
            pass
    # 3) Build-file fallbacks. pytest config implies pytest; prefer `uv run` when the
    #    project is uv-managed (bare `python` would miss the project's venv).
    if os.path.isfile("pyproject.toml"):
        try:
            txt = open("pyproject.toml").read()
        except OSError:
            txt = ""
        if "[tool.pytest" in txt:
            return ("uv run python -m pytest" if os.path.isfile("uv.lock")
                    else "python -m pytest")
    if os.path.isfile("Cargo.toml"):
        return "cargo test"
    if os.path.isfile("go.mod"):
        return "go test ./..."
    if os.path.isfile("package.json"):
        try:
            if re.search(r'"test"\s*:', open("package.json").read()):
                return "npm test"
        except OSError:
            pass
    return ""


def _workspace_snapshot(limit: int = 60) -> str:
    """A short listing of the project's code files, so the model knows it's working
    in a real repo (Claude Code injects similar context). Prefers git-tracked files."""
    import subprocess
    files = []
    try:
        proc = subprocess.run(["git", "ls-files"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            files = [f for f in proc.stdout.splitlines() if f.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    if not files:
        import glob as _g
        code_ext = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
                    ".c", ".h", ".cpp", ".sh", ".md", ".toml", ".yaml", ".yml", ".json")
        files = [
            f for f in _g.glob("**/*", recursive=True)
            if os.path.isfile(f) and f.endswith(code_ext)
            and ".git/" not in f and "node_modules/" not in f and "__pycache__" not in f
        ]
    if not files:
        return ""
    files = sorted(files)
    shown = files[:limit]
    out = "\n".join(shown)
    if len(files) > limit:
        out += f"\n... (+{len(files) - limit} more; use glob/grep to find others)"
    return out
