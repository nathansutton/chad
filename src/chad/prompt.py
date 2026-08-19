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

# Premise: the model already knows the unix toolbox and the exact-match editor
# dialect from pretraining, so the prompt's job shrinks to two things — the
# tool-call emission contract and context economy, which lives in three bash
# habits (wc first, ranged sed, never cat big files). Nothing here teaches a
# chad-specific dialect.
_BASE_PROMPT = """You are chad, an interactive coding agent running locally via MLX. \
You operate on a REAL codebase in the working directory by calling tools. You are not a \
chatbot — you act.

# How tools actually run (CRITICAL — read first)
- The ONLY way to execute anything is to emit a real tool call, in the exact <tool_call>/<function=…> format specified above: one <function=…> block per call, one <parameter=…> block per argument.
- Writing a command inside a ```bash, ```python, or ```sh markdown code fence does NOTHING. It is not executed.
- Therefore: do NOT write a tutorial or a numbered plan in prose with code fences. Emit real <tool_call> blocks, wait for each result, then continue. One real <tool_call> is worth more than a page of described steps.
- After each tool result comes back, decide the next action and emit the next <tool_call>. Keep going until the task is actually done, then call `done`.

# Your tools
- `bash` — your primary tool: locate, read, run, verify. You know the unix toolbox; use it directly.
- `edit` — change an existing file by exact text replacement (`old` → `new`).
- `write` — create a new file (whole content).
- `write_todos` — record/update a short plan for any task with 2+ steps; call it first, keep statuses current.
- `done` — end your turn when the task is complete and verified, with a one-line summary.

# Plan first, then work the plan
- For any task with 2+ steps, your FIRST call is `write_todos` laying out a short plan; then work it, marking each item `in_progress` before you start it and `completed` right after.
- A typical turn is: write_todos → bash (locate, then read the region) → edit → bash (run the project's tests) → done. Do not skip straight to a final text answer.

# Working in bash (context is scarce — read SMALL)
- Locate code with ripgrep: `rg -n 'pattern' src/` (add `-C 2` for context, `-l` for files only). Find files with `rg --files | rg name` or `ls`.
- Orient in a file: `wc -l file` first, then read only the region you need: `sed -n '120,180p' file`. Only `cat -n` a file you know is short (under ~100 lines).
- Never dump whole large files or flood output — long output is clipped and wastes your context. Narrow the path, use `head`, `rg -m`.
- Run and verify with the project's real commands (its test runner, its build) straight from bash.

# Editing files
- Change existing files ONLY with the `edit` tool: `old` is the exact current text copied from what you just read — including its tabs/spaces — and `new` is the replacement. Include enough surrounding lines to make `old` unique in the file.
- Do NOT modify existing files via `sed -i`, `awk -i`, `perl -i`, `echo >>`, or shell redirection — quoting and indentation get mangled. (Heredocs/redirection are fine for creating new scratch scripts.)
- When refactoring a function, read the WHOLE function first, then replace its entire body in one `edit` (old = the full original function text). Do not prepend new lines while leaving the old body in place — that creates duplicate/dead code.
- Whenever the user mentions a function, file, symbol, error, or "this code", your FIRST action is to locate it with bash and read it. Never answer from memory or assumption; never propose changes to code you haven't read.
- To change code, edit the real file. Do NOT paste a rewritten function into your chat reply and call it done — an answer that isn't applied to a file is not a real change.

# Persistence
- Keep going until the user's request is completely resolved before yielding back. Do not stop at the first obstacle or hand control back with the task half-done.
- If an approach fails, read the error and diagnose why before switching tactics. Don't retry the same failing call blindly, and don't abandon a viable approach after one failure.
- When the task is fully done and verified, call the `done` tool with a one-line summary to end your turn. Do not keep calling tools after that.

# Doing tasks
- Make minimal, surgical edits that match existing conventions; before choosing HOW to fix, check how the repo already handles the same pattern and imitate that precedent.
- Verify your work by running it. NEVER claim a test passed or the task is done when the tool output shows an error or a different result. Quote the actual output you observed.
- Fixing a reported bug: FIRST write and run a minimal script/test that reproduces it and see it FAIL. Only then edit. After the fix, re-run it and assert the expected behavior actually holds, then run the project's real tests.
- Don't over-engineer: no features, refactors, helpers, or error handling beyond what was asked. Don't create files unless necessary.
- Write safe, secure code (avoid command/SQL injection, path traversal, leaking secrets).
- Only answer purely in prose (no tools) when the user asks a conceptual question that involves no file in their project.

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
    "generate", "rewrite", "extend", "append", "document", "save",
)
# Run-task verbs: the ask is to put the SYSTEM in a state (start a service, boot an
# image, install a package), not to land a file edit — completable with zero edits, so
# these must NOT feed the `action` class (whose no-empty-diff done gate demands a landed
# edit; arming it here would hard-block every legitimate zero-edit completion). They get
# their own `run` class, which arms only the anti-bail nudges (a qemu-startup task
# was classified neither action nor read-only, so a prose give-up with
# 733s left sailed through the weakest path).
_RUN_WORDS_RE = re.compile(
    r"\b(start|launch|boot|run|serve|deploy|install|configure|mount|restart|enable"
    r"|spin up|bring up|set up)\b")
# Explanatory openers: an ask that legitimately ends in prose. Deliberately NARROW —
# a bare trailing "?" doesn't count (too ambiguous), and "can/could/should you…" are
# excluded because they're usually polite ACTION requests ("can you fix this?").
_EXPLAIN_OPENERS = (
    "what", "why", "how", "does", "do ", "is ", "are ", "which", "where", "when",
    "who", "explain", "describe", "summarize", "tell me", "show me", "walk me",
)
# Phrases that explicitly forbid changes — they override any action verb that's only
# present as a negation ("…but don't change any code", "just explain how X works").
# Split by whether a scoped target can neutralize them: a change-verb negation can be
# aimed at one file ("do not edit main.tex"); "just explain" / "read-only" cannot.
_READ_ONLY_NEGATIONS = (
    "don't change", "do not change", "don't edit", "do not edit", "don't modify",
    "do not modify", "without changing", "without editing", "don't write",
    "do not write",
)
_READ_ONLY_GLOBAL = ("just explain", "only explain", "read-only", "read only")
_READ_ONLY_PHRASES = _READ_ONLY_NEGATIONS + _READ_ONLY_GLOBAL
# …unless the negation names a SPECIFIC file/object — then it's a scope constraint on
# an action task, not a read-only ask. A LaTeX task ("Do not edit main.tex or
# synonyms.txt" — the task is to edit input.tex) and query-optimize ("Do not modify the
# database file in any way" — the task writes /app/sol.sql) were both classified
# read_only, which disarmed the no-empty-diff gate, the deliverable recheck, and the
# done-audit on two real wrong-done failures. A concrete filename (dot-extension or /
# path) or a definite-article object ("the database file") within a few words of the
# negation marks it scoped; "any code" / "anything" stay global prohibitions.
_SCOPED_NEGATION_RE = re.compile(
    r"[^\n.;]{0,40}?(?:\S*[/.]\w{1,8}\b"
    r"|\b(?:the|this|that)\b[^\n.;]{0,30}?"
    r"\b(?:files?|database|db|repo|repository|director(?:y|ies)|folders?|tables?|"
    r"scripts?)\b)")
# A demand to deliver the answer INTO a file ("write the number … to /app/answer.txt",
# "save the results to a file") — overrides an explanatory opener. The measured miss:
# "Tell me how many … write the integer to the file /app/answer.txt" classified
# read_only, which disarmed every no-progress gate and let a garbled final step end the
# task with an empty diff. Explicit negations
# (_READ_ONLY_PHRASES) still win: they are OR'd in separately below.
# Periods are allowed in the gap (abbreviations like `e.g.` and quoted examples sit
# mid-sentence in real task text); newlines and ?/! still bound the match.
_FILE_DEMAND_RE = re.compile(
    r"\b(?:write|save|store|append|output|put)\b[^\n?!]{0,120}?"
    r"\b(?:to|into|in)\s+(?:the\s+|a\s+)?(?:file\b|/\S+|\S+\.\w{1,8}\b)")


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
    file_demand = bool(_FILE_DEMAND_RE.search(ut))

    def _global_negation(phrase: str) -> bool:
        # Every occurrence of the phrase must be checked: one scoped mention
        # ("do not edit main.tex") must not mask a later global one.
        return any(not _SCOPED_NEGATION_RE.match(ut, i + len(phrase))
                   for i in range(len(ut))
                   if ut.startswith(phrase, i))

    read_only = ((explanatory and not file_demand)
                 or any(p in ut for p in _READ_ONLY_GLOBAL)
                 or any(_global_negation(p) for p in _READ_ONLY_NEGATIONS if p in ut))
    action = any(w in ut for w in _ACTION_WORDS)
    # `run`: a system-state imperative (start/boot/serve/…). Kept SEPARATE from
    # `action` — run tasks are completable with zero file edits, so they arm the
    # anti-bail nudges but never the edit-demanding no-empty-diff gates.
    run = bool(_RUN_WORDS_RE.search(ut)) and not explanatory
    return {"action": action, "read_only": read_only, "run": run}


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
            "\n# Workspace files (a real project — inspect with bash (rg / sed -n) "
            "before answering)\n" + snapshot
        )
    manifest = _env_manifest()
    if manifest:
        dynamic.append(
            "\n# Environment manifest (detected at session start; later installs are "
            "NOT reflected here)\n" + manifest
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
                # errors="replace": these are prompt text, not code we execute — a
                # non-UTF-8 project doc must not crash build_system_prompt (agent won't
                # construct); replacement chars in a stray byte are harmless here.
                doc = open(fname, encoding="utf-8", errors="replace").read().strip()[:4000]
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
    return (_BASE_PROMPT + _verify_baseline_block() + "\n".join(_dynamic_context()))


def _verify_baseline_block() -> str:
    """Ask for one test run BEFORE the first edit. The pre-existing failures of a real
    project are unknowable after the fact — the measured cost of finding out late was
    nine turns of re-running subsets and grepping the suite for `test.failing` markers
    to decide which of two failures the model had caused. The run is also what the
    verify_baseline ambient line has to recall; without it there is nothing recorded.
    Static for the session, so it sits above the cache boundary."""
    from . import levers
    if not levers.enabled("verify_baseline"):
        return ""
    return ("\n- Run the project's test command ONCE before your first edit and read "
            "the result. A suite you have not seen pass cannot tell you later whether "
            "a failure is yours or the project's, and that question is unanswerable "
            "after you have edited — the one run up front is cheaper than the turns "
            "spent reconstructing it.\n")


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
        txt = open(path, encoding="utf-8", errors="replace").read()
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
            if re.search(r'^test:', open("Makefile", encoding="utf-8", errors="replace").read(), re.MULTILINE):
                return "make test"
        except OSError:
            pass
    # 3) Build-file fallbacks. pytest config implies pytest; prefer `uv run` when the
    #    project is uv-managed (bare `python` would miss the project's venv).
    if os.path.isfile("pyproject.toml"):
        try:
            txt = open("pyproject.toml", encoding="utf-8", errors="replace").read()
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
            if re.search(r'"test"\s*:', open("package.json", encoding="utf-8", errors="replace").read()):
                return "npm test"
        except OSError:
            pass
    return ""


def _env_manifest() -> str:
    """The lever-gated (`env_manifest`) session-start toolchain inventory — the
    environment analogue of the workspace map, answering the which/--version/pip-list
    probe class before it is asked (probes measured fail-enriched 3.33 vs
    2.46/trial). Built once per session in ambient.py, so the sub-agent prompt reuses
    it without re-probing; "" (no block) when off or nothing detected."""
    from . import ambient
    try:
        return ambient.env_manifest()
    except Exception:  # noqa: BLE001 - orientation is best-effort; never block startup
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
