"""Regression tests for the safety-critical gate in run_turn.

Tier 1 (always runs, no model): characterizes the three highest-blast-radius branches
whose *wiring* had no test — pinning behavior that is correct today so a future refactor
can't silently re-open the hole:

1. The plan-mode / confirm gate (agent.py ~905): in mode='plan' every mutating tool is
   blocked EXCEPT a write/edit under ./plans/. The escape predicate `_under_plans`
   (tools.py) decides what "under ./plans/" means; a path-normalization regression
   (`plans/../secret.py` escaping) would let plan mode scribble anywhere. We table-test
   `_under_plans` (incl. `..` and sibling-prefix escapes) and the gate's compose logic.
2. The destructive-bash seatbelt in `_confirm` (agent.py ~390): a catastrophic shell
   command (`rm -rf ~`, `curl|sh`) is screened even in --yolo mode, and headless
   with no confirm channel it BLOCKS rather than runs. The predicate
   (`guardrails.is_destructive_bash`) is tested elsewhere; this pins the run-vs-block
   WIRING — the part a refactor that dropped `not sys.stdin.isatty()` would break.

These characterize behavior; if any comes out RED that is a real bug (a broken guard),
not a test to rewrite green — STOP and report (STOP conditions).

Run: `uv run python -m pytest tests/test_gate.py -q`
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad.tools import _under_plans, is_mutating  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


# --- Agent construction without weights (mirrors test_subagent._NoModelEngine) ------------
class _NoModelEngine:
    """Stand-in engine for Agent construction: Agent.__init__ never calls the model, and
    _confirm reads nothing from the engine. Enough to build an Agent and probe the gate."""
    tok = None
    model_id = "test/model"
    cache_dir = None


def _mk_agent(**kw):
    from chad.agent import Agent
    return Agent(_NoModelEngine(), **kw)


# === 1. _under_plans: the plan-mode write-escape predicate ==================================

def test_under_plans_accepts_and_rejects():
    """`_under_plans(path)` is the ONLY thing standing between plan mode's write/edit and
    the whole filesystem. It must accept paths that resolve inside ./plans/ and reject
    everything else — crucially after `..`/symlink-style normalization, so a crafted
    `plans/../outside.py` cannot escape. (All paths are relative so the assertions are
    cwd-independent: both the path and the ./plans/ root share the same cwd base.)"""
    # Accepted: the dir itself and any file within it (any depth).
    check("plans dir itself is under plans", _under_plans("plans") is True)
    check("a plan file is under plans", _under_plans("plans/048-x.md") is True)
    check("a nested plan path is under plans", _under_plans("plans/sub/x.md") is True)

    # Rejected escapes — these are the security-critical cases.
    check("`..` escape is rejected", _under_plans("plans/../outside.py") is False,
          "plans/../outside.py must normalize OUT of ./plans/")
    check("absolute path outside cwd is rejected", _under_plans("/etc/passwd") is False)
    check("sibling-prefix (plans-evil) is rejected", _under_plans("plans-evil/x.py") is False,
          "startswith must be guarded by a trailing os.sep")
    check("unrelated relative path is rejected", _under_plans("src/chad/agent.py") is False)
    check("empty path is rejected", _under_plans("") is False)


# === 2. The plan-mode gate: is_mutating + _under_plans compose as run_turn wires them ========

# run_turn's plan-mode arm (agent.py ~905-911) is an inline boolean, not an extractable
# function, so we transcribe its exact expression here — built on the REAL predicates — and
# table-test it. A reviewer changing the live gate must keep this transcription in step.
def _plan_gate(mode, name, args):
    """'blocked' | 'allowed' — mirror of run_turn's plan-mode gate expression."""
    plan_write = (mode == "plan" and name in ("write", "edit")
                  and _under_plans(args.get("path", "")))
    if mode == "plan" and is_mutating(name) and not plan_write:
        return "blocked"
    return "allowed"


def test_is_mutating_covers_every_mutator():
    """The gate's blocking arm keys off is_mutating; pin the set it screens so a tool that
    silently drops out of MUTATING (and thus past the plan-mode block) is caught here."""
    for name in ("bash", "write", "edit", "replace_symbol", "insert_symbol", "rename_symbol"):
        check(f"{name} is mutating", is_mutating(name) is True, name)
    for name in ("read", "grep", "glob", "repo_map", "find_symbol", "done"):
        check(f"{name} is NOT mutating", is_mutating(name) is False, name)


def test_plan_mode_blocks_mutating_tools():
    """In plan mode: every mutating tool is blocked, EXCEPT write/edit under ./plans/.
    A write outside ./plans/, an escaping `plans/../x`, bash, and symbol edits all block;
    a legitimate plan write/edit is allowed. (Outside plan mode the plan-block never
    fires — it is a plan-mode-only clamp.)"""
    # Blocked in plan mode:
    check("plan: bash blocked", _plan_gate("plan", "bash", {"command": "ls"}) == "blocked")
    check("plan: write outside ./plans/ blocked",
          _plan_gate("plan", "write", {"path": "src/chad/agent.py"}) == "blocked")
    check("plan: `..` escape write blocked",
          _plan_gate("plan", "write", {"path": "plans/../src/x.py"}) == "blocked")
    check("plan: symbol edit blocked",
          _plan_gate("plan", "replace_symbol", {"path": "plans/x.md"}) == "blocked")

    # Allowed in plan mode (the one escape hatch):
    check("plan: write under ./plans/ allowed",
          _plan_gate("plan", "write", {"path": "plans/050-x.md"}) == "allowed")
    check("plan: edit under ./plans/ allowed",
          _plan_gate("plan", "edit", {"path": "plans/050-x.md"}) == "allowed")

    # The clamp is plan-mode-only: auto/normal don't hit the plan-block at all.
    check("auto: bash not plan-blocked", _plan_gate("auto", "bash", {"command": "ls"}) == "allowed")
    check("normal: write not plan-blocked",
          _plan_gate("normal", "write", {"path": "src/x.py"}) == "allowed")


# === 3. The destructive-bash seatbelt in Agent._confirm =====================================

# A command guardrails.is_destructive_bash flags today (verified in that module's tests).
_RM_HOME = "rm -rf ~"
_SAFE = "ls -la"


def test_confirm_auto_safe_bash_runs():
    """yolo mode + a non-dangerous bash → runs without a prompt (returns True)."""
    agent = _mk_agent(mode="yolo")
    check("safe bash auto-approves", agent._confirm("bash", {"command": _SAFE}) is True)


def test_confirm_auto_destructive_headless_blocks(monkeypatch):
    """yolo mode + a destructive bash + no confirm channel (no callback, not a TTY) → the
    seatbelt BLOCKS (returns False) rather than executing on a possible injection, and says
    so on the transcript. This is the exact wiring a refactor dropping the
    `not sys.stdin.isatty()` block would silently re-open."""
    monkeypatch.delenv("CHAD_NO_DESTRUCTIVE_GUARD", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    seen = []
    agent = _mk_agent(mode="yolo", emit=lambda k, t: seen.append((k, t)))
    # _confirm_cb defaults to None (no callback), so the headless-block branch is reached.
    check("destructive bash is blocked headless",
          agent._confirm("bash", {"command": _RM_HOME}) is False)
    check("blocked notice emitted to transcript",
          any("blocked destructive" in str(t) for _, t in seen), seen)


def test_confirm_destructive_consults_callback(monkeypatch):
    """When a confirm channel EXISTS (a callback), a destructive command is not silently
    blocked — the channel is consulted and its answer honored (here: approve)."""
    monkeypatch.delenv("CHAD_NO_DESTRUCTIVE_GUARD", raising=False)
    calls = []

    def cb(name, args):
        calls.append((name, args))
        return True

    agent = _mk_agent(mode="yolo", confirm=cb)
    check("callback answer honored", agent._confirm("bash", {"command": _RM_HOME}) is True)
    check("callback was actually consulted", calls == [("bash", {"command": _RM_HOME})], calls)


def test_confirm_headless_block_sets_truthful_deny_reason(monkeypatch):
    """The headless guard block must hand the MODEL a truthful explanation via
    `_deny_reason` (the dispatch site substitutes it for '[denied by user]'): the bare
    denial reads as a human refusal, and the measured trace is a model re-phrasing the
    same delete 30 times against it. Lever OFF restores the bare text (reason unset)."""
    monkeypatch.delenv("CHAD_NO_DESTRUCTIVE_GUARD", raising=False)
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    agent = _mk_agent(mode="yolo")
    check("block still blocks", agent._confirm("bash", {"command": _RM_HOME}) is False)
    check("deny reason names the guard, not the user",
          agent._deny_reason is not None and "destructive-command guard" in agent._deny_reason,
          agent._deny_reason)
    monkeypatch.setenv("CHAD_DISABLE", "scoped_destructive_guard")
    agent2 = _mk_agent(mode="yolo")
    check("lever OFF: still blocks", agent2._confirm("bash", {"command": _RM_HOME}) is False)
    check("lever OFF: bare '[denied by user]' preserved (no reason set)",
          agent2._deny_reason is None, agent2._deny_reason)


def test_confirm_guard_opt_out(monkeypatch):
    """CHAD_NO_DESTRUCTIVE_GUARD=1 disables the seatbelt: the same catastrophic command in
    yolo mode with no channel now runs (returns True) instead of blocking."""
    monkeypatch.setenv("CHAD_NO_DESTRUCTIVE_GUARD", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    agent = _mk_agent(mode="yolo")
    check("opt-out lets the destructive command run",
          agent._confirm("bash", {"command": _RM_HOME}) is True)


# === 4. The accept-edits gate: 'auto' auto-approves edits but NOT the shell ==============

def test_auto_mode_approves_edits_without_prompting():
    """'auto' (auto-accept edits) waves through file edits: no confirm callback is
    consulted and _confirm returns True on its own."""
    calls = []
    agent = _mk_agent(mode="auto", confirm=lambda n, a: calls.append(n) or True)
    for name in ("write", "edit", "replace_lines", "insert_lines",
                 "replace_symbol", "insert_symbol", "rename_symbol"):
        check(f"auto approves {name} silently",
              agent._confirm(name, {"path": "src/x.py"}) is True)
    check("no confirm channel consulted for edits", calls == [], calls)


def test_auto_mode_still_confirms_bash():
    """The whole point of the mode: a terminal command still reaches the human, and a
    denial is honored. (A mutating MCP tool takes the same path — see
    test_auto_mode_fails_closed_on_unknown_mutating_tool for the policy call, which is
    where it can be pinned without standing up an MCP server.)"""
    seen = []

    def cb(name, args):
        seen.append(name)
        return False

    agent = _mk_agent(mode="auto", confirm=cb)
    check("bash is confirmed, not auto-approved",
          agent._confirm("bash", {"command": "ls -la"}) is False)
    check("the human channel was consulted", seen == ["bash"], seen)


def test_auto_mode_fails_closed_on_unknown_mutating_tool():
    """AUTO_EDIT_TOOLS is an allowlist, not `MUTATING - {bash}`: a mutating tool nobody
    has classified yet must ask rather than inherit auto-approval."""
    from chad.agent import AUTO_EDIT_TOOLS, auto_approves
    check("unknown mutating tool is not auto-approved",
          auto_approves("auto", "deploy_to_prod") is False)
    check("a mutating MCP tool is not auto-approved",
          auto_approves("auto", "mcp__srv__send_email") is False)
    check("bash is never in the edits allowlist", "bash" not in AUTO_EDIT_TOOLS)
    check("yolo approves everything", auto_approves("yolo", "deploy_to_prod") is True)
    check("normal approves nothing mutating", auto_approves("normal", "edit") is False)
    check("plan approves nothing mutating", auto_approves("plan", "edit") is False)


def test_mode_cycle_reaches_every_mode():
    """shift-tab must reach yolo, and cycling must return home — a mode you can enter but
    not leave (or can't reach) is a UX trap."""
    from chad.agent import MODE_LABEL, MODES
    agent = _mk_agent(mode="normal")
    seq = [agent.cycle_mode() for _ in range(len(MODES))]
    check("cycle visits every mode", set(seq) == set(MODES), seq)
    check("cycle returns to the start", seq[-1] == "normal", seq)
    check("every mode has a label", all(m in MODE_LABEL for m in MODES), MODE_LABEL)


if __name__ == "__main__":
    import inspect
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        if inspect.signature(fn).parameters:  # needs pytest fixtures (monkeypatch)
            print(f"  (skipping {fn.__name__} — needs pytest fixtures; run under pytest)")
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            FAIL += 1
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
