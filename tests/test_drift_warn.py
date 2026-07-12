"""Plan 074 — semantic-drift warning for parse-clean whole-unit rewrites.

The measured origin: a dogfood replace_symbol of bench.main dropped the
`--context-tokens` argparse line while `_run_agentic` still read
`args.context_tokens`; the file parsed, every gate stayed silent, and
`chad-bench --agentic` shipped an AttributeError. Each case here is that bug or a
masking/negative boundary of the two detection tiers. Pure + fast — no model load.
"""

from chad import symbols, tools
from chad.syntaxgate import drift_warn

# A faithful miniature of bench.py at the moment of the regression: the flag string,
# a same-named PARAMETER elsewhere (which masks Tier A's whole-file binding diff —
# the reason Tier B exists), and the args.<attr> consumer.
BENCH = '''\
import argparse

def _run_agentic(context_tokens):
    return context_tokens

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-tokens", type=int, default=24000)
    ap.add_argument("--agentic", action="store_true")
    args = ap.parse_args(argv)
    if args.agentic:
        return _run_agentic(args.context_tokens)
    return 0
'''


def _drop(text, needle):
    return "\n".join(ln for ln in text.split("\n") if needle not in ln)


def test_dropped_flag_with_live_consumer_warns(tmp_path):
    """The regression: flag definition gone, args.context_tokens still read."""
    after = _drop(BENCH, "--context-tokens")
    w = drift_warn("bench.py", BENCH, after)
    assert w and "--context-tokens" in w and "context_tokens" in w
    assert "line 11" in w, w   # args.context_tokens in the after-file


def test_full_flag_removal_stays_quiet():
    """Flag AND its args.<attr> consumer both removed: a legitimate feature removal.
    The surviving bare-name uses (the _run_agentic parameter) must not keep it warm —
    Tier B checks attribute reads only."""
    after = _drop(_drop(BENCH, "--context-tokens"), "args.context_tokens")
    assert drift_warn("bench.py", BENCH, after) is None


def test_dropped_function_still_called_warns():
    before = "def helper():\n    return 1\n\ndef main():\n    return helper()\n"
    after = "def main():\n    return helper()\n"
    w = drift_warn("m.py", before, after)
    assert w and "'helper'" in w and "line 2" in w


def test_dropped_constant_still_read_warns():
    before = "LIMIT = 10\n\ndef f(n):\n    return n < LIMIT\n"
    after = "def f(n):\n    return n < LIMIT\n"
    w = drift_warn("m.py", before, after)
    assert w and "'LIMIT'" in w


def test_removal_with_all_uses_stays_quiet():
    before = "LIMIT = 10\n\ndef f(n):\n    return n < LIMIT\n"
    after = "def f(n):\n    return n < 10\n"
    assert drift_warn("m.py", before, after) is None


def test_duplicate_binding_masks_local_drop():
    """A local dropped in one function is masked by another function's binding of the
    same name — the whole-file set diff is the false-positive control."""
    before = ("def a():\n    result = 1\n    return result\n\n"
              "def b():\n    result = 2\n    return result\n")
    after = ("def a():\n    return 1\n\n"
             "def b():\n    result = 2\n    return result\n")
    assert drift_warn("m.py", before, after) is None


def test_rename_fully_applied_stays_quiet():
    before = "def old_name():\n    return 1\n\nx = old_name()\n"
    after = "def new_name():\n    return 1\n\nx = new_name()\n"
    assert drift_warn("m.py", before, after) is None


def test_unknown_lang_and_broken_files_stay_quiet():
    assert drift_warn("notes.md", "a\n", "b\n") is None
    assert drift_warn("m.py", "def f(:\n", "x = 1\n") is None   # before doesn't parse
    assert drift_warn("m.py", None, "x = 1\n") is None          # new file


def test_javascript_dropped_function_warns():
    """The general (non-Python) tier rides the repo map's tree-sitter tags queries —
    same two-tier doctrine as check_syntax: Python exact, everything else best-effort."""
    before = ("function helper() { return 1; }\n"
              "function main() { return helper(); }\n")
    after = "function main() { return helper(); }\n"
    w = drift_warn("app.js", before, after)
    assert w and "'helper'" in w and "line 1" in w, w
    # Removal including the call site: quiet.
    assert drift_warn("app.js", before, "function main() { return 2; }\n") is None


def test_fires_through_the_real_tools(tmp_path):
    """End to end through replace_symbol (the tool that regressed) and replace_lines."""
    p = str(tmp_path / "bench.py")
    with open(p, "w") as f:
        f.write(BENCH)
    svc = symbols.SymbolService(str(tmp_path))
    new_main = ('def main(argv=None):\n'
                '    ap = argparse.ArgumentParser()\n'
                '    ap.add_argument("--agentic", action="store_true")\n'
                '    args = ap.parse_args(argv)\n'
                '    if args.agentic:\n'
                '        return _run_agentic(args.context_tokens)\n'
                '    return 0')
    res = svc.replace_symbol("main", new_main, path=p)
    assert res.startswith("[replaced") and "DROPPED" in res and "--context-tokens" in res

    p2 = str(tmp_path / "m.py")
    with open(p2, "w") as f:
        f.write("LIMIT = 10\n\n\ndef f(n):\n    return n < LIMIT\n")
    res = tools.tool_replace_lines(p2, 1, 1, "START = 0")
    assert res.startswith("[edited") and "DROPPED" in res and "'LIMIT'" in res
