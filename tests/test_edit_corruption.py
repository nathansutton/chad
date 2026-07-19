"""Line-addressed edits must not corrupt multi-line structures.

Each case reproduces a measured failure from the 073 dogfood session (9B/35B on the
CHAD_PREFILL_CHUNK task): stale line numbers reused after the file shifted, a fragment
replacing one physical line of a multi-line def signature, ten ignored "no longer
parses" warnings compounding into a broken tree, and the loop guard escalating only
after the damage. Pure + fast — no model load.
"""

from chad import guardrails, symbols, syntaxgate, tools

# A miniature engine.py: two adjacent defs, the second with a multi-line signature —
# the exact shape whose severing produced the orphaned-parameter corruption.
ENGINE = (
    "class Engine:\n"                        # 1
    "    def _prefill(self, ids):\n"         # 2
    "        n = len(ids)\n"                 # 3
    "        return n\n"                     # 4
    "\n"                                     # 5
    "    def generate(\n"                    # 6
    "        self,\n"                        # 7
    "        prompt_ids,\n"                  # 8
    "        max_tokens,\n"                  # 9
    "    ):\n"                               # 10
    "        return prompt_ids\n"            # 11
)


def _seed(tmp_path, content=ENGINE, name="engine.py"):
    p = str(tmp_path / name)
    with open(p, "w") as f:
        f.write(content)
    return p


def test_severed_signature_rejected_with_span_and_echo(tmp_path):
    """The literal killer call from the session log: replace_lines on ONE line of the
    multi-line `def generate(` signature. Must be rejected; the message must name the
    full statement span and echo current numbered lines so the retry is anchored."""
    p = _seed(tmp_path)
    res = tools.tool_replace_lines(p, 7, 7, "def generate(")
    assert res.startswith("[edit rejected") and "unparseable" in res
    assert "lines 6-10" in res, f"must name the severed signature's span: {res}"
    assert "Current lines" in res and "8  " in res, f"must echo numbered lines: {res}"
    assert "replace_symbol('Engine/generate')" in res, f"must steer to the symbol: {res}"
    assert open(p).read() == ENGINE, "the file must be left byte-identical"


def test_insert_into_signature_rejected(tmp_path):
    """insert_lines into the middle of a multi-line signature severs it just the same."""
    p = _seed(tmp_path)
    res = tools.tool_insert_lines(p, 8, "x = 1")
    assert res.startswith("[edit rejected") and "unparseable" in res
    assert open(p).read() == ENGINE


def test_whole_statement_replace_lands_and_echoes_new_numbers(tmp_path):
    """The sanctioned path: replace the ENTIRE signature range in one call. It lands,
    and the result echoes the region with post-edit numbers plus the shift note."""
    p = _seed(tmp_path)
    res = tools.tool_replace_lines(
        p, 6, 10, "def generate(\n    self,\n    prompt_ids,\n):")
    assert res.startswith("[edited")
    assert "use THESE numbers" in res
    assert "shifted by -1" in res, f"10->9 lines must be called out: {res}"
    assert "return prompt_ids" in open(p).read()


def test_delete_range_still_works(tmp_path):
    p = _seed(tmp_path, "a = 1\nb = 2\nc = 3\n", "m.py")
    res = tools.tool_replace_lines(p, 2, 2, "")
    assert res.startswith("[edited") and "deleted lines 2-2" in res
    assert open(p).read() == "a = 1\nc = 3\n"


def test_already_broken_file_stays_editable(tmp_path):
    """The carve-out: when the file is ALREADY unparseable, line edits still land (a
    real fix passes through still-broken states) — only clean files are protected."""
    broken = "def f(:\n    return 1\n"
    p = _seed(tmp_path, broken, "b.py")
    res = tools.tool_replace_lines(p, 2, 2, "    return 2")
    assert res.startswith("[edited"), res
    assert "return 2" in open(p).read()


def test_stale_numbers_after_own_edit_are_covered_by_echo(tmp_path):
    """The session's stale-number sequence: an edit changes the line count, and the
    next call reuses pre-edit numbers. The first result's echo carries the shift note;
    a further edit with echoed numbers lands cleanly (no lockout, no corruption)."""
    p = _seed(tmp_path)
    res = tools.tool_replace_lines(p, 3, 4, "        return len(ids)")
    assert "shifted by -1" in res
    # Follow-up using the ECHOED numbering (generate now starts at line 5).
    res2 = tools.tool_replace_lines(
        p, 5, 9, "    def generate(self, prompt_ids):")
    assert res2.startswith("[edited"), res2
    import ast
    ast.parse(open(p).read())


def test_out_of_band_change_rejects_once_then_allows(tmp_path):
    """bash/sed/git moving the file under the model: one reject with a fresh numbered
    view, and the reject itself refreshes the anchor so the retry goes through."""
    p = _seed(tmp_path)
    tools.tool_read(p)
    with open(p, "w") as f:                       # out-of-band: two lines prepended
        f.write("import os\nimport re\n" + ENGINE)
    res = tools.tool_replace_lines(p, 3, 3, "        n = len(ids)")
    assert res.startswith("[edit rejected") and "changed on disk" in res
    assert "Current lines" in res
    res2 = tools.tool_replace_lines(p, 5, 5, "        n = len(list(ids))")
    assert res2.startswith("[edited"), res2


def test_replace_symbol_with_bad_code_rejected(tmp_path):
    """A replace_symbol whose new definition doesn't parse is reverted, and the message
    blames the sent code (placement is the harness's job and correct by construction)."""
    p = _seed(tmp_path)
    svc = symbols.SymbolService(str(tmp_path))
    res = svc.replace_symbol("Engine/_prefill",
                             "def _prefill(self, ids, chunk=256:\n        return 1",
                             path=p)
    assert res.startswith("[edit rejected"), res
    assert "code you sent" in res
    assert open(p).read() == ENGINE


def test_guardrails_classify_structure_and_stale():
    """The reject shapes feed the edit-failure streak with their own remedies."""
    structure = "[edit rejected: it would leave engine.py unparseable — invalid syntax]"
    stale = "[edit rejected: engine.py changed on disk since you last read it]"
    indent = "[edit rejected: it would break engine.py — unexpected indent]"
    assert guardrails.edit_fail_kind(structure) == "structure"
    assert guardrails.edit_fail_kind(stale) == "stale"
    assert guardrails.edit_fail_kind(indent) == "indent"
    for r in (structure, stale, indent):
        assert guardrails.edit_failed_to_land(r)
    brk = guardrails.edit_loop_break(2, 0, "structure")
    assert brk and "replace_symbol" in brk and "multi-line" in brk


def test_severed_span_diagnosis():
    """_severed_span picks the innermost cut structure: a compound statement counts
    only through its header (the def line through the colon), not its whole body."""
    import ast
    tree = ast.parse(ENGINE)
    # Cutting one line of the generate signature names the header 6-10.
    assert syntaxgate._severed_span(tree, 7, 7) == (6, 10)
    # A range fully covering the signature severs nothing at that level.
    assert syntaxgate._severed_span(tree, 6, 10) is None
    # An insertion boundary inside the signature (between 8 and 9).
    assert syntaxgate._severed_span(tree, 9, 8) == (6, 10)
    # Editing whole lines inside a def BODY doesn't flag the def itself.
    assert syntaxgate._severed_span(tree, 3, 4) is None
