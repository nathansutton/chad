"""Battery for the pre-approval preview builder (render.confirm_preview).

The y/n confirmation is the human's safeguard against model mistakes and
prompt-injected edits. Before this, symbolic edits rendered a blank preview and
text edits showed only a path. These assertions lock in that the preview now
shows the meaningful change, and that it stays bounded (a huge write must not
flood the prompt).

Run: `uv run python test_confirm_preview.py`
"""

from chad.render import confirm_preview

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def test_preview():
    # symbolic edit: name + body are visible (old blank-preview bug is gone)
    p = confirm_preview("replace_symbol", {"name": "Foo/bar", "new": "def bar(): ..."})
    check("replace_symbol shows name", "Foo/bar" in p, f"p={p!r}")
    check("replace_symbol shows body", "def bar" in p, f"p={p!r}")

    # rename: arrow form
    p = confirm_preview("rename_symbol", {"name": "a", "new_name": "b"})
    check("rename_symbol arrow", p == "a → b", f"p={p!r}")

    # text edit: path + diff lines
    p = confirm_preview("edit", {"path": "x.py", "old": "A", "new": "B"})
    check("edit shows path", "x.py" in p, f"p={p!r}")
    check("edit shows old line", "- A" in p, f"p={p!r}")
    check("edit shows new line", "+ B" in p, f"p={p!r}")

    # pathologically long content is clipped -> bounded
    p = confirm_preview("write", {"path": "x", "content": "y" * 100000})
    check("write bounded", len(p) < 5000, f"len={len(p)}")


if __name__ == "__main__":
    test_preview()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
