"""Battery for the pre-approval preview builder (render.confirm_preview).

The y/n confirmation is the human's safeguard against model mistakes and
prompt-injected edits. Before this, symbolic edits rendered a blank preview and
text edits showed only a path. These assertions lock in that the preview now
shows the meaningful change, and that it stays bounded (a huge write must not
flood the prompt).

Run: `uv run python tests/test_confirm_preview.py`
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
