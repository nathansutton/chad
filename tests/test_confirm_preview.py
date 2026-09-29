"""Battery for the pre-approval preview builder (render.confirm_preview).

The y/n confirmation is the human's safeguard against model mistakes and
prompt-injected edits. Before this, symbolic edits rendered a blank preview and
text edits showed only a path. These assertions lock in that the preview now
shows the meaningful change, and that it stays bounded (a huge write must not
flood the prompt).

Run: `uv run python tests/test_confirm_preview.py`
"""

import os
import tempfile

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


def test_preview_names_where_the_write_really_lands():
    """A symlink (or a `..`) sends the write somewhere the path string does not show,
    and that is precisely the case the human is being asked to approve."""
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.realpath(tmp)   # macOS: /var is itself a link to /private/var
        real = os.path.join(d, "real.txt")
        link = os.path.join(d, "link.txt")
        os.symlink(real, link)
        p = confirm_preview("write", {"path": link, "content": "x"})
        check("symlink preview shows the given path", link in p, f"p={p!r}")
        check("symlink preview shows the real target", real in p, f"p={p!r}")
        p = confirm_preview("edit", {"path": link, "old": "A", "new": "B"})
        check("edit preview resolves too", real in p, f"p={p!r}")
        # An ordinary path is shown once, with no arrow noise.
        plain = os.path.join(d, "plain.txt")
        p = confirm_preview("write", {"path": plain, "content": "x"})
        check("plain path is not decorated", "→" not in p, f"p={p!r}")


def test_full_preview_shows_what_the_default_clips():
    cmd = "echo " + "z" * 2995
    p = confirm_preview("bash", {"command": cmd})
    check("default bash preview still clips", p.endswith(" …") and len(p) == 402,
          f"len={len(p)}")
    check("full bash preview is the whole command",
          confirm_preview("bash", {"command": cmd}, full=True) == cmd)
    lines = [f"line {i}" for i in range(40)]
    p = confirm_preview("write", {"path": "x", "content": "\n".join(lines)}, full=True)
    check("full write shows every line", all(ln in p for ln in lines), f"p={p!r}")
    check("full write has no elision", "more lines" not in p, f"p={p!r}")

if __name__ == "__main__":
    test_preview()
    test_preview_names_where_the_write_really_lands()
    test_full_preview_shows_what_the_default_clips()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
