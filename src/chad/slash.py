"""Slash commands that do not exist.

Both front ends match builtins exactly and hand everything else to the model, so a
typo (`/hepl`), a stray argument (`/undo 2`) or a command the other front end has
(`/undo` in the plain REPL) used to become a task: a full turn on a local model, and in
yolo mode one that may act on it. This module is the check that runs first.
"""

import difflib
import re
from typing import Iterable, Optional

# A first token shaped like a command: a slash, a letter, then letters, digits, dashes
# or underscores, and nothing else. `/usr/local/bin is broken` is a path, not a command.
_COMMAND_RE = re.compile(r"^/[A-Za-z][A-Za-z0-9_-]*$")


def command_token(text: str) -> Optional[str]:
    """The command-shaped first word of `text`, or None when the text is not one."""
    words = text.split(None, 1)
    if not words or not _COMMAND_RE.match(words[0]):
        return None
    return words[0]


def unknown_message(text: str, known: Iterable[str],
                    elsewhere: Iterable[str] = ()) -> Optional[str]:
    """What to tell the user when `text` starts with a command that is not in `known`.

    `known` is every command and skill this front end runs, by its first word.
    `elsewhere` is the commands only the other front end has. None means `text` is not
    an unknown command, and the caller carries on."""
    token = command_token(text)
    if token is None:
        return None
    names = sorted(set(known))
    if token in names:
        # A real command with something after it that the exact match rejected.
        return (f"{token} takes no argument here — type {token} on its own "
                f"(/help lists commands)")
    if token in set(elsewhere):
        return f"{token} is only available in the full terminal UI (run chad without --repl)"
    close = difflib.get_close_matches(token, names, n=1, cutoff=0.6)
    hint = f" — did you mean {close[0]}?" if close else ""
    return (f"unknown command {token}{hint} (/help lists commands). "
            f"To send this text to the model, start it with a word.")
