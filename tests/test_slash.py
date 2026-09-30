"""A slash command that does not exist is reported, not sent to the model."""

from chad.slash import command_token, unknown_message


def test_plain_text_is_not_a_command():
    assert unknown_message("fix the bug", ["/help"]) is None


def test_a_path_is_not_a_command():
    assert unknown_message("/usr/local/bin is broken", ["/help"]) is None


def test_empty_and_bare_slash_are_not_commands():
    assert unknown_message("", ["/help"]) is None
    assert unknown_message("/", ["/help"]) is None


def test_a_typo_suggests_the_closest_command():
    msg = unknown_message("/hepl", ["/help", "/undo"])
    assert "unknown command /hepl" in msg
    assert "did you mean /help?" in msg


def test_nothing_close_gets_no_suggestion():
    msg = unknown_message("/zzzz", ["/help"])
    assert "unknown command /zzzz" in msg
    assert "did you mean" not in msg


def test_a_stray_argument_on_a_real_command():
    assert "/compact takes no argument here" in unknown_message("/compact now", ["/compact"])


def test_a_command_only_the_other_front_end_has():
    msg = unknown_message("/undo", ["/help"], ["/undo"])
    assert "only available in the full terminal UI" in msg


def test_command_token_is_the_first_word():
    assert command_token("/mcp login linear") == "/mcp"
