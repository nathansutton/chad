# Known gaps

Shapes whose `expect.json` says what SHOULD happen and whose `reason` says why it does not
yet. `tests/test_toolcall_corpus.py` asserts each gap is still open, so a parser change that
closes one has to move its pair out of this directory (and a change that opens one fails the
test). Each `reason` is either a reason the gap must stay or a pointer to the fix.
