"""Regression anchor for the consolidated skip-directory lists (plan 032).

The base list lives in one place now (`chad.ignore`); these expected literals are
the pre-refactor values, pasted here so the test proves the consolidation changed
no module's *effective* skip set. repomap keeps its 3 extra dirs; every other module
keeps exactly the base 9.
"""

from chad import repomap, skills, symbols, tools

EXPECTED_BARE = (".git", "node_modules", "__pycache__", ".venv", "venv",
                 ".mypy_cache", ".pytest_cache", "dist", "build")
EXPECTED_REPOMAP_NAMES = frozenset(EXPECTED_BARE + (".cache", "models", "site-packages"))


def test_ignore_sets_unchanged():
    assert tools.IGNORE_DIRS == EXPECTED_BARE
    assert tools._SKIP_DIRS == tuple(f"/{d}/" for d in EXPECTED_BARE)
    assert skills._SKIP_DIRS == set(EXPECTED_BARE)
    # symbols prunes bare names during its os.walk now (no slash-wrapped post-filter)
    assert symbols.IGNORE_DIRS == EXPECTED_BARE
    assert repomap._SKIP_NAMES == EXPECTED_REPOMAP_NAMES


if __name__ == "__main__":
    test_ignore_sets_unchanged()
    print("ok")
