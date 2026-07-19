"""Single source of truth for directories never worth walking.

`IGNORE_DIRS` is the base set every tree-walk skips (VCS internals, virtualenvs,
build/cache output). `REPOMAP_EXTRA` are the additional dirs the repo-analysis tools
(repomap) skip — model weights, installed packages, caches — which a symbol *editor*
doesn't need to exclude. Kept here so a new ignore entry is added in exactly one place.
"""

IGNORE_DIRS = (".git", "node_modules", "__pycache__", ".venv", "venv",
               ".mypy_cache", ".pytest_cache", "dist", "build")

# repomap (whole-repo indexing) additionally skips these; a symbol edit doesn't.
REPOMAP_EXTRA = (".cache", "models", "site-packages")


def slash_wrapped(names) -> tuple:
    """`/name/` substring forms for path-contains tests (the symbols/repomap style)."""
    return tuple(f"/{d}/" for d in names)
