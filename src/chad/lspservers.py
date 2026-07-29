"""The language-server registry: which server speaks for which language, as DATA.

Prior art shapes this file: nvim-lspconfig and helix's languages.toml prove that
once a generic client exists (lspclient.py), each language server is a ~15-line
declaration — command candidates, an LSP languageId, and quirks. solidlsp reached
the same conclusion the heavy way: 50 of its 69 adapters were launch-only code.

The availability contract, per row, honest and explicit:
  * "path"  — the server must already be on PATH (rust-analyzer, gopls, clangd:
              toolchain-managed binaries chad must not install),
  * "uvx"   — fetchable via uv's tool runner. `uvx chad-code` PROVES uv is
              installed, so python precision is guaranteed on the primary install
              path (network permitting) with zero configuration,
  * "npx"   — fetchable via npm's runner when node is present.
Candidates are tried in order; PATH binaries win over fetchers (no network, no
version surprise). Languages that need a toolchain bootstrap chad refuses to own
(jdtls: java + a gradle download; omnisharp: a .NET install) get `commands=()` —
a row that SAYS it's unsupported beats a half-support that hangs.

`decorative_ok` marks servers cheap enough to *start* for a decorative "used by"
annotation (measured: pyright ~0.9s flat; clangd chewed one file ~10s and a mixed
repo's disambiguation for 86s). Everyone else only decorates when already running.

`ready_log` is the readiness signal: pyright answers references with a confident
EMPTY list until initial analysis finishes, and the only reliable tell is its
`window/logMessage` "Found N source files" (solidlsp's comment: "pyright is
unreliable and there seems to be no better way"). Rows without a rule are usable
right after `initialize` returns.
"""

import os
import shutil
from typing import NamedTuple, Optional

# One exact pyright: a fetched server must not drift under chad between sessions.
_PYRIGHT_PIN = "1.1.403"


class ServerSpec(NamedTuple):
    language: str            # chad's language id (lowercase; ours, not a vendor enum)
    language_id: str         # LSP languageId sent in didOpen
    server_key: str          # servers that speak several languages share one process
    commands: tuple          # candidate argv tuples, first resolvable wins
    tier: str                # "path" | "uvx" | "npx"  (of the *preferred* fetch row)
    decorative_ok: bool = False   # cheap enough to SPAWN for a "used by …" hint
    ready_log: Optional[str] = None   # logMessage regex that marks real readiness
    ready_timeout_s: float = 60.0
    # Project files, one of which must exist under the root before the server
    # starts at all. This is the honesty gate: clangd without a compile database
    # and TS without a tsconfig both "work" while returning confidently EMPTY
    # cross-file results (both measured) — worse than the labeled name-match
    # fallback. Empty tuple = no gate.
    project_gate: tuple = ()
    unsupported: str = ""    # human reason when commands is empty
    settings: Optional[dict] = None   # pushed via workspace/didChangeConfiguration
                                      # right after initialize; pyright never starts
                                      # analyzing without one (measured: refs hang)


_PYRIGHT = ServerSpec(
    language="python", language_id="python", server_key="pyright",
    commands=(("pyright-langserver", "--stdio"),
              ("uvx", "--from", f"pyright=={_PYRIGHT_PIN}",
               "pyright-langserver", "--stdio")),
    tier="uvx", decorative_ok=True,
    ready_log=r"Found \d+ source files?",
    settings={"python": {"analysis": {"autoSearchPaths": True,
                                      "useLibraryCodeForTypes": True,
                                      "diagnosticMode": "openFilesOnly"}}},
)

# TypeScript 7 (the native compiler) ships its own LSP inside the `typescript`
# package (`tsc --lsp --stdio`) — no typescript-language-server wrapper, no
# workspace-typescript resolution dance (measured: the wrapper refuses to start
# unless the WORKSPACE has a compatible typescript install; the native LSP just
# runs). PATH wrapper first for classic setups that have it; native via npx else.
_TYPESCRIPT = ServerSpec(
    language="typescript", language_id="typescript", server_key="typescript",
    commands=(("typescript-language-server", "--stdio"),
              ("npx", "-y", "-p", "typescript", "tsc", "--lsp", "--stdio")),
    tier="npx",
    project_gate=("tsconfig.json", "jsconfig.json"),
)

_CPP = ServerSpec(
    language="cpp", language_id="cpp", server_key="clangd",
    commands=(("clangd", "--background-index"),),
    tier="path",
    project_gate=("compile_commands.json",
                  os.path.join("build", "compile_commands.json"), ".clangd"),
)

SERVERS = (
    _PYRIGHT,
    _TYPESCRIPT,
    _TYPESCRIPT._replace(language="javascript", language_id="javascript"),
    ServerSpec(language="go", language_id="go", server_key="gopls",
               commands=(("gopls", "serve"),), tier="path"),
    ServerSpec(language="rust", language_id="rust", server_key="rust-analyzer",
               commands=(("rust-analyzer",),), tier="path"),
    _CPP._replace(language="c", language_id="c"),
    _CPP,
    ServerSpec(language="ruby", language_id="ruby", server_key="solargraph",
               commands=(("solargraph", "stdio"),), tier="path"),
    ServerSpec(language="php", language_id="php", server_key="intelephense",
               commands=(("intelephense", "--stdio"),
                         ("npx", "-y", "intelephense", "--stdio")),
               tier="npx"),
    ServerSpec(language="kotlin", language_id="kotlin", server_key="kotlin",
               commands=(("kotlin-language-server",),), tier="path"),
    ServerSpec(language="java", language_id="java", server_key="jdtls",
               commands=(), tier="path",
               unsupported="jdtls needs a JDK plus a workspace bootstrap chad "
                           "does not own; use find_refs (name-match) for java"),
    ServerSpec(language="csharp", language_id="csharp", server_key="omnisharp",
               commands=(), tier="path",
               unsupported="omnisharp/roslyn needs a .NET install chad does not "
                           "own; use find_refs (name-match) for c#"),
)

_BY_LANGUAGE = {s.language: s for s in SERVERS}

# Extension -> chad language id. Chad's OWN identity: .js is javascript (served by
# the typescript server, but named for what it is) and .c/.h are c, not "CPP" —
# both were vendor-enum spellings inherited from solidlsp.
LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
    ".cs": "csharp", ".php": "php", ".kt": "kotlin",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
}


def lang_for(path: str) -> Optional[str]:
    return LANG_BY_EXT.get(os.path.splitext(path)[1].lower())


def spec_for(lang: str) -> Optional[ServerSpec]:
    return _BY_LANGUAGE.get(lang)


def resolve_commands(spec: ServerSpec):
    """Every candidate whose launcher exists on PATH, best first. A `uvx`/`npx`
    row resolves when the runner exists — the runner then fetches the server.
    Callers try them in order: a candidate that resolves but fails to START
    (e.g. typescript-language-server with no workspace typescript) must not
    block the next one."""
    return [list(cand) for cand in spec.commands if shutil.which(cand[0])]


def resolve_command(spec: ServerSpec):
    """The single best candidate, or None (see resolve_commands)."""
    cands = resolve_commands(spec)
    return cands[0] if cands else None
