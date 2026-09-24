"""Every coding agent the kit can run besides chad in process, as data, and the lock that
pins which version of each a result was measured with.

    python benchmarks/polyglot/harnesses.py              # each arm: installed vs locked
    python benchmarks/polyglot/harnesses.py lock pi      # pin what is installed now

An entry is the harness's own documented headless form — its argv, the environment
variables it reads, the config files it expects in its home — with two things made the
same for every arm: each auto-approves its own tools (a harness that stops to ask is not
measured on the same terms as one that does not), and each is pointed at the one server
the block runs against. Everything about isolation is `harness/cli.py`'s, not an entry's.

The entries are the nine arms of the matrix grid (`archive/matrix-nine-harnesses`,
`benchmarks/matrix/run.py`), with each config writer turned into the file it wrote. Three
more were installed there and dropped at smoke; they are recorded below as unsupported so
the reason travels with the name.

`harnesses.lock` (JSON, written by `lock`) holds, per arm, the version, the install command
for it, and the sha256 of the resolved entry point. `run.py` refuses an arm whose installed
version is not the locked one; a different sha256 at the same version is reported and
recorded, not refused, because an entry point installed by a Python tool embeds its own
interpreter's path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from string import Template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog  # noqa: E402
import server  # noqa: E402
from catalog import JsonValue, is_object, is_text  # noqa: E402
from harness.cli import CliHarness, CliSpec, Endpoint, HarnessError, Pin, entry_sha256  # noqa: E402

LOCK = os.path.join(catalog.ROOT, "harnesses.lock")
SERVED_MODEL = server.ALIAS            # the llama-server --alias every arm asks for

_OPENCODE = """{
  "provider": {
    "llama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server-local",
      "options": {"baseURL": "${origin}/v1", "apiKey": "${api_key}"},
      "models": {"${model}": {"name": "${model}", "limit": {"context": ${context}, "output": 8192}}}
    }
  }
}
"""

# `compat` tells pi-ai this is not OpenAI itself: no `developer` role, `max_tokens` not
# `max_completion_tokens`, which is what an OpenAI-shaped gateway needs.
_DSH = """- id: llm-pi-ai
  config:
    providers:
      llama:
        displayName: llama-server-local
        apiKeyEnv: LLAMA_API_KEY
        api: openai-completions
        baseURL: ${origin}/v1
        compat:
          supportsDeveloperRole: false
          maxTokensField: max_tokens
        models:
          - id: ${model}
            name: ${model}
            contextWindow: ${context}
            maxTokens: 8192
- id: agent-default-model
  config:
    provider: llama
    model: ${model}
"""

_GOOSE = """GOOSE_PROVIDER: openai
GOOSE_MODEL: ${model}
GOOSE_MODE: auto
OPENAI_HOST: ${origin}
OPENAI_BASE_PATH: v1/chat/completions
extensions:
  developer:
    enabled: true
    type: builtin
    name: developer
    display_name: Developer
    timeout: 300
    bundled: true
"""

# LiteLLM refuses to price a model it has never heard of; a zero-cost entry keeps mini's
# cost tracking from raising on a local one.
_MINI_REGISTRY = """{
  "openai/${model}": {"max_tokens": 8192, "max_input_tokens": ${context},
                      "max_output_tokens": 8192, "input_cost_per_token": 0.0,
                      "output_cost_per_token": 0.0, "litellm_provider": "openai",
                      "mode": "chat"}
}
"""

# Auto-approve lives in this file for crush (`permissions.skip_requests`).
_CRUSH = """{
  "$$schema": "https://charm.land/crush.json",
  "providers": {
    "llama": {"type": "openai-compat", "name": "llama-server-local",
              "base_url": "${origin}/v1", "api_key": "${api_key}",
              "models": [{"id": "${model}", "name": "${model}",
                          "context_window": ${context}, "default_max_tokens": 8192}]}
  },
  "models": {"large": {"model": "${model}", "provider": "llama"},
             "small": {"model": "${model}", "provider": "llama"}},
  "options": {"disable_provider_auto_update": true, "disable_metrics": true,
              "disable_default_providers": true},
  "permissions": {"skip_requests": true}
}
"""

_CODEX = """model = "${model}"
model_provider = "llama"

[model_providers.llama]
name = "llama-server-local"
base_url = "${origin}/v1"
wire_api = "responses"
env_key = "LLAMA_API_KEY"
requires_openai_auth = false
"""

SPECS: dict[str, CliSpec] = {s.name: s for s in (
    # chad's own command line on the same server as every foreign arm: the paired
    # counterpart to each of them. It drives llama-server's raw /completion with token
    # ids, so it needs the served model's tokenizer; the one it ships with shares the
    # GGUF's vocabulary.
    CliSpec(
        name="chad-llama", binary="chad", install="this checkout: uv sync", locked=False,
        argv=("${prompt}", "--yolo", "--backend", "llama", "--base-url", "${origin}",
              "--tokenizer", "${tokenizer}"),
        env={"CHAD_TRAJECTORY_JSON": "${home}/trajectory.json"},
        trajectory="trajectory.json"),
    CliSpec(
        name="pi", binary="pi", install="npm i -g @earendil-works/pi-coding-agent@${version}",
        argv=("-p", "${prompt}", "--provider", "llama", "--model", "${model}", "-a"),
        files={".pi/agent/models.json": """{
  "providers": {
    "llama": {"baseUrl": "${origin}/v1", "api": "openai-completions",
              "apiKey": "${api_key}", "models": [{"id": "${model}"}]}
  }
}
"""}),
    # opencode resolves its project from $PWD, not the cwd it is given, hence `--dir`.
    CliSpec(
        name="opencode", binary="opencode", install="npm i -g opencode-ai@${version}",
        argv=("run", "${prompt}", "--model", "llama/${model}", "--dir", "${workspace}",
              "--auto"),
        files={".config/opencode/opencode.json": _OPENCODE}),
    # deepseek-harness: `headless` is a profile (one task, print, exit); approval is an
    # environment variable, the provider the profile's patch layer.
    CliSpec(
        name="dsh", binary="dsh", install="npm i -g @deepseek-ai/dsh@${version}",
        argv=("--profile", "headless", "${prompt}"),
        env={"DSH_PERMISSION_MODE": "danger-full-access", "LLAMA_API_KEY": "${api_key}",
             "DSH_TELEMETRY_MODE": "DISABLED"},
        files={".dsh/profiles/headless/cordis.patch.yml": _DSH}),
    CliSpec(
        name="goose", binary="goose",
        install="curl -fsSL https://github.com/block/goose/releases/download/stable/"
                "download_cli.sh | GOOSE_VERSION=v${version} CONFIGURE=false bash",
        argv=("run", "-t", "${prompt}", "--no-session", "-q", "--provider", "openai",
              "--model", "${model}", "--with-builtin", "developer", "--max-turns", "200"),
        env={"GOOSE_PROVIDER": "openai", "GOOSE_MODEL": "${model}", "OPENAI_HOST": "${origin}",
             "OPENAI_BASE_PATH": "v1/chat/completions", "OPENAI_API_KEY": "${api_key}",
             "GOOSE_MODE": "auto", "GOOSE_DISABLE_KEYRING": "1"},
        files={".config/goose/config.yaml": _GOOSE}),
    # mini-swe-agent: bash only, no tool schemas at all. A `-c` REPLACES its default
    # config, so the bundled `mini.yaml` is named first (mini resolves the bare name to
    # its own copy) and overridden after. `ignore_errors` is its documented cost-tracking
    # value for a local model; MSWEA_CONFIGURED skips the first-run questions. It has no
    # --version: its own interpreter is asked instead.
    CliSpec(
        name="mini", binary="mini", install="uv tool install mini-swe-agent==${version}",
        argv=("-t", "${prompt}", "-y", "--exit-immediately", "-c", "mini.yaml",
              "-c", "agent.cost_limit=0", "-c", "model.model_name=openai/${model}",
              "-c", "model.model_kwargs.api_base=${origin}/v1",
              "-c", "model.model_kwargs.api_key=${api_key}"),
        env={"OPENAI_API_KEY": "${api_key}",
             "LITELLM_MODEL_REGISTRY_PATH": "${home}/litellm_registry.json",
             "MSWEA_CONFIGURED": "true", "MSWEA_MODEL_NAME": "openai/${model}",
             "MSWEA_COST_TRACKING": "ignore_errors", "MSWEA_SILENT_STARTUP": "true"},
        files={"litellm_registry.json": _MINI_REGISTRY},
        version=("${entry_dir}/python", "-c",
                 "import minisweagent; print(minisweagent.__version__)")),
    CliSpec(
        name="crush", binary="crush", install="npm i -g @charmland/crush@${version}",
        argv=("run", "-c", "${workspace}", "-q", "${prompt}"),
        files={".config/crush/crush.json": _CRUSH}),
    # cline keeps its provider in a store of its own; `cline auth` is the documented
    # way to write it without a prompt. `openai` is its id for OpenAI-compatible.
    CliSpec(
        name="cline", binary="cline", install="npm i -g cline@${version}",
        argv=("${prompt}", "-P", "openai", "-m", "${model}", "-c", "${workspace}",
              "--auto-approve", "true", "-t", "${inner_cap_s}"),
        setup=(("auth", "-p", "openai", "-k", "${api_key}", "-m", "${model}",
                "-b", "${origin}/v1"),)),
    # codex speaks only the Responses wire, which llama-server also serves. Its own
    # sandbox is bypassed: Seatbelt profiles do not nest, and this one is the kit's.
    CliSpec(
        name="codex", binary="codex", install="npm i -g @openai/codex@${version}",
        argv=("exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
              "-C", "${workspace}", "--json", "-c", "model_provider=llama",
              "-c", 'model="${model}"', "${prompt}"),
        env={"LLAMA_API_KEY": "${api_key}"},
        files={".codex/config.toml": _CODEX}),
    CliSpec(
        name="qwen", binary="qwen", install="npm i -g @qwen-code/qwen-code@${version}",
        unsupported="llama-server answers `400 failed to parse grammar`: its tool schemas, "
                    "converted to GBNF under --jinja, exceed the rule cap (qwen-code 0.22.3, "
                    "llama.cpp build 10470)"),
    CliSpec(
        name="deepagents", binary="dcode", install="uv tool install deepagents-code==${version}",
        unsupported="llama-server answers `400 failed to parse grammar`: its tool schemas, "
                    "converted to GBNF under --jinja, exceed the rule cap (deepagents-code "
                    "0.1.65, llama.cpp build 10470)"),
    CliSpec(
        name="aider", binary="aider", install="uv tool install aider-chat==${version}",
        unsupported="an edit-block chat loop with no tool calls: it never returns a server "
                    "`timings` object, so the per-request instrument cannot measure it "
                    "(aider 0.86.2)"),
)}


def load_lock(path: str = LOCK) -> dict[str, Pin]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        doc: JsonValue = json.load(f)
    if not is_object(doc):
        raise SystemExit(f"{path} is not a JSON object")
    pins = {}
    for name, entry in doc.items():
        if not is_object(entry):
            continue
        version, sha256, install = entry.get("version"), entry.get("sha256"), entry.get("install")
        if is_text(version) and is_text(sha256) and is_text(install):
            pins[name] = Pin(version, sha256, install)
    return pins


def write_lock(pins: dict[str, Pin], path: str = LOCK) -> None:
    doc = {name: {"version": p.version, "install": p.install, "sha256": p.sha256}
           for name, p in sorted(pins.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def _probe(spec: CliSpec, pin: Pin | None) -> CliHarness:
    """A harness good for `version()` alone: no server is contacted."""
    return CliHarness(spec, Endpoint("http://127.0.0.1:0", "unused", 0, ""), pin)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", choices=["list", "lock"], default="list")
    ap.add_argument("names", nargs="*", help="lock: the arms to pin (default: every installed one)")
    args = ap.parse_args()
    pins = load_lock()
    if args.command == "list":
        for spec in SPECS.values():
            pin = pins.get(spec.name)
            if spec.unsupported:
                state = f"unsupported: {spec.unsupported}"
            else:
                try:
                    installed = _probe(spec, pin).version()
                except HarnessError as e:
                    installed = f"not runnable ({e})"
                locked = "not locked" if not spec.locked else pin.version if pin else "no lock entry"
                state = f"installed {installed} | lock {locked}"
            print(f"{spec.name:12s} {state}")
        return 0
    unknown = set(args.names) - set(SPECS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")
    for name in args.names or [n for n, s in SPECS.items() if s.locked and not s.unsupported]:
        spec = SPECS[name]
        if spec.unsupported or not spec.locked:
            raise SystemExit(f"{name} is not lockable: "
                             f"{spec.unsupported or 'it is this checkout'}")
        probe = _probe(spec, pins.get(name))
        try:
            exe, version = probe.resolve(), probe.version()
        except HarnessError as e:
            print(f"{name:12s} skipped: {e}")
            continue
        pins[name] = Pin(version, entry_sha256(exe),
                         Template(spec.install).substitute(version=version))
        print(f"{name:12s} {version}")
    write_lock(pins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
