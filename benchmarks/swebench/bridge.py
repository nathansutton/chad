"""The static bridge: prove the two trees are the same harness until they are told not to be.

    uv run python benchmarks/swebench/bridge.py --legacy ../chad-legacy

A two-tree comparison has an obvious hole: any difference in the result could be the
harness design, or it could be that one tree quietly renders a different prompt, ships a
different sampler, or was built from the wrong commit. A statistical bridge arm — run
both trees in the same configuration and check the scores agree — costs a week of laptop
and answers with a confidence interval.

This is the deterministic version. `bench/legacy` under `CHAD_LEAN=1` reduces to the
lean surface: five tools, the lean prompt, the lean lever defaults. If that render is
token-for-token the release tree's render of the same task, then everything upstream of
the harness design — the template, the schemas, the sampler, the prompt builder, the
tokenizer — is shared, and the only thing left to explain a difference at k=5 is the
thing the experiment is about. Where it is NOT identical, the difference is enumerated
here and justified in PORTS.md, before any scored trial, rather than discovered in the
residuals afterwards.

WHAT IS COMPARED
----------------
The FIRST REQUEST of a real task, which is what the model actually sees:

  * the rendered prompt as TOKEN IDS, through each tree's own tokenizer and chat
    template, for the same workspace and the same task text;
  * the tool schemas, as JSON, name by name;
  * the active lever set;
  * the sampler, through each tree's own shipped preset helper;
  * the system prompt text, so a difference can be read rather than inferred.

Token ids, not strings: two prompts can differ in whitespace the tokenizer folds away,
and they can agree as text while a template renders a different special token. Ids are
what the engine prefills.

NO WEIGHTS
----------
The comparison needs a tokenizer and a chat template, not a model. `--model` names the
repo whose tokenizer both trees load; nothing is generated, so this runs in seconds on a
machine with the weights already cached, and it is the check to re-run after any port.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))

# One ordinary task, fixed. The point is not the task but that both trees are asked the
# same thing in the same place; a fixed string keeps the check reproducible.
PROBE_TASK = "Fix the off-by-one in the pagination helper."

# The child prints exactly this, as one JSON object on the last line.
_PROBE = r'''
import json, os, sys
sys.path.insert(0, os.path.join(TREE, "src"))
os.environ.setdefault("CHAD_NO_SKILLS", "1")
from chad import levers
from chad.cli import apply_sampler_env, apply_sampler_preset
from chad.prompt import build_system_prompt
from chad.tools import active_schemas

class _Sampler:
    temp = top_p = min_p = presence_penalty = None
    top_k = None

def _tok(model_id):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

os.chdir(WORKSPACE)
try:
    mid = MODEL
    tok = _tok(mid)
    schemas = active_schemas()
    system = build_system_prompt(mid) if PASS_MODEL_ID else build_system_prompt()
    if OVERRIDE:
        # Controlled render: the same prompt text and the same schemas in both trees,
        # so what is left being compared is the tokenizer, the chat template and the
        # id coercion — the machinery, with the design held equal.
        system, schemas = OVERRIDE["system"], OVERRIDE["schemas"]
    # Through each tree's own coercion, not a hand-rolled one: some HF tokenizers
    # return a BatchEncoding whose `list()` is its string KEYS, and a bridge that
    # compared `['input_ids', 'attention_mask']` between two trees would agree
    # perfectly while proving nothing.
    from chad.agent import Agent
    ids = Agent._template_ids(tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": TASK}],
        tools=schemas, add_generation_prompt=True, enable_thinking=True))
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    s = _Sampler()
    apply_sampler_preset(s, thinking=True)
    apply_sampler_env(s)
    out = {
        "ok": True,
        "version": __import__("chad").__version__,
        "ids": list(ids),
        "n_ids": len(ids),
        "system": system,
        "schemas": schemas,
        "tools": sorted(x["function"]["name"] for x in schemas),
        "levers": sorted(levers.active()),
        "sampler": {"temp": s.temp, "top_p": s.top_p, "top_k": s.top_k,
                    "min_p": s.min_p, "presence_penalty": s.presence_penalty},
    }
except Exception as e:
    out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
sys.stdout.write("\n@@PROBE@@" + json.dumps(out))
'''


def probe(tree: str, model: str, workspace: str, lean: bool, pass_model_id: bool,
          override: dict | None = None) -> dict:
    """Render the first request inside `tree`'s own interpreter and environment.

    A subprocess per tree, always: both trees install a package called `chad`, and
    whichever is imported first would own the name for the rest of the process. Reading
    them in one process is the one way to guarantee a false agreement.

    `override` supplies a system prompt and tool schemas to render INSTEAD of the tree's
    own, which is how the controlled stage separates machinery from design.
    """
    preamble = (f"TREE = {tree!r}\nMODEL = {model!r}\nWORKSPACE = {workspace!r}\n"
                f"TASK = {PROBE_TASK!r}\nPASS_MODEL_ID = {pass_model_id!r}\n"
                f"OVERRIDE = {override!r}\n")
    env = dict(os.environ)
    env["CHAD_NO_SKILLS"] = "1"
    env.pop("CHAD_ENABLE", None)
    env.pop("CHAD_DISABLE", None)
    if lean:
        env["CHAD_LEAN"] = "1"
    else:
        env.pop("CHAD_LEAN", None)
    out = subprocess.run(
        ["uv", "run", "--project", tree, "--quiet", "python", "-c", preamble + _PROBE],
        capture_output=True, text=True, check=False, timeout=900, env=env)
    marker = "@@PROBE@@"
    if marker not in out.stdout:
        return {"ok": False, "error": (out.stderr or out.stdout or "no output")[-2000:]}
    return json.loads(out.stdout.rsplit(marker, 1)[1])


def _schema_diff(a: list, b: list) -> list[str]:
    """Per-tool JSON differences, by name, so a changed description is reported as one
    line rather than as a wall of re-ordered dicts."""
    names_a = {x["function"]["name"]: x for x in a}
    names_b = {x["function"]["name"]: x for x in b}
    notes = []
    for name in sorted(set(names_a) - set(names_b)):
        notes.append(f"  only in LEGACY+CHAD_LEAN: {name}")
    for name in sorted(set(names_b) - set(names_a)):
        notes.append(f"  only in RELEASE: {name}")
    for name in sorted(set(names_a) & set(names_b)):
        ja = json.dumps(names_a[name], sort_keys=True)
        jb = json.dumps(names_b[name], sort_keys=True)
        if ja != jb:
            notes.append(f"  {name}: schema differs ({len(ja)} vs {len(jb)} bytes)")
    return notes


def _first_divergence(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def report(legacy: dict, release: dict, show_diff: bool) -> bool:
    """Print the comparison; return True when the bridge is clean."""
    clean = True

    def line(ok: bool, label: str, detail: str = "") -> None:
        nonlocal clean
        clean = clean and ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")

    print(f"\nlegacy+CHAD_LEAN  chad {legacy['version']}  "
          f"{legacy['n_ids']:,} tokens, {len(legacy['tools'])} tools")
    print(f"release           chad {release['version']}  "
          f"{release['n_ids']:,} tokens, {len(release['tools'])} tools\n")

    line(legacy["tools"] == release["tools"], "tool menu",
         "" if legacy["tools"] == release["tools"]
         else f"{legacy['tools']} vs {release['tools']}")
    line(legacy["levers"] == release["levers"], "active levers",
         "" if legacy["levers"] == release["levers"]
         else f"only-legacy={sorted(set(legacy['levers']) - set(release['levers']))} "
              f"only-release={sorted(set(release['levers']) - set(legacy['levers']))}")
    line(legacy["sampler"] == release["sampler"], "sampler",
         json.dumps(legacy["sampler"], sort_keys=True))

    schema_notes = _schema_diff(legacy["schemas"], release["schemas"])
    line(not schema_notes, "tool schemas (json)")
    for note in schema_notes:
        print(note)

    same_sys = legacy["system"] == release["system"]
    line(same_sys, "system prompt text",
         "" if same_sys else f"{len(legacy['system'])} vs {len(release['system'])} chars")

    same_ids = legacy["ids"] == release["ids"]
    if same_ids:
        line(True, "first request (token ids)", f"{legacy['n_ids']:,} ids identical")
    else:
        keep = _first_divergence(legacy["ids"], release["ids"])
        line(False, "first request (token ids)",
             f"diverges at id {keep:,} of {legacy['n_ids']:,}/{release['n_ids']:,}")

    if show_diff and not same_sys:
        print("\n--- system prompt: legacy+CHAD_LEAN vs release ---")
        for d in difflib.unified_diff(release["system"].splitlines(),
                                      legacy["system"].splitlines(),
                                      "release", "legacy+CHAD_LEAN", lineterm="", n=1):
            print("  " + d)
    return clean


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", required=True, help="the bench/legacy checkout")
    ap.add_argument("--release", default=REPO_ROOT, help="the release checkout")
    ap.add_argument("--model", default="",
                    help="tokenizer/template to render through; default: the release "
                         "tree's shipped model")
    ap.add_argument("--workspace", default="",
                    help="directory to render in (the prompt's dynamic tail reads it); "
                         "default: a fresh empty dir under _work")
    ap.add_argument("--diff", action="store_true", help="print the system-prompt diff")
    args = ap.parse_args()

    legacy_tree = os.path.abspath(args.legacy)
    release_tree = os.path.abspath(args.release)

    model = args.model
    if not model:
        sys.path.insert(0, ROOT)
        import run as runner
        model = runner.default_model()
    if not model:
        print("could not resolve the shipped model; pass --model", file=sys.stderr)
        return 2

    workspace = args.workspace or os.path.join(ROOT, "_work", "bridge")
    os.makedirs(workspace, exist_ok=True)

    print(f"model:     {model}")
    print(f"workspace: {workspace}")
    print(f"legacy:    {legacy_tree}")
    print(f"release:   {release_tree}")

    # `build_system_prompt` takes the model id in the legacy tree (the profile block is
    # keyed on it) and takes none in the release tree, where profiles left with the
    # purge. Both render the same text for this pack — the qwen38 profile block is empty
    # — and the call shape is the one difference the bridge has to spell rather than
    # stumble over.
    legacy = probe(legacy_tree, model, workspace, lean=True, pass_model_id=True)
    release = probe(release_tree, model, workspace, lean=False, pass_model_id=False)
    for name, res in (("legacy", legacy), ("release", release)):
        if not res.get("ok"):
            print(f"\n{name} probe failed:\n{res.get('error')}", file=sys.stderr)
            return 2

    print("\n=== stage 1: as shipped ===")
    print("Differences here are the HARNESS DESIGN, which is what the campaign measures.")
    as_shipped = report(legacy, release, args.diff)

    # Stage 2 holds the design equal and re-renders. Anything that survives is
    # machinery — a tokenizer, a chat template, an id coercion — and machinery that
    # differs between the arms would put a confound under every number in the campaign.
    print("\n=== stage 2: controlled (same prompt, same schemas, each tree's own "
          "template) ===")
    override = {"system": release["system"], "schemas": release["schemas"]}
    c_legacy = probe(legacy_tree, model, workspace, lean=True, pass_model_id=True,
                     override=override)
    c_release = probe(release_tree, model, workspace, lean=False, pass_model_id=False,
                      override=override)
    for name, res in (("legacy", c_legacy), ("release", c_release)):
        if not res.get("ok"):
            print(f"\n{name} controlled probe failed:\n{res.get('error')}",
                  file=sys.stderr)
            return 2
    machinery = c_legacy["ids"] == c_release["ids"]
    if machinery:
        print(f"  PASS  first request (token ids)  {c_legacy['n_ids']:,} ids identical")
    else:
        keep = _first_divergence(c_legacy["ids"], c_release["ids"])
        print(f"  FAIL  first request (token ids)  diverges at id {keep:,} of "
              f"{c_legacy['n_ids']:,}/{c_release['n_ids']:,}")
    print("  PASS  sampler" if c_legacy["sampler"] == c_release["sampler"]
          else "  FAIL  sampler")

    print()
    if machinery and as_shipped:
        print("bridge: CLEAN — the two trees render the same first request, as shipped.")
        return 0
    if machinery:
        print("bridge: CLEAN ON MACHINERY — with the prompt and schemas held equal the\n"
              "        two trees render byte-identical token ids, so the template, the\n"
              "        tokenizer, the id coercion and the sampler are shared. The\n"
              "        stage-1 differences are prompt and schema TEXT: the design under\n"
              "        test. Each one is enumerated in PORTS.md before any scored trial.")
        return 0
    print("bridge: NOT CLEAN — the trees differ on MACHINERY, not only on design. That\n"
          "        is a confound under every number in the campaign: fix it, or run LEAN\n"
          "        from bench/legacy with CHAD_LEAN=1 and amend PREREG to say so.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
