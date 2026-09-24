"""A trial's proxy log as an ATIF trajectory, so `trace.py` and `stats.py` read a foreign
harness exactly as they read chad.

    python benchmarks/polyglot/proxy_atif.py crosscheck <native.json> <proxy.json> ...

One agent step per generation request of the agent's own conversation. Which requests
those are: the ones that open with the same first message — the system prompt, or, for
chad's raw `/completion`, the head of the token prompt — as the trial's most frequent
one. The title, summary and compaction requests some harnesses fire beside the loop open
with a different one; they are counted in `final_metrics.extra.side`, not made into
steps. A request that never finished — hung up by the proxy when its trial ended, or
abandoned by its own client, which asked again — becomes an in-flight step, as chad marks
the one it is still generating, counted in no totals. A
step's tool calls are the reply's, its observation the tool results the next
request of the conversation carries back, and its metrics are the server's own counts:
`prompt_tokens` includes the cached ones, as chad's do, so the two sources add up the
same way.

What the server does not count, the log cannot supply: think tokens are counted from the
reply's reasoning text with the served model's tokenizer when one is given, and left out
when not.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from catalog import JsonValue, is_array, is_number, is_object, is_text

SCHEMA_VERSION = "ATIF-v1.7"
CROSSCHECK_TOLERANCE = 0.01
_PROMPT_HEAD = 64                # tokens of a raw prompt that name its conversation


def _obj(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if is_object(value) else {}


def _items(value: JsonValue) -> Sequence[JsonValue]:
    return value if is_array(value) else ()


def _text(value: JsonValue) -> str:
    return value if is_text(value) else ""


def _num(value: JsonValue) -> float:
    return float(value) if is_number(value) else 0.0


def load(path: str) -> list[Mapping[str, JsonValue]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc: JsonValue = json.loads(line)
                if is_object(doc):
                    records.append(doc)
    return records


def _content_text(content: JsonValue) -> str:
    """A message's text, whether a string or OpenAI's list of typed parts."""
    if is_text(content):
        return content
    return "".join(_text(_obj(part).get("text")) for part in _items(content))


def _messages(request: Mapping[str, JsonValue]) -> list[Mapping[str, JsonValue]]:
    """The conversation a request carries, in chat shape: chat's `messages`, or
    Responses' `instructions` + `input` items."""
    if is_array(request.get("messages")):
        return [_obj(m) for m in _items(request.get("messages"))]
    out: list[Mapping[str, JsonValue]] = []
    if is_text(request.get("instructions")):
        out.append({"role": "system", "content": request.get("instructions")})
    raw = request.get("input")
    if is_text(raw):
        return [*out, {"role": "user", "content": raw}]
    for item in _items(raw):
        item = _obj(item)
        if item.get("type") == "function_call_output":
            out.append({"role": "tool", "tool_call_id": item.get("call_id"),
                        "content": item.get("output")})
        elif "role" in item:
            out.append(item)
    return out


def thread_key(request: Mapping[str, JsonValue]) -> str:
    prompt = request.get("prompt")
    if prompt is not None:
        head: JsonValue = list(_items(prompt))[:_PROMPT_HEAD] if is_array(prompt) \
            else _text(prompt)[:4 * _PROMPT_HEAD]
    else:
        messages = _messages(request)
        head = [dict(messages[0])] if messages else []
    return hashlib.sha256(json.dumps(head, sort_keys=True).encode()).hexdigest()[:16]


def _arguments(text: str) -> dict[str, JsonValue]:
    try:
        doc: JsonValue = json.loads(text) if text else {}
    except ValueError:
        return {"raw": text}
    return dict(doc) if is_object(doc) else {"raw": text}


def _stamp(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()


def _completion_step(reply: Mapping[str, JsonValue]) -> tuple[str, str, list[dict[str, JsonValue]]]:
    """(reasoning, visible, tool calls) of a raw `/completion` generation, split and
    parsed by the same code chad's own trajectory uses."""
    from chad.atif import split_think
    from chad.toolcall_parse import parse_tool_calls
    content = _text(reply.get("content"))
    reasoning, visible = split_think(content)
    calls = [{"tool_call_id": f"call_{k}", "function_name": name,
              "arguments": args if is_object(args) else {}}
             for k, (name, args) in enumerate(parse_tool_calls(content))]
    return reasoning, visible, calls


def convert(records: Sequence[Mapping[str, JsonValue]], agent: str, version: str,
            model: str, count_tokens: Callable[[str], int] | None = None) -> dict[str, JsonValue]:
    """One trial's records as an ATIF document."""
    keys = [thread_key(_obj(r.get("request"))) for r in records]
    main = Counter(keys).most_common(1)[0][0] if keys else ""
    agent_records = [r for r, k in zip(records, keys) if k == main]
    side = [r for r, k in zip(records, keys) if k != main]
    steps: list[dict[str, JsonValue]] = []
    if agent_records:
        first = _obj(agent_records[0].get("request"))
        for message in _messages(first):
            if message.get("role") in ("system", "developer", "user") and len(steps) < 2:
                steps.append({"source": "user" if message.get("role") == "user" else "system",
                              "message": _content_text(message.get("content")),
                              "timestamp": _stamp(_num(agent_records[0].get("t")))})
    for i, record in enumerate(agent_records):
        request, reply = _obj(record.get("request")), _obj(record.get("reply"))
        unfinished = not _obj(reply.get("timings")) and (
            record.get("aborted") is True or record.get("client_gone") is True)
        if unfinished:
            # Never finished: hung up when its trial ended (`aborted`), or abandoned by
            # its own client, which then asked again. Marked the way chad marks the step
            # it is still generating, and counted in no totals.
            steps.append({"source": "agent", "message": "", "model_name": model,
                          "timestamp": _stamp(_num(record.get("t"))),
                          "extra": {"in_flight": True,
                                    "aborted": record.get("aborted") is True}})
            continue
        if "prompt" in request:
            reasoning, visible, calls = _completion_step(reply)
        else:
            reasoning, visible = _text(reply.get("reasoning")), _text(reply.get("content"))
            calls = [{"tool_call_id": _text(_obj(c).get("id")) or f"call_{i}_{k}",
                      "function_name": _text(_obj(c).get("name")),
                      "arguments": _arguments(_text(_obj(c).get("arguments")))}
                     for k, c in enumerate(_items(reply.get("tool_calls")))]
        timings = _obj(reply.get("timings"))
        prefill_s, gen_s = _num(timings.get("prompt_ms")) / 1000, _num(timings.get("predicted_ms")) / 1000
        generated = int(_num(timings.get("predicted_n")))
        extra: dict[str, JsonValue] = {
            "prefill_s": round(prefill_s, 3), "gen_s": round(gen_s, 3),
            "tok_per_s": round(generated / gen_s, 1) if gen_s else 0.0,
            "finish": _text(reply.get("finish")), "ttfb_s": record.get("ttfb_s"),
            "sampler": record.get("sampler"), "source": "proxy"}
        if count_tokens is not None:
            extra["think_tokens"] = count_tokens(reasoning) if reasoning else 0
        step: dict[str, JsonValue] = {
            "source": "agent", "message": visible, "model_name": model,
            "timestamp": _stamp(_num(record.get("t"))),
            "metrics": {"prompt_tokens": int(_num(timings.get("prompt_n")) + _num(timings.get("cache_n"))),
                        "cached_tokens": int(_num(timings.get("cache_n"))),
                        "completion_tokens": generated, "extra": extra}}
        if reasoning.strip():
            step["reasoning_content"] = reasoning
        if calls:
            step["tool_calls"] = list(calls)
            results = _observations(calls, agent_records[i + 1] if i + 1 < len(agent_records) else None)
            if results:
                step["observation"] = {"results": results}
        steps.append(step)
    for n, step in enumerate(steps, 1):
        step["step_id"] = n
    doc: dict[str, JsonValue] = {"schema_version": SCHEMA_VERSION, "session_id": str(uuid.uuid4()),
                                 "agent": {"name": agent, "version": version, "model_name": model},
                                 "steps": steps}
    doc["final_metrics"] = _final(steps, side, records, count_tokens is not None)
    return doc


def _observations(calls: Sequence[Mapping[str, JsonValue]],
                  following: Mapping[str, JsonValue] | None) -> list[dict[str, JsonValue]]:
    """The results of `calls` as the next request of the conversation carries them back."""
    if following is None:
        return []
    ids = {_text(c.get("tool_call_id")) for c in calls}
    return [{"source_call_id": _text(m.get("tool_call_id")),
             "content": _content_text(m.get("content"))}
            for m in _messages(_obj(following.get("request")))
            if m.get("role") == "tool" and _text(m.get("tool_call_id")) in ids]


def _final(steps: Sequence[Mapping[str, JsonValue]], side: Sequence[Mapping[str, JsonValue]],
           records: Sequence[Mapping[str, JsonValue]], counted: bool) -> dict[str, JsonValue]:
    mets = [_obj(s.get("metrics")) for s in steps if "metrics" in s]
    extras = [_obj(m.get("extra")) for m in mets]
    prompt = sum(_num(m.get("prompt_tokens")) for m in mets)
    cached = sum(_num(m.get("cached_tokens")) for m in mets)
    generated = sum(_num(m.get("completion_tokens")) for m in mets)
    gen_s = sum(_num(e.get("gen_s")) for e in extras)
    timings = [_obj(_obj(r.get("reply")).get("timings")) for r in side]
    extra: dict[str, JsonValue] = {
        "total_gen_s": round(gen_s, 2),
        "total_prefill_s": round(sum(_num(e.get("prefill_s")) for e in extras), 2),
        "tok_per_s": round(generated / gen_s, 1) if gen_s else 0.0,
        "cache_hit_rate": round(cached / prompt, 3) if prompt else 0.0,
        "finishes": dict(Counter(_text(e.get("finish")) for e in extras)),
        "requests": len(records),
        "aborted": sum(1 for r in records if r.get("aborted") is True),
        "abandoned": sum(1 for r in records if r.get("client_gone") is True
                         and r.get("aborted") is not True),
        "sampler": dict(Counter(_text(r.get("sampler")) for r in records)),
        "side": {"requests": len(side),
                 "prompt_n": int(sum(_num(t.get("prompt_n")) for t in timings)),
                 "cache_n": int(sum(_num(t.get("cache_n")) for t in timings)),
                 "predicted_n": int(sum(_num(t.get("predicted_n")) for t in timings)),
                 "prefill_s": round(sum(_num(t.get("prompt_ms")) for t in timings) / 1000, 2),
                 "gen_s": round(sum(_num(t.get("predicted_ms")) for t in timings) / 1000, 2)},
        "source": "proxy"}
    if counted:
        extra["total_think_tokens"] = int(sum(_num(e.get("think_tokens")) for e in extras))
    return {"total_prompt_tokens": int(prompt), "total_completion_tokens": int(generated),
            "total_cached_tokens": int(cached), "total_steps": len(steps), "extra": extra}


def _agent_steps(doc: JsonValue) -> list[Mapping[str, JsonValue]]:
    return [_obj(s) for s in _items(_obj(doc).get("steps"))
            if _obj(s).get("source") == "agent" and _obj(_obj(s).get("extra")).get("in_flight") is not True]


def crosscheck(native: JsonValue, proxied: JsonValue,
               tolerance: float = CROSSCHECK_TOLERANCE) -> list[str]:
    """Why the proxy's reading of a trial disagrees with the agent's own, or nothing.
    The instrument is believed for foreign arms only once this is empty on chad."""
    a, b = _agent_steps(native), _agent_steps(proxied)
    problems = []
    if len(a) != len(b):
        problems.append(f"agent steps: native {len(a)}, proxy {len(b)}")

    def names(steps: Sequence[Mapping[str, JsonValue]]) -> list[str]:
        return [_text(_obj(c).get("function_name")) for s in steps for c in _items(s.get("tool_calls"))]

    if names(a) != names(b):
        problems.append(f"tool calls: native {names(a)}, proxy {names(b)}")
    for field in ("prompt_tokens", "cached_tokens"):
        x = sum(_num(_obj(s.get("metrics")).get(field)) for s in a)
        y = sum(_num(_obj(s.get("metrics")).get(field)) for s in b)
        if abs(x - y) > tolerance * max(x, y, 1):
            problems.append(f"{field}: native {int(x)}, proxy {int(y)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("crosscheck", help="native and proxy trajectories, in pairs")
    c.add_argument("pairs", nargs="+", help="native.json proxy.json [native.json proxy.json ...]")
    args = ap.parse_args()
    if len(args.pairs) % 2:
        raise SystemExit("crosscheck takes native/proxy pairs")
    failed = 0
    for native_path, proxy_path in zip(args.pairs[::2], args.pairs[1::2]):
        with open(native_path, encoding="utf-8") as f:
            native: JsonValue = json.load(f)
        with open(proxy_path, encoding="utf-8") as f:
            proxied: JsonValue = json.load(f)
        problems = crosscheck(native, proxied)
        failed += bool(problems)
        print(f"{'ok  ' if not problems else 'FAIL'} {native_path}")
        for problem in problems:
            print(f"     {problem}")
    print(f"{len(args.pairs) // 2 - failed} of {len(args.pairs) // 2} trials agree")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
