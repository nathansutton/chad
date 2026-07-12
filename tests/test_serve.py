"""Tests for chad --serve (serve.py, plan 075 WS3) — the OpenAI-compatible surface
on the in-process engine.

Two tiers, both model-free:
  1. The pure helpers: pre-rendered detection, the exact-id suffix splice that keeps
     the prefix cache warm across requests, request parsing, response shapes.
  2. A REAL HTTP round-trip against a fake engine on an OS-picked port — request 1
     cold, request 2 extending request 1's transcript. The load-bearing assertion is
     that request 2's prompt ids are the session ids PLUS an encode of only the
     appended suffix (never a full re-encode, which risks BPE merges across the
     generation boundary and would desync the non-trimmable cache).
"""
import json
import threading
import urllib.request

from chad.base_engine import GenStats
from chad.serve import (
    _Session,
    chunk_payload,
    extend_ids,
    is_prerendered,
    make_server,
    parse_request,
    usage_dict,
)

passed = 0
failed = 0


def check(desc, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        raise AssertionError(f"{desc}{(' — ' + detail) if detail else ''}")


class FakeTok:
    """Encode = one id per character (ord). Deterministic, records every encode call
    so tests can assert the suffix-splice encoded ONLY the appended text."""

    def __init__(self):
        self.encode_calls = []

    def encode(self, text, add_special_tokens=False):
        self.encode_calls.append(text)
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True):
        joined = "|".join(m["content"] for m in messages)
        return [ord(c) for c in joined]


class FakeEngine:
    model_id = "fake-model"
    temp = 0.0

    def __init__(self):
        self.tok = FakeTok()
        self._cached_ids = []
        self.generate_prompts = []

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, stop_texts=None,
                 should_stop=None, **kw):
        self.generate_prompts.append(list(prompt_ids))
        text = "OK!"
        if on_token:
            for ch in text:
                on_token(ch)
        gen_ids = [ord(c) for c in text]
        self._cached_ids = list(prompt_ids) + gen_ids
        return text, GenStats(prompt_tokens=len(prompt_ids), cached_tokens=0,
                              generated_tokens=len(gen_ids), gen_s=0.01)


def test_pure_helpers():
    # -- is_prerendered ------------------------------------------------------
    check("rendered single-user message detected",
          is_prerendered([{"role": "user", "content": "<|im_start|>system\nhi"}]))
    check("plain chat is not prerendered",
          not is_prerendered([{"role": "user", "content": "hello"}]))
    check("multi-message is not prerendered",
          not is_prerendered([{"role": "system", "content": "<|im_start|>"},
                              {"role": "user", "content": "hi"}]))

    # -- extend_ids ----------------------------------------------------------
    tok = FakeTok()
    prev_ids = [ord(c) for c in "abc"]
    got = extend_ids(tok, "abc", "abcdef", prev_ids)
    check("suffix splice keeps prev ids and encodes only the suffix",
          got == prev_ids + [ord("d"), ord("e"), ord("f")], got)
    check("only the suffix was encoded", tok.encode_calls == ["def"],
          tok.encode_calls)
    check("identical text returns prev ids", extend_ids(tok, "abc", "abc", prev_ids)
          == prev_ids)
    check("non-extending text returns None",
          extend_ids(tok, "abc", "xbcd", prev_ids) is None)
    check("empty session returns None", extend_ids(tok, "", "abc", []) is None)

    # -- parse_request / shapes ----------------------------------------------
    req = parse_request({"messages": [{"role": "user", "content": "x"}],
                         "stream": True, "max_tokens": 64, "stop": "END"})
    check("stop string becomes a list", req["stop"] == ["END"], req)
    check("stream parsed", req["stream"] is True)
    check("default max_tokens", parse_request({"messages": []})["max_tokens"] == 2048)

    u = usage_dict(prompt_tokens=10, cached=90, completion=5)
    check("usage counts cached in prompt_tokens", u["prompt_tokens"] == 100, u)
    check("usage exposes cached_tokens detail",
          u["prompt_tokens_details"]["cached_tokens"] == 90, u)

    c = chunk_payload("m", 0, "id", delta="hi")
    check("delta chunk shape", c["choices"][0]["delta"]["content"] == "hi", c)
    c = chunk_payload("m", 0, "id", usage=u)
    check("usage chunk has empty choices", c["choices"] == [] and c["usage"] == u, c)

    # -- _Session.update ------------------------------------------------------
    s = _Session()
    s.update("prompt", [1, 2], "gen", [1, 2, 9, 9])
    check("session adopts engine ids when they extend the prompt",
          s.ids == [1, 2, 9, 9] and s.text == "promptgen", (s.ids, s.text))
    s.update("prompt", [1, 2], "gen", [])   # engine degraded (e.g. OOM reset)
    check("session drops itself when the engine cache was reset",
          s.ids == [] and s.text == "", (s.ids, s.text))


def _post(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode()


def test_http_round_trip():
    eng = FakeEngine()
    httpd = make_server(eng, port=0)   # OS-picked port
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        # /v1/models (the TB2 runner's readiness probe) and /health answer.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                    timeout=10) as r:
            models = json.loads(r.read())
        check("models lists the engine model",
              models["data"][0]["id"] == "fake-model", models)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=10) as r:
            check("health answers", json.loads(r.read())["status"] == "ok")

        # Request 1: cold, pre-rendered, non-streaming.
        text1 = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ"
        out = json.loads(_post(port, {"messages": [{"role": "user", "content": text1}],
                                      "stream": False}))
        check("completion text returned",
              out["choices"][0]["message"]["content"] == "OK!", out)
        check("usage present", out["usage"]["completion_tokens"] == 3, out)
        ids1 = eng.generate_prompts[0]
        check("cold request encoded the full text", ids1 == [ord(c) for c in text1])

        # Request 2: the transcript GREW (assistant reply + a new user turn appended)
        # — the exact append-only agentic shape. The prompt ids must be the session's
        # known ids plus an encode of ONLY the suffix.
        text2 = text1 + "OK!" + "<|im_start|>user\nnext"
        eng.tok.encode_calls.clear()
        json.loads(_post(port, {"messages": [{"role": "user", "content": text2}],
                                "stream": False}))
        ids2 = eng.generate_prompts[1]
        want = ids1 + [ord(c) for c in "OK!"] + \
            [ord(c) for c in "<|im_start|>user\nnext"]
        check("warm request splices session ids + suffix", ids2 == want,
              f"{len(ids2)} vs {len(want)}")
        check("warm request encoded ONLY the suffix",
              eng.tok.encode_calls == ["<|im_start|>user\nnext"],
              eng.tok.encode_calls)

        # Request 3: streaming — SSE chunks + usage + [DONE].
        raw = _post(port, {"messages": [{"role": "user", "content":
                                         text2 + "OK!" + "<|im_start|>user\nz"}],
                           "stream": True})
        payloads = [json.loads(ln[len("data: "):]) for ln in raw.splitlines()
                    if ln.startswith("data: ") and "[DONE]" not in ln]
        deltas = "".join(p["choices"][0]["delta"].get("content", "")
                         for p in payloads if p.get("choices"))
        check("stream deltas reassemble the text", deltas == "OK!", deltas)
        check("stream ends with a usage chunk",
              any(p.get("usage") for p in payloads), raw[-200:])
        check("stream terminates with DONE", "data: [DONE]" in raw)

        # Generic (non-prerendered) multi-message request: chat template applied.
        json.loads(_post(port, {"messages": [{"role": "system", "content": "s"},
                                             {"role": "user", "content": "u"}],
                                "stream": False}))
        check("generic request goes through the chat template",
              eng.generate_prompts[-1] == [ord(c) for c in "s|u"],
              eng.generate_prompts[-1])
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    test_pure_helpers()
    test_http_round_trip()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
