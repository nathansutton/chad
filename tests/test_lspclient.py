"""Protocol mechanics of chad's generic LSP client (lspclient.py), against a
scripted fake server subprocess — real pipes, real Content-Length frames, no
language server install. This is the layer that had NEVER been exercised end to
end while solidlsp owned it (test_lsp.py mocked the package absent).

Covered: framing + request/response correlation, initialize handshake,
server→client requests answered (a server blocked on workspace/configuration
looks exactly like a hang), logMessage capture (readiness rules grep it),
publishDiagnostics generations (wait for OUR edit's analysis, not a stale set),
didOpen/didChange versioning with the unchanged-content skip, result-shape
normalization (Location vs Location[] vs LocationLink[], hover content shapes),
request timeout, and shutdown.
"""

import os
import sys

import pytest

from chad.lspclient import LspClient, path_to_uri, uri_to_path

# A minimal LSP server: enough of the protocol to exercise every client seam.
FAKE_SERVER = r"""
import json, sys

def send(payload):
    body = json.dumps(payload).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
    sys.stdout.buffer.flush()

def notify(method, params):
    send({"jsonrpc": "2.0", "method": method, "params": params})

opens = {}
while True:
    line = sys.stdin.buffer.readline()
    if not line:
        break
    if not line.lower().startswith(b"content-length:"):
        continue
    n = int(line.split(b":", 1)[1])
    sys.stdin.buffer.readline()                 # the blank line
    msg = json.loads(sys.stdin.buffer.read(n))
    m, rid = msg.get("method"), msg.get("id")
    p = msg.get("params") or {}
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"capabilities": {}}})
        notify("window/logMessage", {"type": 3, "message": "Found 3 source files"})
        send({"jsonrpc": "2.0", "id": 999, "method": "workspace/configuration",
              "params": {"items": [{"section": "x"}, {"section": "y"}]}})
    elif rid == 999 and m is None:              # the client's configuration answer
        assert msg["result"] == [None, None], msg
        notify("window/logMessage", {"type": 3, "message": "config-answered"})
    elif m in ("textDocument/didOpen", "textDocument/didChange"):
        td = p["textDocument"]
        opens[td["uri"]] = opens.get(td["uri"], 0) + 1
        notify("textDocument/publishDiagnostics", {
            "uri": td["uri"],
            "diagnostics": [{"range": {"start": {"line": 0, "character": 0},
                                       "end": {"line": 0, "character": 1}},
                            "severity": 1,
                            "message": "boom v%d" % td.get("version", 0)}]})
    elif m == "textDocument/references":
        send({"jsonrpc": "2.0", "id": rid, "result": [
            {"uri": "file:///tmp/other.py",
             "range": {"start": {"line": 9, "character": 2},
                       "end": {"line": 9, "character": 8}}}]})
    elif m == "textDocument/definition":        # single Location, NOT an array
        send({"jsonrpc": "2.0", "id": rid, "result":
              {"uri": p["textDocument"]["uri"],
               "range": {"start": {"line": 4, "character": 0},
                         "end": {"line": 4, "character": 3}}}})
    elif m == "textDocument/hover":
        send({"jsonrpc": "2.0", "id": rid, "result":
              {"contents": {"kind": "markdown", "value": "(function) helper"}}})
    elif m == "textDocument/diagnostic":
        send({"jsonrpc": "2.0", "id": rid, "result":
              {"kind": "full", "items": [{"range": {"start": {"line": 2, "character": 0},
                                                    "end": {"line": 2, "character": 1}},
                                          "severity": 2, "message": "pulled"}]}})
    elif m == "test/error":
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601,
                                                     "message": "nope"}})
    elif m == "test/never":
        pass                                    # exercise the client timeout
    elif m == "shutdown":
        send({"jsonrpc": "2.0", "id": rid, "result": None})
    elif m == "exit":
        sys.exit(0)
"""


@pytest.fixture
def client(tmp_path):
    cli = LspClient([sys.executable, "-u", "-c", FAKE_SERVER], str(tmp_path))
    yield cli
    cli.shutdown()


def test_uri_round_trip(tmp_path):
    p = str(tmp_path / "dir with space" / "f.py")
    assert uri_to_path(path_to_uri(p)) == p


def test_initialize_and_server_requests(client):
    result = client.initialize()
    assert result == {"capabilities": {}}
    # readiness grep sees past AND future log lines
    assert client.wait_log(r"Found \d+ source files", timeout=5.0) is True
    # the server->client workspace/configuration request was answered [null]*items
    # (the fake acks by logging; an unanswered request would log nothing)
    assert client.wait_log("config-answered", timeout=5.0) is True
    assert client.wait_log("never-logged", timeout=0.1) is False


def test_references_definition_hover(client, tmp_path):
    client.initialize()
    f = str(tmp_path / "f.py")
    open(f, "w").write("x = 1\n")
    client.sync_doc(f, "python")
    assert client.references(f, 0, 0) == [("/tmp/other.py", 10, 3)]
    assert client.definition(f, 0, 0) == [(f, 5)]  # single-Location normalization
    assert client.hover(f, 0, 0) == "(function) helper"


def test_diagnostic_generations_and_sync_skip(client, tmp_path):
    client.initialize()
    f = str(tmp_path / "g.py")
    open(f, "w").write("y = 2\n")
    gen0 = client.diag_generation(f)
    client.sync_doc(f, "python")                       # didOpen v1 -> publish
    diags = client.wait_diagnostics(f, gen0, timeout=5.0)
    assert diags and diags[0]["message"] == "boom v1"

    # unchanged content: NO didChange goes out, so no new publish arrives
    gen1 = client.diag_generation(f)
    client.sync_doc(f, "python")
    assert client.wait_diagnostics(f, gen1, timeout=0.3) is None

    # changed content: didChange v2 -> a NEW publish (generation advances)
    open(f, "w").write("y = 3\n")
    client.sync_doc(f, "python")
    diags = client.wait_diagnostics(f, gen1, timeout=5.0)
    assert diags and diags[0]["message"] == "boom v2"


def test_pull_diagnostics(client, tmp_path):
    client.initialize()
    f = str(tmp_path / "p.py")
    open(f, "w").write("z = 1\n")
    items = client.pull_diagnostics(f)
    assert items and items[0]["message"] == "pulled"


def test_error_and_timeout(client):
    client.initialize()
    from chad.lspclient import LspError
    with pytest.raises(LspError, match="nope"):
        client.request("test/error")
    with pytest.raises(TimeoutError):
        client.request("test/never", timeout=0.2)
    # the client survives both and still answers
    assert client.request("shutdown") is None


def test_shutdown_reaps_process(client):
    client.initialize()
    assert client.alive() is True
    client.shutdown()
    assert client.alive() is False


def test_dead_server_fails_pending(tmp_path):
    cli = LspClient([sys.executable, "-c", "import sys; sys.stdin.read(40)"],
                    str(tmp_path))
    # server never answers and then exits: pending requests fail, not hang forever
    with pytest.raises((TimeoutError, Exception)):
        cli.request("initialize", timeout=3.0)
    cli.shutdown()
    assert not cli.alive()


def test_open_docs_track_versions(client, tmp_path):
    client.initialize()
    f = str(tmp_path / "v.py")
    open(f, "w").write("a = 1\n")
    uri = client.sync_doc(f, "python")
    assert client._open_docs[uri][0] == 1
    client.sync_doc(f, "python", text="a = 2\n")
    assert client._open_docs[uri][0] == 2
    client.sync_doc(f, "python", text="a = 2\n")   # identical: version unchanged
    assert client._open_docs[uri][0] == 2


def test_ignores_path_separators_in_messages(client, tmp_path):
    """Frames containing multibyte text survive length-based framing."""
    client.initialize()
    f = str(tmp_path / "unicode.py")
    open(f, "w").write("s = 'héllo — ünïcode'\n")
    client.sync_doc(f, "python")
    assert client.wait_diagnostics(f, 0, timeout=5.0) is not None


def test_fake_server_is_a_real_subprocess(client):
    assert client.pid and client.pid != os.getpid()
