"""Seatbelt confinement for yolo-mode bash (seatbelt.py + the tool_bash seam).

The contract under test: wrapping happens ONLY when all four gates agree (lever on,
executing agent in yolo mode, platform capable, probe green); when any gate says no,
tool_bash's spawn is byte-identical to the pre-seatbelt behavior. The profile is a
deny-writes-outside-allowlist; the note appended on a detected denial is what keeps
the model from retrying into the wall. Real sandbox application is exercised in the
darwin-gated e2e at the bottom; everything else hands a Seatbelt a stand-in for
sandbox-exec so the suite passes on any platform (and inside CI/test sandboxes, where
Seatbelt cannot nest).
"""
import os
import subprocess

import pytest

from chad import config, seatbelt, tools

_real_run = subprocess.run


def _enforcing_run(argv):
    """A sandbox-exec that enforces: only the allowed half of the probe's command
    runs, so the allowed write lands and the denied one does not."""
    _real_run(["/bin/sh", "-c", argv[-1].split(";")[0]], capture_output=True, check=False)


def _capable():
    """A Seatbelt whose probe comes back green on any platform."""
    return seatbelt.Seatbelt(platform_ok=lambda: True, run=_enforcing_run)


# -- wrap gating --------------------------------------------------------------

def test_no_wrap_when_opted_out(monkeypatch):
    monkeypatch.setenv("CHAD_NO_SEATBELT", "1")
    sb = _capable()
    sb.set_context(True, os.getcwd())
    assert sb.wrap_argv("echo hi") is None


def test_no_wrap_outside_yolo_context():
    sb = _capable()
    sb.set_context(False, None)
    assert sb.wrap_argv("echo hi") is None


def test_no_wrap_when_probe_fails():
    sb = seatbelt.Seatbelt(platform_ok=lambda: True, run=lambda argv: None)
    sb.set_context(True, os.getcwd())
    assert sb.wrap_argv("echo hi") is None


def test_wrap_argv_shape(tmp_path):
    sb = _capable()
    sb.set_context(True, str(tmp_path))
    argv = sb.wrap_argv("echo hi > f.txt")
    assert argv is not None
    assert argv[0] == seatbelt.SANDBOX_EXEC and argv[1] == "-f"
    # The shell is bash where one exists (process substitution is a syntax error
    # under /bin/sh), falling back to /bin/sh on a host without it.
    assert argv[3:] == [config.shell_path(), "-c", "echo hi > f.txt"]
    assert os.path.basename(argv[3]) in ("bash", "sh")
    with open(argv[2], encoding="utf-8") as fh:
        assert str(tmp_path.resolve()) in fh.read()


def test_profile_cached_per_workspace(tmp_path):
    sb = _capable()
    sb.set_context(True, str(tmp_path))
    a = sb.wrap_argv("true")[2]
    b = sb.wrap_argv("false")[2]
    assert a == b


# -- profile content ----------------------------------------------------------

def test_profile_denies_by_default_and_allows_workspace(tmp_path):
    text = seatbelt.profile_text(str(tmp_path))
    assert "(deny file-write*)" in text
    assert f'(subpath "{tmp_path.resolve()}")' in text
    assert "(allow default)" in text
    # temp + cache roots a real command stream needs (uv/pip cache on macOS
    # lives under ~/Library/Caches, not ~/.cache)
    for needle in ("/private/var/folders", "Library/Caches", ".chad"):
        assert needle in text


def test_profile_escapes_scheme_metachars(tmp_path):
    evil = tmp_path / 'we"ird'
    evil.mkdir()
    text = seatbelt.profile_text(str(evil))
    assert 'we\\"ird' in text
    assert 'we"ird")' not in text


def _split_at_deny_tail(text: str):
    """(allow-and-before, trailing deny block) — the deny block that carves paths
    back OUT of the writable allowlist sits last, because the last matching
    Seatbelt rule wins."""
    i = text.rindex("(deny file-write*")
    return text[:i], text[i:]


def _worktree_fixture(tmp_path):
    ws = tmp_path / "wt"
    ws.mkdir()
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n")
    (ws / ".git").write_text(f"gitdir: {gitdir}\n")
    return ws, gitdir.resolve(), (gitdir / "../..").resolve()


def test_profile_worktree_gitdir_carveout(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAD_PROTECT_GIT", raising=False)
    ws, gitdir, common = _worktree_fixture(tmp_path)
    head, tail = _split_at_deny_tail(seatbelt.profile_text(str(ws)))
    assert str(gitdir) in head and str(common) in head
    assert str(gitdir) not in tail and str(common) not in tail


def test_profile_protect_git_flips_worktree_gitdirs_to_deny(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_PROTECT_GIT", "1")
    ws, gitdir, common = _worktree_fixture(tmp_path)
    head, tail = _split_at_deny_tail(seatbelt.profile_text(str(ws)))
    assert str(gitdir) in tail and str(common) in tail
    assert str(gitdir) not in head and str(common) not in head


def test_profile_protect_git_denies_workspace_dotgit(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_PROTECT_GIT", "1")
    _head, tail = _split_at_deny_tail(seatbelt.profile_text(str(tmp_path)))
    ws = tmp_path.resolve()
    assert f'(subpath "{ws / ".git"}")' in tail


def test_profile_checkpoints_denied_even_without_git_tier(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAD_PROTECT_GIT", raising=False)
    text = seatbelt.profile_text(str(tmp_path))
    head, tail = _split_at_deny_tail(text)
    ckpt = os.path.join(os.path.expanduser("~"), ".chad", "checkpoints")
    assert f'(subpath "{ckpt}")' in tail
    assert "/.git" not in tail
    # ~/.chad itself stays on the allowlist; only the undo history is carved out
    assert os.path.join(os.path.expanduser("~"), ".chad") in head


def test_profile_cache_distinguishes_git_tier(monkeypatch, tmp_path):
    sb = _capable()
    sb.set_context(True, str(tmp_path))
    monkeypatch.setenv("CHAD_PROTECT_GIT", "1")
    with_tier = sb.wrap_argv("true")[2]
    monkeypatch.delenv("CHAD_PROTECT_GIT")
    without_tier = sb.wrap_argv("true")[2]
    assert with_tier != without_tier
    with open(with_tier, encoding="utf-8") as fh:
        assert str(tmp_path.resolve() / ".git") in fh.read()
    with open(without_tier, encoding="utf-8") as fh:
        assert str(tmp_path.resolve() / ".git") not in fh.read()


def test_profile_plain_repo_no_carveout(tmp_path):
    (tmp_path / ".git").mkdir()  # normal repo: .git is a dir, no pointer to chase
    assert seatbelt._worktree_gitdirs(str(tmp_path)) == []


# -- enforcement probe --------------------------------------------------------
# probe() must prove the profile DENIES, not merely that sandbox-exec runs: a
# profile that fails open would otherwise report confinement it does not have.

def test_probe_rejects_non_enforcing_sandbox(caplog):
    """sandbox-exec runs the command fine but enforces nothing (both writes land):
    the probe must come back False, loudly — this is the fail-open case."""
    def runs_everything(argv):
        _real_run(argv[-3:], capture_output=True, check=False)
    sb = seatbelt.Seatbelt(platform_ok=lambda: True, run=runs_everything)
    with caplog.at_level("ERROR", logger="chad"):
        assert sb.probe() is False
    assert any("FAILED to enforce" in r.getMessage() for r in caplog.records)


def test_probe_rejects_sandbox_that_cannot_run():
    """Nothing executes at all (nested sandbox): neither write lands -> False."""
    sb = seatbelt.Seatbelt(platform_ok=lambda: True, run=lambda argv: None)
    assert sb.probe() is False


def test_probe_accepts_enforcing_sandbox():
    """The allowed write lands and the denied one does not -> True."""
    assert _capable().probe() is True


def test_probe_result_is_cached():
    calls = []
    def counting(argv):
        calls.append(argv)
        _enforcing_run(argv)
    sb = seatbelt.Seatbelt(platform_ok=lambda: True, run=counting)
    assert sb.probe() is True
    assert sb.probe() is True
    assert len(calls) == 1


# -- the spawned shell's environment (bash_env_guard) -------------------------

def test_bash_env_strips_credential_shaped_names(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "k")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "k")
    monkeypatch.setenv("MY_DB_PASSWORD", "k")
    monkeypatch.setenv("SOME_CLIENT_SECRET", "k")
    # The carriers the suffix list used to miss: a live agent socket, connection
    # strings with an embedded password, a bare *_KEY, a PAT, a token on disk.
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/db")
    monkeypatch.setenv("SENTRY_DSN", "https://x@sentry.io/1")
    monkeypatch.setenv("STRIPE_KEY", "k")
    monkeypatch.setenv("GH_PAT", "k")
    monkeypatch.setenv("REGISTRY_AUTH", "k")
    monkeypatch.setenv("MY_SESSION_TOKEN", "k")
    monkeypatch.setenv("GOOGLE_TOKEN_FILE", "/x/t.json")
    monkeypatch.setenv("AWS_PROFILE", "prod")
    monkeypatch.setenv("TOKEN_COUNT", "5")            # TOKEN not at the end: keep
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "1")  # likewise
    monkeypatch.setenv("CHAD_MODEL", "m")
    monkeypatch.setenv("PYTHONPATH", "/x")
    monkeypatch.setenv("HOMEBREW_KEYRING_PATH", "/x")  # KEY mid-word: keep
    env = tools._bash_env()
    assert env is not None
    for gone in ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "GITHUB_TOKEN",
                 "MY_DB_PASSWORD", "SOME_CLIENT_SECRET", "SSH_AUTH_SOCK",
                 "DATABASE_URL", "SENTRY_DSN", "STRIPE_KEY", "GH_PAT",
                 "REGISTRY_AUTH", "MY_SESSION_TOKEN", "GOOGLE_TOKEN_FILE",
                 "AWS_PROFILE"):
        assert gone not in env
    for kept in ("PATH", "HOME", "TOKEN_COUNT", "TOKENIZERS_PARALLELISM",
                 "CHAD_MODEL", "PYTHONPATH", "HOMEBREW_KEYRING_PATH"):
        assert kept in env


def test_bash_env_strips_a_url_only_when_it_carries_credentials(monkeypatch):
    """The one value check: a connection string hides its password in a name
    (REDIS_URL, MONGO_URL) that gives no hint, while a plain address must survive."""
    monkeypatch.setenv("REDIS_URL", "redis://user:pw@cache.internal:6379/0")
    monkeypatch.setenv("MONGO_URL", "mongodb://mongo:27017")
    monkeypatch.setenv("CHAD_BASE_URL", "http://localhost:8080/v1")
    env = tools._bash_env()
    assert env is not None
    assert "REDIS_URL" not in env
    assert "MONGO_URL" in env and "CHAD_BASE_URL" in env


def test_bash_env_off_means_inherit(monkeypatch):
    monkeypatch.setenv("CHAD_NO_ENV_GUARD", "1")
    assert tools._bash_env() is None


def test_bash_env_guard_always_filters(monkeypatch):
    monkeypatch.delenv("CHAD_NO_ENV_GUARD", raising=False)
    monkeypatch.setenv("SOME_API_KEY", "k")
    env = tools._bash_env()
    assert env is not None and "SOME_API_KEY" not in env


def test_bash_env_guard_end_to_end(monkeypatch):
    seatbelt.set_context(False, None)
    monkeypatch.setenv("SOME_API_KEY", "sekrit-value")
    monkeypatch.setenv("HARMLESS_SETTING", "visible-value")
    out = tools.tool_bash("printenv SOME_API_KEY; printenv HARMLESS_SETTING")
    assert "sekrit-value" not in out
    assert "visible-value" in out



# -- naming what the guard withholds, and whose commands it filters -----------

def _clear_withheld(monkeypatch):
    """Drop whatever the developer's own shell exports, so counts are exact."""
    monkeypatch.delenv("CHAD_NO_ENV_GUARD", raising=False)
    for name in tools.withheld_env_names():
        monkeypatch.delenv(name)


def test_withheld_env_names_lists_credential_names(monkeypatch):
    _clear_withheld(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "k")
    monkeypatch.setenv("AWS_PROFILE", "prod")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    names = tools.withheld_env_names()
    assert names == ["AWS_PROFILE", "GITHUB_TOKEN", "SSH_AUTH_SOCK"]
    assert "PATH" not in names


def test_withheld_env_names_includes_a_credential_url(monkeypatch):
    _clear_withheld(monkeypatch)
    monkeypatch.setenv("REDIS_URL", "redis://u:p@h/0")
    assert tools.withheld_env_names() == ["REDIS_URL"]


def test_withheld_env_names_empty_when_guard_off(monkeypatch):
    _clear_withheld(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "k")
    monkeypatch.setenv("CHAD_NO_ENV_GUARD", "1")
    assert tools.withheld_env_names() == []
    assert tools.env_guard_notice() == ""


def test_env_guard_notice_names_but_never_values(monkeypatch):
    _clear_withheld(monkeypatch)
    values = {"GITHUB_TOKEN": "ghp-value-must-not-show",
              "AWS_PROFILE": "prod-account-must-not-show",
              "REDIS_URL": "redis://u:hunter2@h/0"}
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    notice = tools.env_guard_notice()
    for k, v in values.items():
        assert k in notice
        assert v not in notice
    assert "hunter2" not in notice


def test_env_guard_notice_caps_the_list(monkeypatch):
    _clear_withheld(monkeypatch)
    for i in range(6):
        monkeypatch.setenv(f"SERVICE{i}_TOKEN", "k")
    notice = tools.env_guard_notice(limit=4)
    assert "6 credential-shaped" in notice
    assert "+2 more" in notice


def test_the_agents_commands_stay_guarded(monkeypatch):
    monkeypatch.delenv("CHAD_NO_ENV_GUARD", raising=False)
    monkeypatch.setenv("DEPLOY_TOKEN", "value-that-must-not-appear")
    out = tools.tool_bash('printf "%s" "${DEPLOY_TOKEN:-unset}"', wrap=lambda c: None)
    assert out == "unset"


def test_the_users_own_command_is_not_filtered(monkeypatch):
    monkeypatch.delenv("CHAD_NO_ENV_GUARD", raising=False)
    monkeypatch.setenv("DEPLOY_TOKEN", "present")
    out = tools.tool_bash('printf "%s" "${DEPLOY_TOKEN:-unset}"', wrap=lambda c: None,
                          env_guard=False)
    assert out == "present"


def test_the_dispatched_bash_tool_keeps_the_guard(monkeypatch):
    """The agent's path through the dispatch table must never inherit the
    passthrough's exemption."""
    seatbelt.set_context(False, None)
    monkeypatch.delenv("CHAD_NO_ENV_GUARD", raising=False)
    monkeypatch.setenv("DEPLOY_TOKEN", "value-that-must-not-appear")
    out = tools.DISPATCH["bash"]({"command": 'printf "%s" "${DEPLOY_TOKEN:-unset}"'}, None)
    assert out == "unset"


# -- the tool_bash seam -------------------------------------------------------

def test_tool_bash_unwrapped_runs_plain_shell():
    seatbelt.set_context(False, None)
    out = tools.tool_bash("echo plain")
    assert "plain" in out


def test_tool_bash_denial_note_and_fire():
    """A wrapped command whose output shows the EPERM marker gets the explanatory
    note appended."""
    fake = ["/bin/sh", "-c", "echo 'x: Operation not permitted'; exit 1"]
    out = tools.tool_bash("anything", wrap=lambda cmd: fake)
    assert "Operation not permitted" in out
    assert "seatbelt:" in out


def test_tool_bash_wrapped_clean_run_no_note():
    fake = ["/bin/sh", "-c", "echo all good"]
    out = tools.tool_bash("anything", wrap=lambda cmd: fake)
    assert "all good" in out
    assert "seatbelt:" not in out


def test_unwrapped_denial_output_gets_no_note():
    """'Operation not permitted' from an UNSANDBOXED command (plain EPERM) must not
    be blamed on the seatbelt."""
    seatbelt.set_context(False, None)
    out = tools.tool_bash("echo 'y: Operation not permitted'")
    assert "seatbelt:" not in out


# -- real sandbox e2e (darwin only, skipped wherever Seatbelt can't apply) -----

# Gate on the real enforcement probe, not a permissive-profile smoke test: inside a
# CI/harness sandbox a permissive profile can still apply while a deny profile fails
# open — exactly the environment where these tests must skip, not fail.
_can_sandbox = seatbelt.Seatbelt().probe()


@pytest.mark.skipif(not _can_sandbox, reason="Seatbelt cannot apply here")
def test_e2e_denies_outside_write_allows_inside(tmp_path):
    sb = seatbelt.Seatbelt()
    sb.set_context(True, str(tmp_path))
    argv = sb.wrap_argv(f"echo ok > {tmp_path}/in.txt")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and (tmp_path / "in.txt").read_text().strip() == "ok"

    probe = os.path.expanduser("~/chad_seatbelt_test_probe")
    argv = sb.wrap_argv(f"touch {probe}")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    try:
        assert r.returncode != 0
        assert not os.path.exists(probe)
    finally:
        if os.path.exists(probe):  # belt failed: don't leave droppings
            os.unlink(probe)


@pytest.mark.skipif(not _can_sandbox, reason="Seatbelt cannot apply here")
def test_e2e_enforcement_probe_green():
    assert seatbelt.probe() is True


@pytest.mark.skipif(not _can_sandbox, reason="Seatbelt cannot apply here")
def test_e2e_protect_git_denies_gitdir_write(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_PROTECT_GIT", "1")
    (tmp_path / ".git").mkdir()
    sb = seatbelt.Seatbelt()
    sb.set_context(True, str(tmp_path))
    argv = sb.wrap_argv(f"touch {tmp_path}/.git/droppings")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert r.returncode != 0
    assert not (tmp_path / ".git" / "droppings").exists()
    # the workspace around it stays writable
    argv = sb.wrap_argv(f"echo ok > {tmp_path}/normal.txt")
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and (tmp_path / "normal.txt").read_text().strip() == "ok"
