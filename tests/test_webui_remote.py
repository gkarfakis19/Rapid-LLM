from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from webui.service import core
from webui.service import remote
from webui.service import remote_runner


def test_execution_mode_defaults_to_local(monkeypatch, tmp_path):
    monkeypatch.delenv("RAPID_WEBUI_EXECUTION_MODE", raising=False)
    monkeypatch.setenv("RAPID_WEBUI_REMOTE_CONFIG", str(tmp_path / "missing.json"))

    config = remote.load_execution_config()

    assert config.mode == "local"
    assert not config.enabled


def test_remote_config_requires_explicit_host_and_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_WEBUI_EXECUTION_MODE", "remote_ssh")
    monkeypatch.setenv("RAPID_WEBUI_REMOTE_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_HOST", raising=False)
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_USER", raising=False)
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_REPO", raising=False)
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_BRANCH", raising=False)
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_WORKSPACE", raising=False)

    config = remote.load_execution_config()

    assert config.enabled
    assert config.host == ""
    assert config.user is None
    assert config.repo == ""
    assert config.branch == "remote_backend"
    assert config.workspace == ""
    assert any("Remote host is not configured" in item for item in remote.remote_config_errors(config))


def test_remote_config_can_be_loaded_from_ignored_local_json(monkeypatch, tmp_path):
    config_path = tmp_path / "remote_backend.local.json"
    config_path.write_text(
        json.dumps(
            {
                "host": "ssh.example.com",
                "user": "alice",
                "repo": "/srv/rapid-llm",
            }
        )
    )
    monkeypatch.setenv("RAPID_WEBUI_REMOTE_CONFIG", str(config_path))
    monkeypatch.setenv("RAPID_WEBUI_EXECUTION_MODE", "remote_ssh")
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_HOST", raising=False)
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_USER", raising=False)
    monkeypatch.delenv("RAPID_WEBUI_REMOTE_REPO", raising=False)

    config = remote.load_execution_config()

    assert config.enabled
    assert config.host == "ssh.example.com"
    assert config.user == "alice"
    assert config.repo == "/srv/rapid-llm"
    assert config.workspace == "/srv/rapid-llm/webui/workspace"
    assert config.python == "/srv/rapid-llm/.venv/bin/python"


def test_git_freshness_detects_match_mismatch_and_probe_failure():
    clean_frontend = {"ok": True, "branch": "remote_backend", "commit": "abc", "short_commit": "abc", "dirty_tracked": False}
    clean_backend = {"ok": True, "branch": "remote_backend", "commit": "abc", "short_commit": "abc", "dirty_tracked": False}

    assert remote.compare_git_freshness(clean_frontend, clean_backend, "remote_backend")["can_launch"] is True

    mismatch = remote.compare_git_freshness(clean_frontend, clean_backend | {"commit": "def"}, "remote_backend")
    assert mismatch["status"] == "mismatch"
    assert mismatch["can_launch"] is False

    stale = remote.compare_git_freshness(clean_frontend | {"dirty_tracked": True}, clean_backend, "remote_backend")
    assert stale["status"] == "match"
    assert stale["can_launch"] is True

    failed = remote.compare_git_freshness(clean_frontend, {"ok": False, "error": "ssh failed"}, "remote_backend")
    assert failed["status"] == "probe_failed"
    assert failed["can_launch"] is False


def test_remote_ssh_command_construction_uses_arrays_and_constrained_paths():
    config = remote.RemoteSshConfig(
        mode="remote_ssh",
        host="ssh.example.com",
        user="alice",
        repo="/srv/rapid-llm",
        branch="remote_backend",
        workspace="/srv/rapid-llm/webui/workspace",
        python="/srv/rapid-llm/.venv/bin/python",
    )
    client = remote.RemoteSshClient(config)

    argv = client.runner_ssh_argv(["probe", "--json"])

    assert argv[0] == "ssh"
    assert "alice@ssh.example.com" in argv
    assert argv[-1].startswith("/bin/bash -lc ")
    assert "cd /srv/rapid-llm && " in argv[-1]
    assert "webui.service.remote_runner probe --json" in argv[-1]
    with pytest.raises(ValueError):
        client.upload_dir(Path("."), "/tmp/outside-workspace")


def test_rsync_upload_command_is_safe_and_path_constrained(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    config = remote.RemoteSshConfig(
        mode="remote_ssh",
        host="ssh.example.com",
        user=None,
        repo="/repo",
        branch="remote_backend",
        workspace="/repo/webui/workspace",
        python="/repo/.venv/bin/python",
    )
    client = remote.RemoteSshClient(config)

    monkeypatch.setattr(remote.shutil, "which", lambda name: "/usr/bin/rsync" if name == "rsync" else None)

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    local_dir = tmp_path / "job"
    local_dir.mkdir()

    client.upload_dir(local_dir, "/repo/webui/workspace/runs/run-1")

    assert calls[0][:-1] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "ssh.example.com"]
    assert calls[0][-1].startswith("/bin/bash -lc ")
    assert "mkdir -p /repo/webui/workspace/runs/run-1" in calls[0][-1]
    assert calls[1] == ["rsync", "-a", "--delete", f"{local_dir.resolve()}/", "ssh.example.com:/repo/webui/workspace/runs/run-1/"]


def test_remote_event_parser_handles_heartbeat_duplicates_and_malformed_json():
    state = remote.RemoteEventState()

    assert state.apply_line(json.dumps({"seq": 1, "type": "status", "status": "running"}))["status"] == "running"
    assert state.last_seq == 1
    assert state.apply_line(json.dumps({"seq": 1, "type": "status", "status": "running"})) is None
    assert state.duplicate_count == 1
    assert state.apply_line(json.dumps({"type": "heartbeat", "status": "running"}))["type"] == "heartbeat"
    assert state.last_seq == 1
    assert state.apply_line("{bad json") is None
    assert state.malformed_count == 1
    assert state.apply_line(json.dumps({"seq": 2, "type": "completed", "status": "completed"}))["type"] == "completed"


def test_remote_stream_exits_on_terminal_status_without_events(monkeypatch, tmp_path, capsys):
    workspace = tmp_path / "workspace"
    job_root = workspace / "sweeps" / "job-1"
    job_root.mkdir(parents=True)
    (job_root / "status.json").write_text(json.dumps({"status": "cancelled", "progress_completed": 1, "progress_total": 4}))
    monkeypatch.setenv("RAPID_WEBUI_REMOTE_WORKSPACE", str(workspace))

    rc = remote_runner.command_stream(SimpleNamespace(job_root=str(job_root), from_seq="0", heartbeat_interval="1"))

    captured = capsys.readouterr()
    assert rc == 0
    heartbeat = json.loads(captured.out.strip().splitlines()[-1])
    assert heartbeat["type"] == "heartbeat"
    assert heartbeat["status"] == "cancelled"


def test_remote_executor_treats_cancelled_heartbeat_as_terminal(tmp_path):
    executor = remote.RemoteSshExecutor.__new__(remote.RemoteSshExecutor)
    updates = []

    terminal = executor._handle_event(
        {"type": "heartbeat", "status": "cancelled", "progress_total": 4, "progress_completed": 1, "status_record": {"status": "cancelled"}},
        tmp_path,
        "/remote/job",
        lambda job_root, **kwargs: updates.append((job_root, kwargs)),
    )

    assert terminal is True
    assert updates[-1][1]["status"] == "cancelled"
    assert updates[-1][1]["remote_sync_state"] == "streaming"


def test_remote_run_job_cancel_after_upload_does_not_start_supervisor(tmp_path):
    config = remote.RemoteSshConfig(
        mode="remote_ssh",
        host="ssh.example.com",
        user=None,
        repo="/repo",
        branch="remote_backend",
        workspace="/repo/webui/workspace",
        python="/repo/.venv/bin/python",
    )
    executor = remote.RemoteSshExecutor(config)
    cancel_event = threading.Event()
    uploads = []
    starts = []

    class FakeClient:
        def upload_dir(self, local_dir, remote_dir):
            uploads.append((Path(local_dir), remote_dir, json.loads((Path(local_dir) / "status.json").read_text())))
            cancel_event.set()

        def run_json(self, args, *, timeout):
            starts.append((list(args), timeout))
            return {"pid": 123}

    executor.client = FakeClient()
    job_root = tmp_path / "local-job"
    job_root.mkdir()
    (job_root / "status.json").write_text(json.dumps({"status": "queued", "created_at": "2026-05-02T00:00:00+00:00"}))

    def status_writer(path, **kwargs):
        status_path = Path(path) / "status.json"
        record = json.loads(status_path.read_text()) if status_path.exists() else {}
        record.update({key: value for key, value in kwargs.items() if value is not None})
        status_path.write_text(json.dumps(record))

    summary = executor.run_job(
        job_root=job_root,
        job_kind="sweep",
        payload={},
        preview={},
        local_root=Path.cwd(),
        active_record_path=tmp_path / "remote_active_job.json",
        status_writer=status_writer,
        cancel_event=cancel_event,
    )

    assert summary["status"] == "cancelled"
    assert starts == []
    assert len(uploads) == 2
    assert uploads[-1][2]["status"] == "cancelled"


def test_remote_telemetry_replaces_local_psutil_in_remote_mode(monkeypatch):
    monkeypatch.setenv("RAPID_WEBUI_EXECUTION_MODE", "remote_ssh")
    monkeypatch.setenv("RAPID_WEBUI_REMOTE_HOST", "ssh.example.com")
    monkeypatch.setenv("RAPID_WEBUI_REMOTE_REPO", "/srv/rapid-llm")
    monkeypatch.setattr(core, "get_remote_executor", lambda config=None: SimpleNamespace(client=object()))

    class FakeMonitor:
        def __init__(self, _client):
            pass

        def get(self):
            return {"source": "remote", "host": "ssh.example.com", "available_ram_gb": 123.4, "cpu_percent": 5.5, "stale": False}

    monkeypatch.setattr(core, "RemoteTelemetryMonitor", FakeMonitor)
    monkeypatch.setattr(core, "_REMOTE_TELEMETRY_MONITOR", None)

    telemetry = core.get_telemetry()

    assert telemetry["source"] == "remote"
    assert telemetry["available_ram_gb"] == 123.4
    assert telemetry["cpu_percent"] == 5.5


def test_remote_telemetry_monitor_uses_one_shot_poll_not_stream():
    calls = []

    class FakeClient:
        config = SimpleNamespace(host="ssh.example.com", connect_timeout_s=8)

        def run_json(self, args, *, timeout):
            calls.append((list(args), timeout))
            return {
                "available_ram_gb": 98.7,
                "used_percent": 12.3,
                "cpu_percent": 4.5,
                "cpu_count": 64,
                "updated_at": "2026-05-02T00:00:00+00:00",
            }

    telemetry = remote.RemoteTelemetryMonitor(FakeClient(), cache_s=0).get()

    assert calls == [(["telemetry"], 13.0)]
    assert telemetry["source"] == "remote"
    assert telemetry["stale"] is False
    assert telemetry["cpu_count"] == 64
