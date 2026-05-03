from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


RUNNER_PROTOCOL_VERSION = 1
REMOTE_ACTIVE_JOB_FILENAME = "remote_active_job.json"
REMOTE_TERMINAL_STATUSES = {"completed", "failed", "partial", "cancelled", "timed_out"}
REMOTE_TRANSIENT_STATES = {"connecting", "streaming", "reconnecting", "syncing", "stale", "failed"}
DEFAULT_REMOTE_BRANCH = "remote_backend"
DEFAULT_REMOTE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "remote_backend.local.json"


@dataclass(frozen=True)
class RemoteSshConfig:
    mode: str
    host: str
    user: str | None
    repo: str
    branch: str
    workspace: str
    python: str
    connect_timeout_s: int = 8
    heartbeat_interval_s: int = 1
    config_path: str | None = None
    config_error: str | None = None
    remote_env: Dict[str, str] = field(default_factory=dict)

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def enabled(self) -> bool:
        return self.mode == "remote_ssh"


def _env_int(environ: Dict[str, str], name: str, default: int) -> int:
    try:
        return int(environ.get(name, "") or default)
    except ValueError:
        return default


def _remote_config_path(environ: Dict[str, str]) -> Path:
    return Path(environ.get("RAPID_WEBUI_REMOTE_CONFIG") or DEFAULT_REMOTE_CONFIG_PATH).expanduser()


def _load_local_remote_config(environ: Dict[str, str]) -> tuple[Dict[str, Any], str | None, str | None]:
    path = _remote_config_path(environ)
    if not path.exists():
        return {}, str(path), None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(path), str(exc)
    if not isinstance(payload, dict):
        return {}, str(path), "Remote config file must contain a JSON object."
    return payload, str(path), None


def _config_value(env: Dict[str, str], local_config: Dict[str, Any], env_name: str, config_name: str, default: str = "") -> str:
    if env.get(env_name) is not None:
        return str(env.get(env_name) or "").strip()
    value = local_config.get(config_name, default)
    return str(value or "").strip()


def _config_env(environ: Dict[str, str], local_config: Dict[str, Any]) -> Dict[str, str]:
    raw_env = local_config.get("env") if isinstance(local_config.get("env"), dict) else {}
    merged = {str(key): str(value) for key, value in raw_env.items() if str(key).replace("_", "").isalnum()}
    env_json = environ.get("RAPID_WEBUI_REMOTE_ENV_JSON")
    if env_json:
        try:
            payload = json.loads(env_json)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            merged.update({str(key): str(value) for key, value in payload.items() if str(key).replace("_", "").isalnum()})
    return merged


def load_execution_config(environ: Dict[str, str] | None = None) -> RemoteSshConfig:
    env = os.environ if environ is None else environ
    local_config, config_path, config_error = _load_local_remote_config(env)
    mode = str(env.get("RAPID_WEBUI_EXECUTION_MODE") or "local").strip().lower()
    repo = _config_value(env, local_config, "RAPID_WEBUI_REMOTE_REPO", "repo").rstrip("/")
    workspace_default = f"{repo}/webui/workspace" if repo else ""
    python_default = f"{repo}/.venv/bin/python" if repo else ""
    workspace = _config_value(env, local_config, "RAPID_WEBUI_REMOTE_WORKSPACE", "workspace", workspace_default).rstrip("/")
    python = _config_value(env, local_config, "RAPID_WEBUI_REMOTE_PYTHON", "python", python_default)
    user = _config_value(env, local_config, "RAPID_WEBUI_REMOTE_USER", "user")
    return RemoteSshConfig(
        mode=mode,
        host=_config_value(env, local_config, "RAPID_WEBUI_REMOTE_HOST", "host"),
        user=user or None,
        repo=repo,
        branch=_config_value(env, local_config, "RAPID_WEBUI_REMOTE_BRANCH", "branch", DEFAULT_REMOTE_BRANCH),
        workspace=workspace,
        python=python,
        connect_timeout_s=_env_int(env, "RAPID_WEBUI_REMOTE_CONNECT_TIMEOUT", 8),
        heartbeat_interval_s=max(1, _env_int(env, "RAPID_WEBUI_REMOTE_HEARTBEAT_INTERVAL", 1)),
        config_path=config_path,
        config_error=config_error,
        remote_env=_config_env(env, local_config),
    )


def is_remote_mode(config: RemoteSshConfig | None = None) -> bool:
    return (config or load_execution_config()).enabled


def remote_config_errors(config: RemoteSshConfig) -> list[str]:
    if config.mode not in {"local", "remote_ssh"}:
        return [f"Unsupported RAPID_WEBUI_EXECUTION_MODE={config.mode!r}."]
    if not config.enabled:
        return []
    errors: list[str] = []
    if config.config_error:
        errors.append(f"Could not read remote config {config.config_path}: {config.config_error}")
    if not config.host:
        errors.append("Remote host is not configured. Set RAPID_WEBUI_REMOTE_HOST or webui/remote_backend.local.json.")
    if not config.repo:
        errors.append("Remote repo is not configured. Set RAPID_WEBUI_REMOTE_REPO or webui/remote_backend.local.json.")
    if not config.workspace:
        errors.append("Remote workspace is not configured.")
    if not config.python:
        errors.append("Remote Python is not configured.")
    return errors


def validate_remote_path(path: str, *, under: str | None = None) -> str:
    candidate = PurePosixPath(path)
    if not candidate.is_absolute():
        raise ValueError(f"Remote path must be absolute: {path}")
    if any(part in {"", ".", ".."} for part in candidate.parts[1:]):
        raise ValueError(f"Remote path contains an unsafe path segment: {path}")
    if under:
        root = PurePosixPath(under)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Remote path {path} is outside {under}") from exc
    return candidate.as_posix()


def _run_text(argv: list[str], *, timeout: float = 10.0, cwd: Path | None = None) -> Tuple[bool, str, str]:
    try:
        completed = subprocess.run(argv, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout, check=False)  # noqa: S603
    except Exception as exc:  # noqa: BLE001
        return False, "", str(exc)
    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()


def local_git_ref(repo_root: Path) -> Dict[str, Any]:
    branch_ok, branch, branch_err = _run_text(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    commit_ok, commit, commit_err = _run_text(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    dirty_ok, dirty, dirty_err = _run_text(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"])
    ok = branch_ok and commit_ok and dirty_ok
    return {
        "ok": ok,
        "branch": branch if branch_ok else None,
        "commit": commit if commit_ok else None,
        "short_commit": (commit[:12] if commit_ok else None),
        "dirty_tracked": bool(dirty.strip()) if dirty_ok else None,
        "error": None if ok else " ".join(item for item in [branch_err, commit_err, dirty_err] if item),
    }


def compare_git_freshness(frontend: Dict[str, Any], backend: Dict[str, Any], expected_branch: str) -> Dict[str, Any]:
    frontend_ok = bool(frontend.get("ok"))
    backend_ok = bool(backend.get("ok"))
    if not frontend_ok or not backend_ok:
        status = "probe_failed"
    elif frontend.get("branch") != expected_branch or backend.get("branch") != expected_branch:
        status = "mismatch"
    elif frontend.get("commit") != backend.get("commit"):
        status = "mismatch"
    else:
        status = "match"
    dirty_frontend = bool(frontend.get("dirty_tracked"))
    can_launch = status == "match"
    message = "Frontend and remote backend are aligned."
    if status == "probe_failed":
        message = "Could not verify frontend/backend git freshness."
    elif status == "mismatch":
        message = "Frontend and remote backend branch or commit differ."
    elif dirty_frontend:
        message = "Frontend and backend commits match. Fast-forward update is disabled while tracked local modifications exist."
    return {
        "status": status,
        "can_launch": can_launch,
        "message": message,
        "expected_branch": expected_branch,
        "frontend": frontend,
        "backend": backend,
    }


class RemoteSshClient:
    def __init__(self, config: RemoteSshConfig) -> None:
        self.config = config
        validate_remote_path(config.repo)
        validate_remote_path(config.workspace)
        validate_remote_path(config.python, under=config.repo)

    def ssh_argv(self, remote_command: str) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.config.connect_timeout_s}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            self.config.target,
            "/bin/bash -lc " + shlex.quote(remote_command),
        ]

    def runner_shell_command(self, args: Iterable[str]) -> str:
        argv = [self.config.python, "-m", "webui.service.remote_runner", *list(args)]
        env_parts = {
            "PYTHONUNBUFFERED": "1",
            "RAPID_WEBUI_REMOTE_WORKSPACE": self.config.workspace,
            "RAPID_WEBUI_WORKSPACE_ROOT": self.config.workspace,
        }
        env_parts.update(self.config.remote_env)
        env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_parts.items())
        return "cd " + shlex.quote(self.config.repo) + " && " + env_text + " " + " ".join(shlex.quote(part) for part in argv)

    def runner_ssh_argv(self, args: Iterable[str]) -> list[str]:
        return self.ssh_argv(self.runner_shell_command(args))

    def run_json(self, args: Iterable[str], *, timeout: float = 20.0) -> Dict[str, Any]:
        argv = self.runner_ssh_argv(args)
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)  # noqa: S603
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "remote command failed").strip())
        for line in reversed(completed.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise RuntimeError("remote command did not return JSON")

    def popen_runner(self, args: Iterable[str]) -> subprocess.Popen:
        return subprocess.Popen(self.runner_ssh_argv(args), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603

    def _run_transfer(self, argv: list[str], *, timeout: float | None = None) -> None:
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)  # noqa: S603
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "transfer failed").strip())

    def ensure_remote_dir(self, remote_dir: str) -> None:
        remote_dir = validate_remote_path(remote_dir, under=self.config.workspace)
        self._run_transfer(self.ssh_argv("mkdir -p " + shlex.quote(remote_dir)))

    def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        remote_dir = validate_remote_path(remote_dir, under=self.config.workspace)
        local_dir = local_dir.resolve()
        self.ensure_remote_dir(remote_dir)
        if shutil.which("rsync"):
            self._run_transfer(["rsync", "-a", "--delete", f"{local_dir}/", f"{self.config.target}:{remote_dir}/"])
            return
        parent = str(PurePosixPath(remote_dir).parent)
        self._run_transfer(["scp", "-r", str(local_dir), f"{self.config.target}:{parent}/"])

    def pull_dir(self, remote_dir: str, local_dir: Path, *, delete: bool = False) -> None:
        remote_dir = validate_remote_path(remote_dir, under=self.config.workspace)
        local_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync"):
            argv = ["rsync", "-a"]
            if delete:
                argv.append("--delete")
            argv.extend([f"{self.config.target}:{remote_dir}/", f"{local_dir.resolve()}/"])
            self._run_transfer(argv)
            return
        parent = local_dir.parent
        self._run_transfer(["scp", "-r", f"{self.config.target}:{remote_dir}", str(parent)])


class RemoteEventState:
    def __init__(self, last_seq: int = 0) -> None:
        self.last_seq = int(last_seq or 0)
        self.malformed_count = 0
        self.duplicate_count = 0

    def apply_line(self, line: str) -> Dict[str, Any] | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self.malformed_count += 1
            return None
        if not isinstance(event, dict):
            self.malformed_count += 1
            return None
        if event.get("type") == "heartbeat" and "seq" not in event:
            return event
        try:
            seq = int(event.get("seq"))
        except (TypeError, ValueError):
            self.malformed_count += 1
            return None
        if seq <= self.last_seq:
            self.duplicate_count += 1
            return None
        self.last_seq = seq
        return event


class RemoteTelemetryMonitor:
    def __init__(self, client: RemoteSshClient, *, cache_s: float = 5.0) -> None:
        self.client = client
        self.cache_s = cache_s
        self._latest: Dict[str, Any] | None = None
        self._latest_monotonic = 0.0
        self._last_error: str | None = None

    def get(self) -> Dict[str, Any]:
        now = time.monotonic()
        if self._latest is None or now - self._latest_monotonic > self.cache_s:
            try:
                payload = self.client.run_json(["telemetry"], timeout=max(5.0, float(self.client.config.connect_timeout_s) + 5.0))
                self._latest = payload
                self._latest_monotonic = now
                self._last_error = None
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
        latest = dict(self._latest or {})
        stale = not latest or self._last_error is not None
        return {
            "available_ram_gb": latest.get("available_ram_gb"),
            "used_percent": latest.get("used_percent"),
            "cpu_percent": latest.get("cpu_percent"),
            "cpu_count": latest.get("cpu_count"),
            "source": "remote",
            "host": self.client.config.host,
            "stale": stale,
            "updated_at": latest.get("updated_at"),
            "error": self._last_error,
        }


StatusWriter = Callable[..., None]


class RemoteSshExecutor:
    def __init__(self, config: RemoteSshConfig) -> None:
        if not config.enabled:
            raise ValueError("RemoteSshExecutor requires RAPID_WEBUI_EXECUTION_MODE=remote_ssh")
        self.config = config
        self.client = RemoteSshClient(config)

    def probe(self) -> Dict[str, Any]:
        payload = self.client.run_json(["probe", "--json"], timeout=20.0)
        payload.setdefault("ok", True)
        return payload

    def freshness(self, local_root: Path) -> Dict[str, Any]:
        frontend = local_git_ref(local_root)
        try:
            probe = self.probe()
            backend = {
                "ok": bool(probe.get("ok")),
                "branch": probe.get("branch"),
                "commit": probe.get("commit"),
                "short_commit": (str(probe.get("commit") or "")[:12] or None),
                "dirty_tracked": probe.get("dirty_tracked"),
                "error": probe.get("error"),
            }
        except Exception as exc:  # noqa: BLE001
            backend = {"ok": False, "branch": None, "commit": None, "short_commit": None, "dirty_tracked": None, "error": str(exc)}
        return compare_git_freshness(frontend, backend, self.config.branch)

    def update_both(self, local_root: Path) -> Dict[str, Any]:
        before = local_git_ref(local_root)
        if before.get("dirty_tracked"):
            return {"ok": False, "message": "Frontend has tracked local changes; refusing fast-forward update.", "freshness": self.freshness(local_root)}
        fetch = subprocess.run(["git", "-C", str(local_root), "fetch", "origin", self.config.branch], text=True, capture_output=True, check=False)  # noqa: S603
        if fetch.returncode != 0:
            return {"ok": False, "message": (fetch.stderr or fetch.stdout or "frontend git fetch failed").strip(), "freshness": self.freshness(local_root)}
        pull = subprocess.run(["git", "-C", str(local_root), "pull", "--ff-only", "origin", self.config.branch], text=True, capture_output=True, check=False)  # noqa: S603
        if pull.returncode != 0:
            return {"ok": False, "message": (pull.stderr or pull.stdout or "frontend git pull failed").strip(), "freshness": self.freshness(local_root)}
        try:
            backend = self.client.run_json(["update", "--branch", self.config.branch], timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"Frontend updated, backend update failed: {exc}", "restart_required": True, "freshness": self.freshness(local_root)}
        fresh = self.freshness(local_root)
        return {"ok": bool(backend.get("ok")) and fresh.get("status") == "match", "message": backend.get("message") or "Updated frontend and backend.", "restart_required": True, "freshness": fresh}

    def remote_job_root(self, job_kind: str, job_id: str) -> str:
        folder = "sweeps" if job_kind == "sweep" else "runs"
        return validate_remote_path(f"{self.config.workspace}/{folder}/{job_id}", under=self.config.workspace)

    def write_job_bundle(self, job_root: Path, payload: Dict[str, Any], preview: Dict[str, Any], local_root: Path) -> None:
        bundle = {
            "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
            "execution_mode": "remote_ssh",
            "expected_branch": self.config.branch,
            "frontend_git": local_git_ref(local_root),
            "payload": payload,
            "preview": preview,
        }
        (job_root / "remote_bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True))

    def cancel(self, remote_job_root: str) -> Tuple[bool, str]:
        try:
            payload = self.client.run_json(["cancel", "--job-root", remote_job_root], timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return bool(payload.get("ok")), str(payload.get("message") or "Cancellation requested.")

    def run_job(
        self,
        *,
        job_root: Path,
        job_kind: str,
        payload: Dict[str, Any],
        preview: Dict[str, Any],
        local_root: Path,
        active_record_path: Path,
        status_writer: StatusWriter,
        cancel_event: threading.Event | None = None,
    ) -> Dict[str, Any]:
        remote_root = self.remote_job_root(job_kind, job_root.name)
        self.write_job_bundle(job_root, payload, preview, local_root)
        status_writer(job_root, status="running", remote_sync_state="syncing", remote_host=self.config.host, remote_job_root=remote_root)
        if cancel_event and cancel_event.is_set():
            summary = {"status": "cancelled", "error": "Cancelled before remote upload started."}
            (job_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
            if not (job_root / "artifacts.json").exists():
                (job_root / "artifacts.json").write_text(json.dumps({"job_id": job_root.name, "artifacts": []}, indent=2, sort_keys=True))
            status_writer(job_root, status="cancelled", remote_sync_state="synced", remote_host=self.config.host, remote_job_root=remote_root)
            return summary
        self.client.upload_dir(job_root, remote_root)
        status_writer(job_root, status="running", remote_sync_state="connecting", remote_host=self.config.host, remote_job_root=remote_root)
        if cancel_event and cancel_event.is_set():
            summary = {"status": "cancelled", "error": "Cancelled before remote supervisor started."}
            (job_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
            if not (job_root / "artifacts.json").exists():
                (job_root / "artifacts.json").write_text(json.dumps({"job_id": job_root.name, "artifacts": []}, indent=2, sort_keys=True))
            status_writer(job_root, status="cancelled", remote_sync_state="synced", remote_host=self.config.host, remote_job_root=remote_root)
            self.client.upload_dir(job_root, remote_root)
            return summary
        start_payload = self.client.run_json(["start", "--job-root", remote_root, "--detach"], timeout=20.0)
        active_record = {
            "host": self.config.host,
            "remote_path": remote_root,
            "pid": start_payload.get("pid"),
            "last_event_seq": 0,
            "local_job_root": str(job_root),
            "kind": job_kind,
            "job_id": job_root.name,
            "branch": self.config.branch,
            "commit": (local_git_ref(local_root).get("commit")),
            "sync_state": "streaming",
            "updated_at": time.time(),
        }
        active_record_path.write_text(json.dumps(active_record, indent=2, sort_keys=True))
        if cancel_event and cancel_event.is_set():
            status_writer(job_root, status="cancel_requested", remote_sync_state="streaming", remote_host=self.config.host, remote_job_root=remote_root)
            self.cancel(remote_root)
        return self._stream_and_sync(job_root=job_root, remote_root=remote_root, active_record_path=active_record_path, status_writer=status_writer, start_seq=0)

    def resume_job(self, *, active_record: Dict[str, Any], active_record_path: Path, status_writer: StatusWriter) -> Dict[str, Any]:
        job_root = Path(str(active_record["local_job_root"]))
        remote_root = str(active_record["remote_path"])
        start_seq = int(active_record.get("last_event_seq") or 0)
        return self._stream_and_sync(job_root=job_root, remote_root=remote_root, active_record_path=active_record_path, status_writer=status_writer, start_seq=start_seq)

    def _stream_and_sync(self, *, job_root: Path, remote_root: str, active_record_path: Path, status_writer: StatusWriter, start_seq: int) -> Dict[str, Any]:
        state = RemoteEventState(last_seq=start_seq)
        terminal = False
        last_error: str | None = None
        attempts = 0
        while not terminal and attempts < 4:
            sync_state = "streaming" if attempts == 0 else "reconnecting"
            status_writer(job_root, remote_sync_state=sync_state, remote_host=self.config.host, remote_job_root=remote_root, last_event_seq=state.last_seq)
            proc: subprocess.Popen | None = None
            try:
                proc = self.client.popen_runner(["stream", "--job-root", remote_root, "--from-seq", str(state.last_seq), "--heartbeat-interval", str(self.config.heartbeat_interval_s)])
                assert proc.stdout is not None
                for line in proc.stdout:
                    event = state.apply_line(line.strip())
                    if not event:
                        continue
                    terminal = self._handle_event(event, job_root, remote_root, status_writer)
                    self._update_active_record(active_record_path, last_event_seq=state.last_seq, sync_state="streaming")
                    if terminal:
                        break
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=3)
                if terminal:
                    break
                last_error = (proc.stderr.read().strip() if proc.stderr else "") or f"remote stream exited with {proc.returncode}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if proc and proc.poll() is None:
                    proc.terminate()
            attempts += 1
            if not terminal:
                status_writer(job_root, remote_sync_state="reconnecting", remote_error=last_error, last_event_seq=state.last_seq)
                time.sleep(min(5.0, 1.0 + attempts))
        status_writer(job_root, remote_sync_state="syncing", last_event_seq=state.last_seq)
        self.client.pull_dir(remote_root, job_root, delete=False)
        self._verify_final_sync(job_root)
        status = self._load_json(job_root / "status.json")
        summary = self._load_json(job_root / "summary.json")
        if not terminal and str(status.get("status")) not in REMOTE_TERMINAL_STATUSES:
            raise RuntimeError(last_error or "Remote stream ended before a terminal status was observed.")
        status_writer(job_root, status=status.get("status"), remote_sync_state="synced", last_event_seq=state.last_seq)
        active_record_path.unlink(missing_ok=True)
        return summary

    def _handle_event(self, event: Dict[str, Any], job_root: Path, remote_root: str, status_writer: StatusWriter) -> bool:
        event_type = str(event.get("type") or "")
        status_record = event.get("status_record") if isinstance(event.get("status_record"), dict) else {}
        status = event.get("status") or status_record.get("status")
        progress_total = event.get("progress_total", status_record.get("progress_total"))
        progress_completed = event.get("progress_completed", status_record.get("progress_completed"))
        if event_type in {"case_completed", "artifact"}:
            status_writer(job_root, remote_sync_state="syncing", status=status, progress_total=progress_total, progress_completed=progress_completed)
            self.client.pull_dir(remote_root, job_root, delete=False)
        elif event_type == "heartbeat":
            status_writer(job_root, status=status, progress_total=progress_total, progress_completed=progress_completed, remote_sync_state="streaming")
        else:
            status_writer(job_root, status=status, progress_total=progress_total, progress_completed=progress_completed, remote_sync_state="streaming")
        return event_type in {"completed", "failed", "cancelled"} or str(status) in REMOTE_TERMINAL_STATUSES

    def _update_active_record(self, active_record_path: Path, **updates: Any) -> None:
        try:
            record = json.loads(active_record_path.read_text()) if active_record_path.exists() else {}
        except json.JSONDecodeError:
            record = {}
        record.update(updates)
        record["updated_at"] = time.time()
        active_record_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    def _verify_final_sync(self, job_root: Path) -> None:
        required = ["status.json", "summary.json", "request.json", "artifacts.json"]
        missing = [name for name in required if not (job_root / name).exists()]
        if missing:
            raise RuntimeError(f"Remote final sync missing expected file(s): {', '.join(missing)}")

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
