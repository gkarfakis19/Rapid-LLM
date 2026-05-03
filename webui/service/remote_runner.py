from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import psutil

from webui.service.remote import RUNNER_PROTOCOL_VERSION, REMOTE_TERMINAL_STATUSES


BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def remote_workspace() -> Path:
    return Path(os.environ.get("RAPID_WEBUI_REMOTE_WORKSPACE", repo_root() / "webui" / "workspace")).expanduser().resolve()


def json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_git(args: Iterable[str]) -> tuple[bool, str, str]:
    completed = subprocess.run(["git", "-C", str(repo_root()), *list(args)], text=True, capture_output=True, check=False)  # noqa: S603
    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()


def git_probe() -> Dict[str, Any]:
    branch_ok, branch, branch_err = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit_ok, commit, commit_err = run_git(["rev-parse", "HEAD"])
    dirty_ok, dirty, dirty_err = run_git(["status", "--porcelain", "--untracked-files=no"])
    ok = branch_ok and commit_ok and dirty_ok
    return {
        "ok": ok,
        "branch": branch if branch_ok else None,
        "commit": commit if commit_ok else None,
        "dirty_tracked": bool(dirty.strip()) if dirty_ok else None,
        "error": None if ok else " ".join(item for item in [branch_err, commit_err, dirty_err] if item),
    }


def validate_job_root(raw_path: str) -> Path:
    workspace = remote_workspace()
    path = Path(raw_path).expanduser().resolve()
    if not path.is_absolute():
        raise ValueError(f"job-root must be absolute: {raw_path}")
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"job-root must be under {workspace}") from exc
    return path


def next_event_seq(job_root: Path) -> int:
    seq_path = job_root / "event_seq.txt"
    try:
        seq = int(seq_path.read_text().strip()) + 1 if seq_path.exists() else 1
    except ValueError:
        seq = 1
    seq_path.write_text(str(seq))
    return seq


def append_event(job_root: Path, event_type: str, **payload: Any) -> Dict[str, Any]:
    event = {"seq": next_event_seq(job_root), "type": event_type, "created_at": utc_now(), **payload}
    path = job_root / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
    return event


def write_terminal_status(job_root: Path, status_name: str, message: str) -> Dict[str, Any]:
    status = json_load(job_root / "status.json")
    status.update({"status": status_name, "updated_at": utc_now()})
    if message:
        status["error"] = message
    status.setdefault("created_at", status["updated_at"])
    json_dump(job_root / "status.json", status)
    json_dump(job_root / "summary.json", {"status": status_name, "error": message})
    if not (job_root / "artifacts.json").exists():
        json_dump(job_root / "artifacts.json", {"job_id": job_root.name, "artifacts": []})
    return status


def command_probe(_: argparse.Namespace) -> int:
    workspace = remote_workspace()
    dependency_errors: list[str] = []
    try:
        import dash  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        dependency_errors.append(f"dash: {exc}")
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        probe_file = workspace / ".remote_probe_write_test"
        probe_file.write_text(utc_now())
        probe_file.unlink(missing_ok=True)
        workspace_writable = True
    except Exception:  # noqa: BLE001
        workspace_writable = False
    git = git_probe()
    print(
        json.dumps(
            {
                "ok": bool(git.get("ok")) and workspace_writable and not dependency_errors,
                "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
                "repo": str(repo_root()),
                "workspace": str(workspace),
                "python": sys.executable,
                "workspace_writable": workspace_writable,
                "dependencies_ok": not dependency_errors,
                "dependency_errors": dependency_errors,
                **git,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def telemetry_payload() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "type": "telemetry",
        "available_ram_gb": round(vm.available / (1024**3), 1),
        "used_percent": round(vm.percent, 1),
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "cpu_count": psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        "updated_at": utc_now(),
    }


def command_telemetry(args: argparse.Namespace) -> int:
    interval = max(1.0, float(args.interval))
    if not args.stream:
        print(json.dumps(telemetry_payload(), sort_keys=True), flush=True)
        return 0
    while True:
        print(json.dumps(telemetry_payload(), sort_keys=True), flush=True)
        time.sleep(interval)


class RemoteRunManager:
    def __init__(self, job_root: Path) -> None:
        os.environ.setdefault("RAPID_WEBUI_WORKSPACE_ROOT", str(remote_workspace()))
        from webui.service import core

        core.PYTHON_BIN = Path(sys.executable)
        self.core = core
        self.manager = core.RunManager()
        self.job_root = job_root
        self._patch_manager()

    def _patch_manager(self) -> None:
        original_write_status = self.manager._write_status
        original_execute_worker_case = self.manager._execute_worker_case

        def write_status(job_root: Path, **kwargs: Any) -> None:
            original_write_status(job_root, **kwargs)
            status_record = json_load(job_root / "status.json")
            append_event(
                job_root,
                "status",
                status=status_record.get("status"),
                progress_total=status_record.get("progress_total"),
                progress_completed=status_record.get("progress_completed"),
                status_record=status_record,
            )

        def execute_worker_case(job_root: Path, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            result = original_execute_worker_case(job_root, *args, **kwargs)
            event_type = "case_completed" if result.get("status") == "completed" else "case_failed"
            status_record = json_load(job_root / "status.json")
            append_event(
                job_root,
                event_type,
                case_id=result.get("case_id"),
                top_case_id=result.get("top_case_id"),
                status=result.get("status"),
                progress_total=status_record.get("progress_total"),
                progress_completed=status_record.get("progress_completed"),
                status_record=status_record,
            )
            return result

        self.manager._write_status = write_status  # type: ignore[method-assign]
        self.manager._execute_worker_case = execute_worker_case  # type: ignore[method-assign]

    def run(self) -> None:
        bundle = json_load(self.job_root / "remote_bundle.json")
        payload = bundle.get("payload")
        preview = bundle.get("preview")
        if not isinstance(payload, dict) or not isinstance(preview, dict):
            raise RuntimeError("remote_bundle.json is missing payload or preview")
        stop_event = threading.Event()
        watcher = threading.Thread(target=self._watch_cancel, args=(stop_event,), daemon=True)
        watcher.start()
        try:
            append_event(self.job_root, "started", status="running", status_record=json_load(self.job_root / "status.json"))
            self.manager._run_job_thread(self.job_root, payload, preview)
            status = json_load(self.job_root / "status.json")
            final_type = "cancelled" if status.get("status") == "cancelled" else "failed" if status.get("status") == "failed" else "completed"
            append_event(self.job_root, final_type, status=status.get("status"), status_record=status)
        finally:
            stop_event.set()

    def _watch_cancel(self, stop_event: threading.Event) -> None:
        cancel_path = self.job_root / "cancel.requested"
        while not stop_event.is_set():
            if cancel_path.exists():
                self.manager._cancel_event.set()
                status = json_load(self.job_root / "status.json")
                if str(status.get("status")) not in REMOTE_TERMINAL_STATUSES:
                    self.manager._write_status(self.job_root, status="cancel_requested", error="Cancellation requested.")
                with self.manager._lock:
                    processes = list(self.manager._processes.values())
                for active in processes:
                    try:
                        if active.process.poll() is None:
                            active.process.terminate()
                    except Exception:
                        pass
                append_event(self.job_root, "cancel_requested", status="cancel_requested", status_record=json_load(self.job_root / "status.json"))
                return
            time.sleep(0.5)


def command_start(args: argparse.Namespace) -> int:
    job_root = validate_job_root(args.job_root)
    job_root.mkdir(parents=True, exist_ok=True)
    if not args.detach:
        return command_supervise(args)
    stdout_path = job_root / "remote_supervisor.stdout.log"
    stderr_path = job_root / "remote_supervisor.stderr.log"
    cmd = [sys.executable, "-m", "webui.service.remote_runner", "supervise", "--job-root", str(job_root)]
    with stdout_path.open("a") as stdout_handle, stderr_path.open("a") as stderr_handle:
        process = subprocess.Popen(cmd, cwd=str(repo_root()), stdout=stdout_handle, stderr=stderr_handle, start_new_session=True, close_fds=True)  # noqa: S603
    json_dump(job_root / "remote_start.json", {"pid": process.pid, "started_at": utc_now(), "runner_protocol_version": RUNNER_PROTOCOL_VERSION})
    (job_root / "supervisor.pid").write_text(str(process.pid))
    print(json.dumps({"ok": True, "pid": process.pid, "job_root": str(job_root), "message": "Remote supervisor started."}, sort_keys=True), flush=True)
    return 0


def command_supervise(args: argparse.Namespace) -> int:
    job_root = validate_job_root(args.job_root)
    (job_root / "supervisor.pid").write_text(str(os.getpid()))
    try:
        RemoteRunManager(job_root).run()
        return 0
    except Exception as exc:  # noqa: BLE001
        status = json_load(job_root / "status.json")
        status.update({"status": "failed", "error": str(exc), "updated_at": utc_now()})
        status.setdefault("created_at", status["updated_at"])
        json_dump(job_root / "status.json", status)
        json_dump(job_root / "summary.json", {"status": "failed", "error": str(exc)})
        if not (job_root / "artifacts.json").exists():
            json_dump(job_root / "artifacts.json", {"job_id": job_root.name, "artifacts": []})
        append_event(job_root, "failed", status="failed", error=str(exc), status_record=status)
        return 1


def iter_events(path: Path, from_seq: int) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    events: list[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and int(event.get("seq") or 0) > from_seq:
                events.append(event)
    return events


def command_stream(args: argparse.Namespace) -> int:
    job_root = validate_job_root(args.job_root)
    from_seq = max(0, int(args.from_seq))
    interval = max(1.0, float(args.heartbeat_interval))
    emitted_seq = from_seq
    terminal_seen = False
    while True:
        for event in iter_events(job_root / "events.jsonl", emitted_seq):
            emitted_seq = max(emitted_seq, int(event.get("seq") or emitted_seq))
            print(json.dumps(event, sort_keys=True), flush=True)
            if event.get("type") in {"completed", "failed", "cancelled"}:
                terminal_seen = True
        status = json_load(job_root / "status.json")
        if str(status.get("status")) in REMOTE_TERMINAL_STATUSES:
            terminal_seen = True
        heartbeat = {
            "type": "heartbeat",
            "created_at": utc_now(),
            "status": status.get("status"),
            "progress_total": status.get("progress_total"),
            "progress_completed": status.get("progress_completed"),
            "status_record": status,
        }
        print(json.dumps(heartbeat, sort_keys=True), flush=True)
        if terminal_seen:
            return 0
        time.sleep(interval)


def command_cancel(args: argparse.Namespace) -> int:
    job_root = validate_job_root(args.job_root)
    json_dump(job_root / "cancel.requested", {"requested_at": utc_now()})
    status = json_load(job_root / "status.json")
    if str(status.get("status")) not in REMOTE_TERMINAL_STATUSES:
        status.update({"status": "cancel_requested", "updated_at": utc_now(), "error": "Cancellation requested."})
        status.setdefault("created_at", status["updated_at"])
        json_dump(job_root / "status.json", status)
        append_event(job_root, "cancel_requested", status="cancel_requested", status_record=status)
    pid = None
    pid_path = job_root / "supervisor.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = None
    deadline = time.time() + 2.0
    while time.time() < deadline:
        status = json_load(job_root / "status.json")
        if str(status.get("status")) in REMOTE_TERMINAL_STATUSES:
            break
        time.sleep(0.2)
    status = json_load(job_root / "status.json")
    if pid and str(status.get("status")) not in REMOTE_TERMINAL_STATUSES:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        status = write_terminal_status(job_root, "cancelled", "Cancelled by remote request.")
        append_event(job_root, "cancelled", status="cancelled", status_record=status)
    print(json.dumps({"ok": True, "message": "Cancellation requested.", "pid": pid}, sort_keys=True), flush=True)
    return 0


def command_update(args: argparse.Namespace) -> int:
    branch = str(args.branch or "")
    if not BRANCH_RE.fullmatch(branch):
        print(json.dumps({"ok": False, "message": "Unsafe branch name."}, sort_keys=True), flush=True)
        return 2
    dirty_ok, dirty, dirty_err = run_git(["status", "--porcelain", "--untracked-files=no"])
    if not dirty_ok:
        print(json.dumps({"ok": False, "message": dirty_err or "git status failed"}, sort_keys=True), flush=True)
        return 2
    if dirty.strip():
        print(json.dumps({"ok": False, "message": "Backend has tracked local changes; refusing fast-forward update."}, sort_keys=True), flush=True)
        return 2
    fetch = subprocess.run(["git", "-C", str(repo_root()), "fetch", "origin", branch], text=True, capture_output=True, check=False)  # noqa: S603
    if fetch.returncode != 0:
        print(json.dumps({"ok": False, "message": (fetch.stderr or fetch.stdout or "git fetch failed").strip()}, sort_keys=True), flush=True)
        return fetch.returncode
    pull = subprocess.run(["git", "-C", str(repo_root()), "pull", "--ff-only", "origin", branch], text=True, capture_output=True, check=False)  # noqa: S603
    if pull.returncode != 0:
        print(json.dumps({"ok": False, "message": (pull.stderr or pull.stdout or "git pull failed").strip()}, sort_keys=True), flush=True)
        return pull.returncode
    print(json.dumps({"ok": True, "message": "Backend fast-forward update completed.", **git_probe()}, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAPID-LLM Web UI remote job runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--json", action="store_true")
    probe.set_defaults(func=command_probe)
    telemetry = sub.add_parser("telemetry")
    telemetry.add_argument("--stream", action="store_true")
    telemetry.add_argument("--interval", default="1")
    telemetry.set_defaults(func=command_telemetry)
    start = sub.add_parser("start")
    start.add_argument("--job-root", required=True)
    start.add_argument("--detach", action="store_true")
    start.set_defaults(func=command_start)
    supervise = sub.add_parser("supervise")
    supervise.add_argument("--job-root", required=True)
    supervise.set_defaults(func=command_supervise)
    stream = sub.add_parser("stream")
    stream.add_argument("--job-root", required=True)
    stream.add_argument("--from-seq", default="0")
    stream.add_argument("--heartbeat-interval", default="1")
    stream.set_defaults(func=command_stream)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("--job-root", required=True)
    cancel.set_defaults(func=command_cancel)
    update = sub.add_parser("update")
    update.add_argument("--branch", required=True)
    update.set_defaults(func=command_update)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
