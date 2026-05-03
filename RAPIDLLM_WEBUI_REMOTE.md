# RAPID-LLM Web UI Remote SSH Backend

This branch adds an opt-in Web UI execution mode for weak webserver hosts. By default the Web UI still runs jobs locally:

```bash
RAPID_WEBUI_EXECUTION_MODE=local ./launch_webui.sh
```

Remote mode keeps the Dash app and local history mirror on the webserver, but executes jobs on a configured SSH backend over OpenSSH.

Real SSH hostnames, usernames, and filesystem paths must not be committed. Put deployment-specific values in the ignored local file `webui/remote_backend.local.json`:

```json
{
  "host": "ssh.example.com",
  "user": "optional-user",
  "repo": "/srv/rapid-llm",
  "branch": "remote_backend"
}
```

Then launch normally:

```bash
export RAPID_WEBUI_EXECUTION_MODE=remote_ssh
./launch_webui.sh
```

`RAPID_WEBUI_EXECUTION_MODE=remote_ssh` is still required to enable remote execution. Environment variables can override the local JSON file: `RAPID_WEBUI_REMOTE_HOST`, `RAPID_WEBUI_REMOTE_USER`, `RAPID_WEBUI_REMOTE_REPO`, `RAPID_WEBUI_REMOTE_BRANCH`, `RAPID_WEBUI_REMOTE_WORKSPACE`, and `RAPID_WEBUI_REMOTE_PYTHON`. `RAPID_WEBUI_REMOTE_USER` is optional; if unset, SSH config and the current username decide the account. `RAPID_WEBUI_REMOTE_PYTHON` defaults to `<remote_repo>/.venv/bin/python`.

## Design

- `local` remains the default mode. It has no remote badge, no SSH dependency, and uses the original `RunManager` path.
- `remote_ssh` creates the same local job root under `webui/workspace/runs` or `webui/workspace/sweeps`, writes `remote_bundle.json`, uploads the root to the configured SSH backend, starts a detached remote supervisor, streams JSON events, and pulls artifacts back into the local mirror.
- The remote supervisor is not a daemon. It persists `events.jsonl`, `status.json`, `summary.json`, logs, and artifacts in the remote job root. The local frontend can reconnect from the last event sequence.
- Local preview generation stays local. Existing history, detail views, downloads, plots, and tables read the local mirror.
- Remote telemetry is used only in `remote_ssh`. If the stream is stale, the UI shows a stale state instead of reporting local webserver stats.
- Remote launches are blocked unless the frontend and backend are both on `remote_backend`, at the same commit, and the frontend tracked worktree is clean.

## Remote Runner CLI

Run these from the remote checkout with the configured Python:

```bash
.venv/bin/python -m webui.service.remote_runner probe --json
.venv/bin/python -m webui.service.remote_runner telemetry --stream --interval 1
.venv/bin/python -m webui.service.remote_runner start --job-root /abs/path/to/job --detach
.venv/bin/python -m webui.service.remote_runner stream --job-root /abs/path/to/job --from-seq 0 --heartbeat-interval 1
.venv/bin/python -m webui.service.remote_runner cancel --job-root /abs/path/to/job
.venv/bin/python -m webui.service.remote_runner update --branch remote_backend
```

`update` is fast-forward-only. It refuses dirty tracked worktrees and never hard-resets or stashes.

## Operator Runbook

1. Create and push branch `remote_backend`.
2. On the remote execution host, clone or update the branch at the path configured in `webui/remote_backend.local.json`.
3. Create the remote virtualenv and verify:

```bash
cd /srv/rapid-llm
.venv/bin/python -m webui.service.remote_runner probe --json
```

4. From the webserver, verify SSH and telemetry:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 ssh.example.com \
  'cd /srv/rapid-llm && .venv/bin/python -m webui.service.remote_runner probe --json'
```

5. Launch the Web UI with `RAPID_WEBUI_EXECUTION_MODE=remote_ssh`.
6. Submit a tiny single-run job, open details from history, and verify artifacts download from the local mirror.
7. For reconnect validation, interrupt the local SSH stream or restart the Web UI while a remote job is running. The local backend should reconnect from `webui/workspace/remote_active_job.json`.

## Notes

- V1 assumes one active job at a time, matching the existing Web UI.
- Artifact transfer is always webserver-pulls-from-remote; no remote-to-local credentials are required.
- The admin-only “Update both” button runs `git fetch origin remote_backend` and `git pull --ff-only origin remote_backend` locally, then asks the remote runner to do the same on the configured SSH backend. If frontend code updates, restart the Web UI process.
