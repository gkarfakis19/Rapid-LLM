# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a DECLARED SET of mapped-path sweeps and roll them up into a MANIFEST.

WHY THIS EXISTS AND WHAT IT IS NOT. ``tools/fws_qif_dse.py`` sweeps ONE
(hardware, model) pair and writes that pair's report. A report set — "the four
models that are the default run set and the report" — was until now a shell
loop somebody typed, which means the set itself was never written down, never
checked, and never readable by a page. This tool makes the SET a checked-in
artifact:

* a BUNDLE (``configs/run-bundles/<name>.yaml``) is the strict-keyed list of
  members. Strict-keyed matters here more than usual: a typo'd key in a run
  list is a member that silently never ran, or ran against the wrong config,
  so every key is named and anything else is REFUSED BY NAME.
* a MANIFEST (``docs/qif/dse/<bundle>.json``) is what this tool writes: one
  row per member, each carrying the ``run_id`` READ BACK OUT of that member's
  own ``dse_report.json``. It is the file the atlas selector reads to know
  which sweeps exist and which generation each row is.

THE MANIFEST IS NOT A SECOND ACCOUNTING. It stores no metric this tool
computed; every field is copied from the member's report, and the headline is
rendered by ``fws_qif_dse.headline_text`` — the same function the report's own
markdown uses — so a number can never disagree between the two documents.

THE MANIFEST ALWAYS COVERS THE WHOLE BUNDLE. ``--only`` restricts what this
invocation RE-RUNS, never what the manifest describes: a manifest holding one
row because somebody re-ran one member would be a truncated picture of the set
that the atlas would then draw. A member with no report on disk is refused by
name, with the command that would produce it.
"""

import argparse
import concurrent.futures
import datetime
import json
import multiprocessing
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from fws_qif_dse import headline_text, run_sweep  # noqa: E402

#: The manifest schema this tool writes. The atlas selector keys off it.
MANIFEST_SCHEMA = "fws_run_bundle/1"

#: Top-level keys a bundle may declare. Anything else is refused by name.
BUNDLE_KEYS = ("bundle", "label", "manifest", "members")
#: Keys a bundle MEMBER may declare, and every one of them is REQUIRED: the
#: bundle is the record of what ran, so no field may come from a default.
MEMBER_KEYS = (
    "name",
    "model_id",
    "label",
    "model_config",
    "hardware_config",
    "mode",
    "output_dir",
)
#: The modes fws_qif_dse evaluates. A bundle naming another one is refused
#: here rather than several minutes later inside a worker.
MODES = ("LLM", "VIT")


class BundleError(ValueError):
    """A bundle this tool refuses to run, named at the key that broke it."""


def _reject_unknown_keys(context, mapping, allowed):
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        raise BundleError(
            "%s declares unknown key(s) %s. A bundle is STRICT-KEYED: legal keys are "
            "%s. An ignored key in a run list is a member that silently never ran (or "
            "ran against the wrong config), so it is refused by name instead."
            % (context, ", ".join(repr(key) for key in sorted(unknown)),
               ", ".join(repr(key) for key in allowed))
        )


def _require_keys(context, mapping, required):
    missing = [key for key in required if key not in mapping]
    if missing:
        raise BundleError(
            "%s is missing required key(s) %s. Every member declares all of %s: the "
            "bundle is the record of WHAT RAN, and a defaulted field would make that "
            "record depend on this tool's version rather than on the file."
            % (context, ", ".join(repr(key) for key in missing),
               ", ".join(repr(key) for key in MEMBER_KEYS))
        )


def load_bundle(path):
    """Parse and validate a bundle YAML; return the checked dict."""
    path = Path(path)
    if not path.exists():
        raise BundleError("bundle %s does not exist." % path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise BundleError("%s is not a YAML mapping." % path)
    _reject_unknown_keys(str(path), raw, BUNDLE_KEYS)
    for key in ("bundle", "manifest", "members"):
        if key not in raw:
            raise BundleError("%s declares no %r key." % (path, key))
    name = raw["bundle"]
    if not isinstance(name, str) or not name.strip():
        raise BundleError("%s: 'bundle' must be the set's name; got %r." % (path, name))
    members = raw["members"]
    if not isinstance(members, list) or not members:
        raise BundleError(
            "%s: 'members' must be a non-empty list of members; got %r."
            % (path, type(members).__name__)
        )
    seen = {}
    checked = []
    for index, member in enumerate(members):
        context = "%s: members[%d]" % (path, index)
        if not isinstance(member, dict):
            raise BundleError("%s is not a mapping." % context)
        _reject_unknown_keys(context, member, MEMBER_KEYS)
        _require_keys(context, member, MEMBER_KEYS)
        member_name = str(member["name"])
        context = "%s: member %r" % (path, member_name)
        if member_name in seen:
            raise BundleError(
                "%s is declared twice (members[%d] and members[%d]). Two members under "
                "one name would write one output directory twice and the manifest "
                "would carry whichever finished last." % (context, seen[member_name], index)
            )
        seen[member_name] = index
        mode = str(member["mode"])
        if mode not in MODES:
            raise BundleError(
                "%s declares mode = %r; fws_qif_dse evaluates %s."
                % (context, mode, " or ".join(MODES))
            )
        checked.append({key: member[key] for key in MEMBER_KEYS})
    return {
        "bundle": name.strip(),
        "label": str(raw.get("label", "")),
        "manifest": str(raw["manifest"]),
        "members": checked,
        "path": str(path),
    }


def select_members(bundle, only):
    """The members ``--only`` asks for, refusing an unknown name BY NAME."""
    if not only:
        return list(bundle["members"])
    known = {member["name"]: member for member in bundle["members"]}
    picked = []
    for name in only:
        if name not in known:
            raise BundleError(
                "--only %r names no member of bundle %r. Its members are: %s."
                % (name, bundle["bundle"], ", ".join(sorted(known)))
            )
        if known[name] not in picked:
            picked.append(known[name])
    return picked


def _resolve(root, relative):
    """Resolve an OUTPUT path against the output root."""
    path = Path(relative)
    return path if path.is_absolute() else Path(root) / path


def _resolve_input(relative):
    """Resolve an INPUT config against the repo, never against --output-root.

    ``--output-root`` redirects where artifacts LAND (a scratch tree, so a
    rehearsal cannot overwrite a checked-in report). The configs a member names
    are checked-in inputs and are always read from the repo, so the rehearsal
    prices the SAME machine the real run would.
    """
    path = Path(relative)
    return path if path.is_absolute() else REPO_ROOT / path


def run_member(member, root, *, quiet=False, atlas_points=True):
    """Run ONE member through fws_qif_dse.run_sweep; return (name, exit_code)."""
    output_dir = _resolve(root, member["output_dir"])
    exit_code, _payload = run_sweep(
        str(_resolve_input(member["hardware_config"])),
        str(_resolve_input(member["model_config"])),
        mode=str(member["mode"]),
        model_id=str(member["model_id"]),
        output_dir=str(output_dir),
        quiet=quiet,
        atlas_points=atlas_points,
    )
    return member["name"], int(exit_code)


def member_row(member, root):
    """One manifest row, every field READ BACK from the member's own report."""
    output_dir = _resolve(root, member["output_dir"])
    report_path = output_dir / "dse_report.json"
    if not report_path.exists():
        raise BundleError(
            "member %r has no %s. The manifest describes the WHOLE bundle, so a "
            "member with no report on disk is refused rather than dropped from the "
            "row set (a truncated manifest is a truncated atlas). Run it first:\n"
            "  python tools/fws_run_bundle.py --bundle <bundle> --only %s"
            % (member["name"], report_path, member["name"])
        )
    report = json.loads(report_path.read_text())
    selected = report.get("selected")
    return {
        "name": member["name"],
        "model_id": report.get("model_id") or member["model_id"],
        # The picker shows this. The SWEEP label is report prose, never a
        # picker name; the member's own short label is the only source.
        "label": str(member["label"]),
        "mode": report.get("mode") or member["mode"],
        "dir": str(member["output_dir"]),
        "run_id": report.get("run_id"),
        "selected_id": report.get("selected_id"),
        "headline": None if selected is None else headline_text(selected),
    }


def build_manifest(bundle, root, *, generated_at=None):
    """The bundle MANIFEST: one row per member, in the bundle's own order."""
    return {
        "schema": MANIFEST_SCHEMA,
        "bundle": bundle["bundle"],
        "label": bundle.get("label", ""),
        "generated_at": generated_at
        or datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "members": [member_row(member, root) for member in bundle["members"]],
    }


def write_manifest(bundle, root, *, generated_at=None):
    manifest = build_manifest(bundle, root, generated_at=generated_at)
    path = _resolve(root, bundle["manifest"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path, manifest


def run_bundle(bundle_path, *, only=(), jobs=2, root=None, run=True, quiet=False,
               atlas_points=True):
    """Run the selected members, then write the whole bundle's manifest.

    Returns ``(exit_code, manifest_path, manifest)``. The exit code is the
    WORST member's: a set in which one sweep found nothing feasible has not
    succeeded, whatever the other three did.
    """
    root = Path(root or REPO_ROOT)
    bundle = load_bundle(bundle_path)
    members = select_members(bundle, only)
    jobs = max(1, int(jobs))

    worst = 0
    if run:
        print(
            "[bundle] %s: running %d of %d member(s) with --jobs %d"
            % (bundle["bundle"], len(members), len(bundle["members"]), jobs)
        )
        results = {}
        if jobs == 1 or len(members) == 1:
            for member in members:
                name, code = run_member(member, root, quiet=quiet,
                                        atlas_points=atlas_points)
                results[name] = code
        else:
            # PROCESSES, not threads. A member's wall clock is spent inside the
            # evaluator, which is pure Python and holds the GIL the whole way
            # (nothing under fws_mapping -> fws_build -> fws_eval releases it or
            # parallelises internally), so a thread pool would report --jobs 2
            # and deliver one job. Each member writes its own output directory,
            # so the workers never touch the same file, and the manifest is
            # assembled afterwards in THIS process by reading what they wrote.
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, mp_context=multiprocessing.get_context("fork")
            ) as pool:
                futures = {
                    pool.submit(run_member, member, root, quiet=quiet,
                                atlas_points=atlas_points): member
                    for member in members
                }
                for future in concurrent.futures.as_completed(futures):
                    name, code = future.result()
                    results[name] = code
        for name in (member["name"] for member in members):
            code = results.get(name, 0)
            print("[bundle] %s: exit %d" % (name, code))
            worst = max(worst, code)

    path, manifest = write_manifest(bundle, root)
    print("[bundle] manifest %s (%d member row(s))" % (path, len(manifest["members"])))
    for row in manifest["members"]:
        print(
            "[bundle]   %-20s run %s  selected %s  %s"
            % (row["name"], row["run_id"], row["selected_id"], row["headline"])
        )
    return worst, path, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fws_run_bundle",
        description=(
            "Run a declared bundle of mapped-path sweeps (fws_qif_dse.run_sweep) and "
            "write the bundle manifest the atlas selector reads."
        ),
    )
    parser.add_argument("--bundle", required=True, help="Bundle YAML (configs/run-bundles/*.yaml).")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Re-run ONLY this member (repeatable). The manifest still covers the whole "
            "bundle; an unknown name is refused by name."
        ),
    )
    parser.add_argument("--jobs", type=int, default=2, help="Members to run in parallel (default 2).")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Write every artifact (member directories and the manifest) under this root "
            "instead of the repo. The configs a member names are still read from the "
            "repo, so a scratch rehearsal prices the same machine the real run would."
        ),
    )
    parser.add_argument(
        "--no-run",
        dest="run",
        action="store_false",
        help="Do not sweep anything; rebuild the manifest from the reports already on disk.",
    )
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to every member sweep.")
    parser.add_argument(
        "--no-atlas-points",
        dest="atlas_points",
        action="store_false",
        help="Pass --no-atlas-points to every member sweep.",
    )
    parser.set_defaults(run=True, atlas_points=True)
    args = parser.parse_args(argv)

    try:
        exit_code, _path, _manifest = run_bundle(
            args.bundle,
            only=tuple(args.only or ()),
            jobs=args.jobs,
            root=args.output_root,
            run=args.run,
            quiet=args.quiet,
            atlas_points=args.atlas_points,
        )
    except (BundleError, ValueError) as exc:
        print("[bundle] error: %s" % exc)
        return 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
