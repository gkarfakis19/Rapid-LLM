"""subset_v1 — the DECLARED DEFAULT RUN SET and the manifest it rolls up to.

``configs/run-bundles/<name>.yaml`` names the (model, hardware, mode) triples
that ARE the report, and ``tools/fws_run_bundle.py`` runs them and writes
``docs/qif/dse/<bundle>.json`` — the manifest the atlas selector reads.

What is under test here is the two things that can silently go wrong with a run
list: a key nobody validated (so a member never ran, or ran against the wrong
config), and a manifest assembled from something other than the members' own
reports (so a page shows a number no sweep produced). The manifest assembly is
exercised against FAKE reports on purpose — it must copy fields, not compute
them, and a test that had to run a real sweep would not be able to tell the
difference.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from tests.qif_artifacts import qif_artifact
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "configs" / "run-bundles"
SUBSET_V1 = BUNDLE_DIR / "subset_v1.yaml"
RUNNER_PATH = PROJECT_ROOT / "tools" / "fws_run_bundle.py"


def _load_runner():
    if str(PROJECT_ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    name = "fws_run_bundle_module"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered BEFORE exec: --jobs > 1 forks workers and pickles run_member by
    # reference, which needs the module reachable under the name it was loaded
    # with. Under the real CLI that name is __main__, which a fork inherits.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _write_bundle(tmp_path, raw, name="b.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _valid_raw(**overrides):
    raw = {
        "bundle": "t",
        "label": "a test bundle",
        "manifest": "out/t.json",
        "members": [
            {
                "name": "one",
                "model_id": "One",
                "label": "One",
                "model_config": "m.yaml",
                "hardware_config": "h.yaml",
                "mode": "LLM",
                "output_dir": "out/one_frontier",
            }
        ],
    }
    raw.update(overrides)
    return raw


# ---------------------------------------------------------------------------
# The shipped bundle
# ---------------------------------------------------------------------------


def test_subset_v1_is_the_four_declared_members():
    bundle = runner.load_bundle(SUBSET_V1)
    assert bundle["bundle"] == "subset_v1"
    assert bundle["manifest"] == "docs/qif/dse/subset_v1.json"
    assert [member["name"] for member in bundle["members"]] == [
        "mamba2_2p7b",
        "vit_large_story_64",
        "vit_huge_story_64",
        "granite_4_0_h_1b",
    ]


@pytest.mark.parametrize("index", range(4))
def test_every_subset_v1_member_points_at_files_that_exist(index):
    member = runner.load_bundle(SUBSET_V1)["members"][index]
    for key in ("model_config", "hardware_config"):
        path = PROJECT_ROOT / member[key]
        assert path.exists(), f"{member['name']}: {key} {path} does not exist"
    # The report directory is derived from the member's own name, so a renamed
    # member cannot keep writing into the old model's directory.
    assert member["output_dir"] == f"docs/qif/dse/{member['name']}_frontier"


@pytest.mark.parametrize("index", range(4))
def test_every_subset_v1_member_parses_through_the_dse_config_path(index):
    import config as config_module

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fws_qif_dse as dse

    member = runner.load_bundle(SUBSET_V1)["members"][index]
    hw_path = PROJECT_ROOT / member["hardware_config"]
    raw_hw = dse._load_yaml(hw_path)
    spec = dse.SweepSpec.from_raw(raw_hw, str(hw_path))
    assert spec.size > 0
    base = dse._hw_from_raw(dse._base_without_sweep(raw_hw))
    assert str(base.device_class).lower() == "fws_cim"
    assert base.mapping_config is not None
    dse._card_names(base, str(hw_path))
    model = config_module.parse_config(
        str(PROJECT_ROOT / member["model_config"]), member["mode"]
    )
    assert model.model_config is not None


def test_the_two_new_members_sweep_the_platform_box():
    # D35-REVISED REWRITE. OLD CLAIM: both NEW hardware configs sweep
    # adc_capacity_mux over [1, 4, 16, 64]. NEW CLAIM: the fold is DERIVED
    # (mux = Y/Z) and is refused as an axis by name; what the configs sweep is
    # the CONSTRAINT-NATIVE box — macro_z (converters) and macro_y (storage
    # tile) — on top of the template's mapping axes.
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import cim_analog_calc
    import fws_mapping
    import fws_qif_dse as dse

    for name in ("mamba2_2p7b", "vit_large_story_64"):
        member = next(
            m for m in runner.load_bundle(SUBSET_V1)["members"] if m["name"] == name
        )
        hw_path = PROJECT_ROOT / member["hardware_config"]
        spec = dse.SweepSpec.from_raw(dse._load_yaml(hw_path), str(hw_path))
        assert "adc_capacity_mux" not in spec.axes, name
        assert set(spec.axes["macro_z"]) <= set(cim_analog_calc.MACRO_Z_VALUES), name
        assert set(spec.axes["macro_y"]) <= set(cim_analog_calc.MACRO_Y_VALUES), name
        assert len(spec.axes["macro_z"]) >= 2 and len(spec.axes["macro_y"]) >= 2, name
        # D36/D38 REWRITE. OLD CLAIM: the platform box rides on top of the
        # template's MAPPING axes, so layers_per_chip is swept beside it. NEW
        # CLAIM: D38 retired the partition ("The partition is INTENT, not an
        # axis. The fill fraction is the axis") and D36 derived the chiplet
        # count, so the mapping half of the sweep is now exactly pp_util.
        assert "layers_per_chip" not in spec.axes, name
        assert set(spec.axes["pp_util"]) <= set(fws_mapping.PP_UTIL_VALUES), name
        assert set(spec.axes) == set(dse.AXIS_TARGETS), name


def test_the_pure_ssm_machine_declares_no_derived_fabric_geometry():
    # SA-GEOM: `cim.fabric.geometry: derived` reads R off the DECODE attention
    # call's row count, and a pure Mamba-2 stack has no attention call at any
    # layer — the key is refused by name. The config must therefore not carry
    # it, and the ViT machine (lockstep, prefill-only) must not either.
    for name in ("fws_cim_mamba2_frontier.yaml", "fws_cim_vit_large_frontier.yaml"):
        raw = yaml.safe_load((PROJECT_ROOT / "configs" / "hardware-config" / name).read_text())
        assert "geometry" not in (raw["cim"].get("fabric") or {}), name


# ---------------------------------------------------------------------------
# Strict-keyed refusals, BY NAME
# ---------------------------------------------------------------------------


def test_an_unknown_bundle_key_is_refused_by_name(tmp_path):
    path = _write_bundle(tmp_path, _valid_raw(memebers=[]))
    with pytest.raises(runner.BundleError, match="'memebers'"):
        runner.load_bundle(path)


def test_an_unknown_member_key_is_refused_by_name(tmp_path):
    raw = _valid_raw()
    raw["members"][0]["harware_config"] = "h.yaml"
    path = _write_bundle(tmp_path, raw)
    with pytest.raises(runner.BundleError, match="'harware_config'"):
        runner.load_bundle(path)


def test_a_missing_member_key_is_refused_by_name(tmp_path):
    raw = _valid_raw()
    del raw["members"][0]["model_id"]
    path = _write_bundle(tmp_path, raw)
    with pytest.raises(runner.BundleError, match="'model_id'"):
        runner.load_bundle(path)


def test_a_duplicate_member_name_is_refused(tmp_path):
    raw = _valid_raw()
    raw["members"].append(copy.deepcopy(raw["members"][0]))
    path = _write_bundle(tmp_path, raw)
    with pytest.raises(runner.BundleError, match="declared twice"):
        runner.load_bundle(path)


def test_an_unknown_mode_is_refused_before_anything_runs(tmp_path):
    raw = _valid_raw()
    raw["members"][0]["mode"] = "ViT"  # the YAML spelling, not the parse mode
    path = _write_bundle(tmp_path, raw)
    with pytest.raises(runner.BundleError, match="mode = 'ViT'"):
        runner.load_bundle(path)


@pytest.mark.parametrize("missing", ["bundle", "manifest", "members"])
def test_a_bundle_missing_a_required_top_level_key_is_refused(tmp_path, missing):
    raw = _valid_raw()
    del raw[missing]
    path = _write_bundle(tmp_path, raw, name=f"{missing}.yaml")
    with pytest.raises(runner.BundleError, match=missing):
        runner.load_bundle(path)


def test_only_refuses_an_unknown_member_by_name():
    bundle = runner.load_bundle(SUBSET_V1)
    with pytest.raises(runner.BundleError, match="names no member"):
        runner.select_members(bundle, ["vit_gigantic"])
    picked = runner.select_members(bundle, ["vit_large_story_64"])
    assert [member["name"] for member in picked] == ["vit_large_story_64"]
    # No filter means the whole set.
    assert len(runner.select_members(bundle, ())) == 4


# ---------------------------------------------------------------------------
# Manifest assembly — from the members' OWN reports and nothing else
# ---------------------------------------------------------------------------


def _fake_report(root, member, *, run_id, selected_id, tokens_per_s=None,
                 requests_per_s=None, label="a label"):
    out = root / member["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if tokens_per_s is not None:
        metrics["tokens_per_s"] = tokens_per_s
    if requests_per_s is not None:
        metrics["requests_per_s"] = requests_per_s
    report = {
        "tool": "fws_qif_dse",
        "schema": "fws_qif_dse/1",
        "run_id": run_id,
        "model_id": member["model_id"],
        "mode": member["mode"],
        "sweep": {"label": label},
        "selected_id": selected_id,
        "selected": None if selected_id is None else {"id": selected_id, "metrics": metrics},
    }
    (out / "dse_report.json").write_text(json.dumps(report, indent=2))
    return out


@pytest.fixture
def two_member_bundle(tmp_path):
    raw = {
        "bundle": "pair",
        "label": "two fake members",
        "manifest": "docs/qif/dse/pair.json",
        "members": [
            {
                "name": "alpha",
                "model_id": "Alpha-1B",
                "label": "Alpha-1B",
                "model_config": "m_a.yaml",
                "hardware_config": "h_a.yaml",
                "mode": "LLM",
                "output_dir": "docs/qif/dse/alpha_frontier",
            },
            {
                "name": "beta",
                "model_id": "Beta-Vision",
                "label": "Beta-Vision",
                "model_config": "m_b.yaml",
                "hardware_config": "h_b.yaml",
                "mode": "VIT",
                "output_dir": "docs/qif/dse/beta_frontier",
            },
        ],
    }
    return _write_bundle(tmp_path, raw, name="pair.yaml")


def test_the_manifest_is_assembled_from_the_member_reports(tmp_path, two_member_bundle):
    bundle = runner.load_bundle(two_member_bundle)
    _fake_report(tmp_path, bundle["members"][0], run_id="aaaa1111", selected_id="c007",
                 tokens_per_s=6110.95, label="Alpha: stage granularity")
    _fake_report(tmp_path, bundle["members"][1], run_id="bbbb2222", selected_id="c000",
                 requests_per_s=42.5, label="Beta: stage granularity")

    manifest = runner.build_manifest(bundle, tmp_path, generated_at="2026-08-28T00:00:00Z")
    assert manifest["schema"] == runner.MANIFEST_SCHEMA
    assert manifest["bundle"] == "pair"
    assert manifest["generated_at"] == "2026-08-28T00:00:00Z"
    assert manifest["members"] == [
        {
            "name": "alpha",
            "model_id": "Alpha-1B",
            "label": "Alpha-1B",  # the MEMBER label, never the sweep prose
            "mode": "LLM",
            "dir": "docs/qif/dse/alpha_frontier",
            "run_id": "aaaa1111",
            "selected_id": "c007",
            "headline": "6110.95 tokens/s",
        },
        {
            "name": "beta",
            "model_id": "Beta-Vision",
            "label": "Beta-Vision",  # the MEMBER label, never the sweep prose
            "mode": "VIT",
            "dir": "docs/qif/dse/beta_frontier",
            "run_id": "bbbb2222",
            "selected_id": "c000",
            "headline": "42.5 requests/s",
        },
    ]


def test_the_manifest_row_carries_the_reports_run_id_not_a_new_one(tmp_path, two_member_bundle):
    # D33: the run id identifies a GENERATION. The manifest must READ it, never
    # mint one, or a page holding the manifest and a report would see two.
    bundle = runner.load_bundle(two_member_bundle)
    for member, run_id in zip(bundle["members"], ("aaaa1111", "bbbb2222")):
        _fake_report(tmp_path, member, run_id=run_id, selected_id="c000", tokens_per_s=1.0)
    manifest = runner.build_manifest(bundle, tmp_path)
    assert [row["run_id"] for row in manifest["members"]] == ["aaaa1111", "bbbb2222"]


def test_an_unranked_member_reports_no_headline_rather_than_a_number(tmp_path, two_member_bundle):
    bundle = runner.load_bundle(two_member_bundle)
    _fake_report(tmp_path, bundle["members"][0], run_id="aaaa1111", selected_id=None)
    _fake_report(tmp_path, bundle["members"][1], run_id="bbbb2222", selected_id="c000",
                 requests_per_s=1.5)
    manifest = runner.build_manifest(bundle, tmp_path)
    assert manifest["members"][0]["selected_id"] is None
    assert manifest["members"][0]["headline"] is None


def test_a_member_with_no_report_is_refused_by_name(tmp_path, two_member_bundle):
    bundle = runner.load_bundle(two_member_bundle)
    _fake_report(tmp_path, bundle["members"][0], run_id="aaaa1111", selected_id="c000",
                 tokens_per_s=1.0)
    with pytest.raises(runner.BundleError, match="'beta' has no"):
        runner.build_manifest(bundle, tmp_path)


def test_only_does_not_truncate_the_manifest(tmp_path, two_member_bundle):
    # --only restricts what is RE-RUN. The manifest still describes the whole
    # bundle, because the atlas draws whatever the manifest lists.
    bundle = runner.load_bundle(two_member_bundle)
    for member, run_id in zip(bundle["members"], ("aaaa1111", "bbbb2222")):
        _fake_report(tmp_path, member, run_id=run_id, selected_id="c000", tokens_per_s=2.0)
    exit_code, path, manifest = runner.run_bundle(
        two_member_bundle, only=("alpha",), root=tmp_path, run=False
    )
    assert exit_code == 0
    assert path == tmp_path / "docs" / "qif" / "dse" / "pair.json"
    assert [row["name"] for row in manifest["members"]] == ["alpha", "beta"]
    assert json.loads(path.read_text()) == manifest


def test_the_cli_writes_the_manifest_without_running_a_sweep(tmp_path, two_member_bundle, capsys):
    bundle = runner.load_bundle(two_member_bundle)
    for member, run_id in zip(bundle["members"], ("aaaa1111", "bbbb2222")):
        _fake_report(tmp_path, member, run_id=run_id, selected_id="c003", tokens_per_s=9.5)
    code = runner.main(
        ["--bundle", str(two_member_bundle), "--output-root", str(tmp_path), "--no-run"]
    )
    assert code == 0
    written = json.loads((tmp_path / "docs" / "qif" / "dse" / "pair.json").read_text())
    assert [row["run_id"] for row in written["members"]] == ["aaaa1111", "bbbb2222"]
    assert "9.5 tokens/s" in capsys.readouterr().out


def test_the_cli_refuses_a_bad_bundle_with_a_nonzero_exit(tmp_path, capsys):
    path = _write_bundle(tmp_path, _valid_raw(bundel="t"))
    assert runner.main(["--bundle", str(path), "--no-run"]) == 2
    assert "'bundel'" in capsys.readouterr().out


def test_run_member_calls_run_sweep_with_the_members_own_fields(tmp_path, two_member_bundle, monkeypatch):
    # The runner is THIN: it forwards the member verbatim and adds nothing.
    seen = {}

    def fake_run_sweep(hardware_config, model_config_path, **kwargs):
        seen["args"] = (hardware_config, model_config_path)
        seen["kwargs"] = kwargs
        return 0, {}

    monkeypatch.setattr(runner, "run_sweep", fake_run_sweep)
    member = runner.load_bundle(two_member_bundle)["members"][1]
    name, code = runner.run_member(member, tmp_path)
    assert (name, code) == ("beta", 0)
    # --output-root moves the ARTIFACTS, not the inputs: the configs are still
    # read from the repo, so a scratch rehearsal prices the same machine.
    assert seen["args"] == (
        str(PROJECT_ROOT / "h_b.yaml"),
        str(PROJECT_ROOT / "m_b.yaml"),
    )
    assert seen["kwargs"]["mode"] == "VIT"
    assert seen["kwargs"]["model_id"] == "Beta-Vision"
    assert seen["kwargs"]["output_dir"] == str(tmp_path / "docs/qif/dse/beta_frontier")


def _report_writing_run_sweep(hardware_config, model_config_path, **kwargs):
    """A stand-in for run_sweep that writes the report shape and nothing else.

    Module level on purpose: --jobs > 1 runs members in SEPARATE PROCESSES, so
    the worker has to be able to reach this function.
    """
    out = Path(kwargs["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "dse_report.json").write_text(
        json.dumps(
            {
                "run_id": "run-" + out.name,
                "model_id": kwargs["model_id"],
                "mode": kwargs["mode"],
                "sweep": {"label": "fake " + out.name},
                "selected_id": "c000",
                "selected": {"id": "c000", "metrics": {"tokens_per_s": 3.25}},
            }
        )
    )
    return 0, {}


def test_jobs_runs_the_members_in_parallel_processes(tmp_path, two_member_bundle, monkeypatch):
    # --jobs uses PROCESSES: the evaluator is pure Python and holds the GIL, so
    # a thread pool would report a parallel run and deliver a serial one.
    monkeypatch.setattr(runner, "run_sweep", _report_writing_run_sweep)
    exit_code, path, manifest = runner.run_bundle(
        two_member_bundle, jobs=2, root=tmp_path, quiet=True
    )
    assert exit_code == 0
    assert path.exists()
    assert [row["run_id"] for row in manifest["members"]] == [
        "run-alpha_frontier",
        "run-beta_frontier",
    ]
    assert {row["headline"] for row in manifest["members"]} == {"3.25 tokens/s"}


def test_the_bundle_exit_code_is_the_worst_members(tmp_path, two_member_bundle, monkeypatch):
    # One member finding no feasible candidate is a failed SET, whatever the
    # other one did; the exit code must not be the last future to land.
    def failing(hardware_config, model_config_path, **kwargs):
        code, payload = _report_writing_run_sweep(hardware_config, model_config_path, **kwargs)
        return (3 if "alpha" in kwargs["output_dir"] else 0), payload

    monkeypatch.setattr(runner, "run_sweep", failing)
    exit_code, _path, _manifest = runner.run_bundle(
        two_member_bundle, jobs=1, root=tmp_path, quiet=True
    )
    assert exit_code == 3


def test_the_manifest_label_is_the_member_short_label_never_the_sweep_prose():
    """The picker shows the manifest label. A sweep's mapping_dse.label is
    report prose ("...: stage granularity x bank depth x capacity mux,
    everything else derived") and reached the model picker once; the member's
    own short label is the only legal source, and it stays picker-sized."""
    import json, io, yaml
    bundle = yaml.safe_load(io.open("configs/run-bundles/subset_v1.yaml", encoding="utf-8"))
    by_name = {m["name"]: m for m in bundle["members"]}
    manifest = json.loads(open("docs/qif/dse/subset_v1.json").read())
    for row in manifest["members"]:
        member = by_name[row["name"]]
        assert row["label"] == member["label"]
        assert len(row["label"]) <= 48, row["label"]
        assert "derived" not in row["label"]
        report = json.loads(qif_artifact(row["dir"] + "/dse_report.json").read_text())
        sweep_label = (report.get("sweep") or {}).get("label", "")
        assert row["label"] != sweep_label
