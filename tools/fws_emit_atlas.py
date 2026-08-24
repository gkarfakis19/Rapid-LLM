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

"""Emit an `fws_atlas/1` document from a real FWS-CIM mapping (QIF P3.6).

    tools/fws_emit_atlas.py \\
        --hardware_config configs/hardware-config/fws_cim_llama7b.yaml \\
        --model_config configs/model-config/llama2_7b_fws_inf.yaml \\
        --tp 2 --shared_chiplets 2 --out docs/qif/atlas/p3_llama7b_tp2.json

The tool builds the mapping, prints the P3 mapping report (placement,
occupancy, boundary table, disclosures) and writes the atlas document
`docs/qif/atlas/atlas.html` loads. It prices nothing: P3 places and P4 prices
(A1), so the document carries no time.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

import config as config_module  # noqa: E402
import fws_atlas_export  # noqa: E402
import fws_mapping  # noqa: E402


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _relative(path: str) -> str:
    """Repo-relative when the path is inside the repo, so the provenance
    ``command`` string is the command a reader can actually retype."""
    absolute = os.path.abspath(path)
    if absolute.startswith(PROJECT_ROOT + os.sep):
        return os.path.relpath(absolute, PROJECT_ROOT)
    return path


def _hardware(path: str, *, tp: int = None):
    """Parse the hardware YAML, optionally overriding parallelism.tp.

    The tp override moves the HARDWARE setting, not a second copy of it: the
    mapping reads the degree from exactly one place (D21), so a `--tp` that
    only reached the mapping block would be refused by name.
    """
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle)
    raw = copy.deepcopy(raw)
    if tp is not None:
        raw.setdefault("parallelism", {})["tp"] = int(tp)
    config_module.convert(raw)
    return config_module.HWConfig.from_dict(raw)


def build(
    hardware_config: str,
    model_config: str,
    *,
    mode: str = "LLM",
    tp: int = None,
    shared_chiplets: int = None,
    macros_per_chip: int = None,
    layers_per_chip=None,
    title: str = None,
    subtitle: str = "",
    priced: bool = False,
    model_id: str = None,
    label: str = None,
):
    """(mapping, boundary rows, atlas document) for one config pair."""
    hw = _hardware(hardware_config, tp=tp)
    model = config_module.parse_config(model_config, mode)
    declared = getattr(hw, "mapping_config", None)
    spec = declared.system if declared is not None else config_module.MappingSystemConfig()
    overrides = {}
    if shared_chiplets is not None:
        overrides["shared_chiplets"] = int(shared_chiplets)
    if macros_per_chip is not None:
        overrides["macros_per_chip"] = int(macros_per_chip)
    if layers_per_chip is not None:
        overrides["layers_per_chip"] = layers_per_chip
    if overrides:
        spec = config_module.MappingSystemConfig(
            chips=spec.chips,
            macros_per_chip=overrides.get("macros_per_chip", spec.macros_per_chip),
            shared_chiplets=overrides.get("shared_chiplets", spec.shared_chiplets),
            parallelism=spec.parallelism,
            layers_per_chip=overrides.get("layers_per_chip", spec.layers_per_chip),
            membership=spec.membership,
            decode_window=spec.decode_window,
        )
    # ``model_id`` is the model's OWN name. It defaults to model_param.model_type,
    # which is a PRICING CARRIER, not an identity: Granite-4.0-H-Tiny declares
    # model_type: llama for the SwiGLU gated MLP, so without this flag every tile
    # owner in its atlas would read "llama" and tell a reader the wrong model.
    mapping = fws_mapping.build_mapping(
        hw, model, spec=spec, model_id=model_id, label=label
    )
    rows = mapping.boundary_table()
    command = (
        "tools/fws_emit_atlas.py --hardware_config "
        f"{_relative(hardware_config)} --model_config {_relative(model_config)}"
        + (f" --tp {tp}" if tp else "")
        + (f" --shared_chiplets {shared_chiplets}" if shared_chiplets is not None else "")
        + (f" --model_id {model_id}" if model_id else "")
        + (f' --label "{label}"' if label else "")
        + (" --priced" if priced else "")
    )
    # --priced runs P4 over the placed DAG and hands the atlas the duty cycles
    # and labeled metrics it measured. WITHOUT the flag nothing here changes:
    # the document is the placement-only export it always was, byte for byte,
    # because P3 has no timeline (A1) and must never print one.
    duty_cycles = None
    extra_metrics = None
    pool_sizing = None
    extra_relaxations = None
    if priced:
        import fws_eval
        from program.fws_build import build_fws_program

        evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
        duty_cycles = [evaluation.atlas_duty_cycles()]
        extra_metrics = [evaluation.atlas_metrics()]
        # P4 measured the pool concurrency and the decode-window truncation on
        # the SAME timeline these metrics come from. Carrying both keeps the
        # atlas and the report ONE accounting (D21) instead of two documents
        # printing different numbers under the same field name.
        pool_sizing = [evaluation.atlas_pool_sizing()]
        extra_relaxations = [evaluation.atlas_relaxations()]
    document = fws_atlas_export.export_atlas(
        [mapping],
        boundary_rows=[rows],
        title=title or f"{mapping.label} (tp = {mapping.degrees['tp']})",
        subtitle=subtitle
        or (
            "Emitted by QIF P3 from a real mapping, priced by P4: macro duty "
            "cycles and the timing metrics are read off one timeline."
            if priced
            else "Emitted by QIF P3 from a real mapping. Placement only: P3 places, P4 prices."
        ),
        reference_command=command,
        duty_cycles=duty_cycles,
        extra_metrics=extra_metrics,
        pool_sizing=pool_sizing,
        extra_relaxations=extra_relaxations,
    )
    return mapping, rows, document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fws_emit_atlas",
        description="Emit an fws_atlas/1 document from a real FWS-CIM mapping (P3.6).",
    )
    parser.add_argument("--hardware_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--mode", default="LLM", choices=("LLM", "VIT"))
    parser.add_argument("--tp", type=int, default=None, help="Override parallelism.tp.")
    parser.add_argument("--shared_chiplets", type=int, default=None)
    parser.add_argument("--macros_per_chip", type=int, default=None)
    parser.add_argument("--layers_per_chip", default=None)
    parser.add_argument(
        "--model_id",
        default=None,
        help=(
            "The model's own name for owners and labels. Defaults to "
            "model_param.model_type, which is a PRICING CARRIER and not always "
            "the model's identity."
        ),
    )
    parser.add_argument("--label", default=None, help="Override the system label.")
    parser.add_argument(
        "--priced",
        action="store_true",
        help=(
            "Run P4 over the placed DAG and carry its duty cycles and timing "
            "metrics into the document. Off by default: the placement-only "
            "export is unchanged."
        ),
    )
    parser.add_argument("--out", default=None, help="Where to write the atlas document.")
    args = parser.parse_args(argv)

    layers = args.layers_per_chip
    if layers is not None and layers != "auto":
        layers = int(layers)
    try:
        mapping, rows, document = build(
            args.hardware_config,
            args.model_config,
            mode=args.mode,
            tp=args.tp,
            shared_chiplets=args.shared_chiplets,
            macros_per_chip=args.macros_per_chip,
            layers_per_chip=layers,
            priced=args.priced,
            model_id=args.model_id,
            label=args.label,
        )
    except ValueError as exc:
        print(f"[FWS-CIM ATLAS] error: {exc}")
        return 2
    print(mapping.report(rows))
    if args.out:
        fws_atlas_export.write_atlas_json(document, args.out)
        print(f"[FWS-CIM ATLAS] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
