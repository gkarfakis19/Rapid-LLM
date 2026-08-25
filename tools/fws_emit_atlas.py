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

`--pd` emits the PD two-system document instead (D16): a prefill inventory,
a decode inventory and the one handoff link priced as bytes, which the atlas
draws side by side.

    tools/fws_emit_atlas.py --pd \\
        --hardware_config configs/hardware-config/fws_cim_llama7b_mapped.yaml \\
        --model_config configs/model-config/llama2_7b_fws_inf.yaml \\
        --pd_prefill_layers_per_chip 8 --pd_decode_layers_per_chip 4 --priced

The tool builds the mapping, prints the P3 mapping report (placement,
occupancy, boundary table, disclosures) and writes the atlas document
`docs/qif/atlas/atlas.html` loads. Without `--priced` it prices nothing: P3
places and P4 prices (A1), so the document carries no time.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
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


def _override_spec(spec, *, shared_chiplets=None, macros_per_chip=None, layers_per_chip=None,
                   decode_window=None):
    """A copy of ``spec`` with the CLI's declared fields replacing its own.

    Every field is a DECLARED claim the mapper checks (P3.1), so an override
    here is still a claim and still refused when the placement contradicts it.
    """
    if all(
        value is None
        for value in (shared_chiplets, macros_per_chip, layers_per_chip, decode_window)
    ):
        return spec
    # dataclasses.replace, not a field-by-field rebuild: a rebuild DROPS every
    # field it does not name, and it has now done that twice — the declared
    # PLACEMENT LAW (P7.3) and then the declared SERVING REGIME (P7.7), each of
    # which would have drawn a different machine under the same title. replace
    # carries everything the spec has, including fields added after this line
    # was written.
    overrides = {}
    if macros_per_chip is not None:
        overrides["macros_per_chip"] = int(macros_per_chip)
    if shared_chiplets is not None:
        overrides["shared_chiplets"] = int(shared_chiplets)
    if layers_per_chip is not None:
        overrides["layers_per_chip"] = layers_per_chip
    if decode_window is not None:
        overrides["decode_window"] = int(decode_window)
    return dataclasses.replace(spec, **overrides)


def _price(mapping):
    """P4 over one placed DAG: (duty cycles, metrics, pool sizing, disclosures).

    One call per INVENTORY. A PD pair is two inventories (D16) and each half is
    priced on its own timeline; sharing one evaluation between them would print
    a single measurement twice under two system ids.
    """
    import fws_eval
    from program.fws_build import build_fws_program

    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
    return (
        evaluation.atlas_duty_cycles(),
        evaluation.atlas_metrics(),
        evaluation.atlas_pool_sizing(),
        evaluation.atlas_relaxations(),
    )


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
    spec = _override_spec(
        spec,
        shared_chiplets=shared_chiplets,
        macros_per_chip=macros_per_chip,
        layers_per_chip=layers_per_chip,
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
        # P4 measured the pool concurrency and the decode-window truncation on
        # the SAME timeline these metrics come from. Carrying both keeps the
        # atlas and the report ONE accounting (D21) instead of two documents
        # printing different numbers under the same field name.
        duty, metrics, pools, disclosures = _price(mapping)
        duty_cycles = [duty]
        extra_metrics = [metrics]
        pool_sizing = [pools]
        extra_relaxations = [disclosures]
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


# The two disclosures a PD document owes its reader. Both are properties of how
# D16 is implemented, not of any one run, so they ride every PD document.
PD_WHOLE_MODEL_DISCLOSURE = {
    "constraint": "pd_each_half_maps_the_whole_model",
    "value": "both inventories place every layer of the model",
    "reason": (
        "D16 splits the PHASE, not the weights: a prefill system spec and a "
        "decode system spec, each a complete inventory, plus a handoff priced "
        "as bytes. Every macro count, column count and silicon figure in this "
        "document is therefore a WHOLE machine's, once per system, and the two "
        "must not be read as halves of one."
    ),
}
PD_WHOLE_REQUEST_DISCLOSURE = {
    "constraint": "pd_each_half_prices_the_whole_request",
    "value": "prefill + a bounded decode window lowered on BOTH inventories",
    "reason": (
        "P3's DAG builder lowers a whole request onto whichever inventory it is "
        "given, so the prefill system's decode-step metrics say what that "
        "inventory would do if it also decoded, and the decode system's prefill "
        "latency likewise. Every metric names its system (metrics[].key is "
        "prefixed with it); the only coupling D16 declares between the two is "
        "the handoff link's bytes, and no figure here is a split-workload one."
    ),
}


def build_pd(
    hardware_config: str,
    model_config: str,
    *,
    mode: str = "LLM",
    tp: int = None,
    prefill_shared_chiplets: int = None,
    decode_shared_chiplets: int = None,
    prefill_layers_per_chip=None,
    decode_layers_per_chip=None,
    decode_window: int = None,
    title: str = None,
    subtitle: str = "",
    priced: bool = False,
    label: str = "PD pair",
):
    """(pair, per-system boundary rows, atlas document) for a PD machine (D16).

    Two inventories and one handoff. The halves start from the hardware
    config's ``mapping.pd`` block when it declares one, else from its unified
    ``mapping:`` block, and the per-half flags override from there — so a
    config that has never heard of PD can still be drawn as one.
    """
    hw = _hardware(hardware_config, tp=tp)
    model = config_module.parse_config(model_config, mode)
    declared = getattr(hw, "mapping_config", None)
    base = declared.system if declared is not None else config_module.MappingSystemConfig()
    prefill_base = declared.prefill if declared is not None and declared.prefill else base
    decode_base = declared.decode if declared is not None and declared.decode else base
    prefill_spec = _override_spec(
        prefill_base,
        shared_chiplets=prefill_shared_chiplets,
        layers_per_chip=prefill_layers_per_chip,
    )
    decode_spec = _override_spec(
        decode_base,
        shared_chiplets=decode_shared_chiplets,
        layers_per_chip=decode_layers_per_chip,
        decode_window=decode_window,
    )
    pair = fws_mapping.build_pd_pair(
        hw, model, prefill_spec=prefill_spec, decode_spec=decode_spec, label=label
    )
    rows = [pair.prefill.boundary_table(), pair.decode.boundary_table()]
    command = (
        "tools/fws_emit_atlas.py --pd --hardware_config "
        f"{_relative(hardware_config)} --model_config {_relative(model_config)}"
        + (f" --tp {tp}" if tp else "")
        + (
            f" --pd_prefill_layers_per_chip {prefill_layers_per_chip}"
            if prefill_layers_per_chip is not None
            else ""
        )
        + (
            f" --pd_decode_layers_per_chip {decode_layers_per_chip}"
            if decode_layers_per_chip is not None
            else ""
        )
        + (
            f" --pd_prefill_shared_chiplets {prefill_shared_chiplets}"
            if prefill_shared_chiplets is not None
            else ""
        )
        + (
            f" --pd_decode_shared_chiplets {decode_shared_chiplets}"
            if decode_shared_chiplets is not None
            else ""
        )
        + (f" --pd_decode_window {decode_window}" if decode_window is not None else "")
        + (f' --label "{label}"' if label != "PD pair" else "")
        + (" --priced" if priced else "")
    )
    duty_cycles = None
    extra_metrics = None
    pool_sizing = None
    # Whichever PD facts apply ride BOTH halves, because they are facts about
    # the DOCUMENT and not about one inventory: an entry every system states
    # identically merges to one unattributed line, while one only the prefill
    # half raised would print as the prefill half's claim (D21).
    #
    # The whole-MODEL fact rides every PD document. The whole-REQUEST fact rides
    # priced ones only, and the branch below is where it is added: an unpriced
    # document lowers no request, so there is no request-shaped claim to make
    # and disclosing one would be a relaxation of a constraint nothing imposed.
    extra_relaxations = [[PD_WHOLE_MODEL_DISCLOSURE], [PD_WHOLE_MODEL_DISCLOSURE]]
    if priced:
        halves = [_price(pair.prefill), _price(pair.decode)]
        duty_cycles = [half[0] for half in halves]
        extra_metrics = [half[1] for half in halves]
        pool_sizing = [half[2] for half in halves]
        shared = [PD_WHOLE_MODEL_DISCLOSURE, PD_WHOLE_REQUEST_DISCLOSURE]
        extra_relaxations = [shared + list(halves[0][3]), shared + list(halves[1][3])]
    document = fws_atlas_export.export_pd_atlas(
        pair,
        title=title or f"{label} — prefill and decode systems (D16)",
        subtitle=subtitle
        or (
            "Two inventories, one priced handoff. Each half is placed by P3 and "
            "priced by P4 on its OWN timeline."
            if priced
            else "Two inventories, one priced handoff. Placement only: P3 places, P4 prices."
        ),
        reference_command=command,
        duty_cycles=duty_cycles,
        extra_metrics=extra_metrics,
        pool_sizing=pool_sizing,
        extra_relaxations=extra_relaxations,
    )
    return pair, rows, document



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
        "--pd",
        action="store_true",
        help=(
            "Emit the PD two-system document (D16): a prefill inventory, a "
            "decode inventory and the one handoff link priced as bytes."
        ),
    )
    parser.add_argument("--pd_prefill_layers_per_chip", default=None)
    parser.add_argument("--pd_decode_layers_per_chip", default=None)
    parser.add_argument("--pd_prefill_shared_chiplets", type=int, default=None)
    parser.add_argument("--pd_decode_shared_chiplets", type=int, default=None)
    parser.add_argument(
        "--pd_decode_window",
        type=int,
        default=None,
        help="Decode steps the decode half lowers (ADJ-6's bounded window).",
    )
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

    def _layers(value):
        if value is None or value == "auto":
            return value
        return int(value)

    try:
        if args.pd:
            built, rows, document = build_pd(
                args.hardware_config,
                args.model_config,
                mode=args.mode,
                tp=args.tp,
                prefill_shared_chiplets=args.pd_prefill_shared_chiplets,
                decode_shared_chiplets=args.pd_decode_shared_chiplets,
                prefill_layers_per_chip=_layers(args.pd_prefill_layers_per_chip),
                decode_layers_per_chip=_layers(args.pd_decode_layers_per_chip),
                decode_window=args.pd_decode_window,
                priced=args.priced,
                label=args.label or "PD pair",
            )
            report = built.report()
        else:
            built, rows, document = build(
                args.hardware_config,
                args.model_config,
                mode=args.mode,
                tp=args.tp,
                shared_chiplets=args.shared_chiplets,
                macros_per_chip=args.macros_per_chip,
                layers_per_chip=_layers(args.layers_per_chip),
                priced=args.priced,
                model_id=args.model_id,
                label=args.label,
            )
            report = built.report(rows)
    except ValueError as exc:
        print(f"[FWS-CIM ATLAS] error: {exc}")
        return 2
    print(report)
    if args.out:
        fws_atlas_export.write_atlas_json(document, args.out)
        print(f"[FWS-CIM ATLAS] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
