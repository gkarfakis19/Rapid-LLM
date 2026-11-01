"""Reproduce the 23 valid archive Mamba2 references on the program IR.

CPU only. Reads reference evidence; all generated files stay under --output
(default tmp/ssm-comparison). Exact floats, no tolerances. Hybrid fallback
goldens and failed/refused probes are deliberately excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import json
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from tools.convert_ssm_config import convert_model


def collective_counts(records):
    """A sorted multiset; dropping duplicate collectives must fail the gate."""
    return [dict(collective=list(key), count=count)
            for key, count in sorted(Counter(records).items())]


def compare_peak_memory(old_memory, device_memory, conv_bytes_saved):
    """Compare every rank exactly after two source-backed archive corrections.

    The archive counts SSM layers as both dense and SSM when allocating
    weights, and stores d_conv history slots instead of d_conv - 1.
    """
    expected = []
    for device in old_memory["summary"]:
        rank = device["gpu_id"]
        saved = old_memory["ssm_layers_per_device"][rank] * conv_bytes_saved
        duplicate = (old_memory["transformer_dense_layers_per_device"][rank]
                     * old_memory["weight_mem_per_layer_dense"])
        expected.append(dict(device=rank, conv_history_bytes_saved=saved,
            archive_duplicate_weight_bytes=duplicate,
            old_static_bytes=device["static_gib"] * 1024**3,
            old_peak_bytes=device["peak_gib"] * 1024**3,
            expected_static_bytes=device["static_gib"] * 1024**3 - saved - duplicate,
            expected_peak_bytes=device["peak_gib"] * 1024**3 - saved - duplicate))
    matches = len(device_memory) == 1 and len(expected) == len(device_memory[0]["peak_bytes"])
    if device_memory:
        for row in expected:
            rank = row["device"]
            row["new_static_bytes"] = device_memory[0]["static_bytes"][rank]
            row["new_peak_bytes"] = device_memory[0]["peak_bytes"][rank]
            matches = matches and row["new_static_bytes"] == row["expected_static_bytes"]
            matches = matches and row["new_peak_bytes"] == row["expected_peak_bytes"]
    return expected, matches


def compare_case(reference: Path, output: Path) -> dict:
    import config
    from inference_timing import TimeCalculationLLMInference
    old = json.loads(reference.read_text())
    case = reference.stem
    source = reference.parent / "cases" / case
    for filename, expected in old["configs_sha256"].items():
        actual = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reference hash mismatch: {case}/{filename}")
    output.mkdir(parents=True, exist_ok=True)
    converted = convert_model(yaml.safe_load((source / "model.yaml").read_text()))
    model_path = output / "model.yaml"
    model_path.write_text(yaml.safe_dump(converted, sort_keys=False))
    hw = config.parse_config(str(source / "hardware.yaml"), "hardware")
    model = config.parse_config(str(model_path), "LLM")
    tc = TimeCalculationLLMInference(hw, model, "LLM", output_dir=str(output))
    events = old["runs"][0]["observed"]["events"]
    old_event = next(e for e in events if e["function"] == "build_ssm_timings")
    collectives = []
    ir_collectives = []
    device_memory = []
    def observe(frame, event, result):
        if event == "return" and frame.f_code.co_name == "_prepare_layer_plan_graphs" and frame.f_code.co_filename.endswith("inference_timing.py"):
            # Observe every graph build, including both sampled decode steps,
            # rather than comparing the archive's calls to one final template.
            template = result[0][0].blocks.dense
            for entry in template.entries:
                for key in entry.forward.comm_keys:
                    spec = template.comm_metadata[key]
                    ir_collectives.append((entry.name, spec.kind.value,
                                           spec.size_bytes, spec.participants))
        if event == "return" and frame.f_code.co_name == "simulate_memory" and frame.f_code.co_filename.endswith("program/memory_sim.py"):
            snapshot = frame.f_locals["memory_snapshot"]
            device_memory.append(dict(static_bytes=snapshot.static, peak_bytes=snapshot.peak))
        if event == "return" and frame.f_code.co_name == "collective" and frame.f_code.co_filename.endswith("base_timing.py"):
            loc = frame.f_locals
            if loc.get("participants", 1) > 1 and loc.get("size_bytes", 0) > 0:
                collectives.append(dict(kind=loc["kind"].value, bytes=loc["size_bytes"],
                    participants=loc["participants"], label=loc.get("debug_label", "")))
    start = time.perf_counter()
    with (output / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        sys.setprofile(observe)
        try:
            prefill, _ = tc.calc_time()
            decode, _, samples = tc.calc_decode_time()
        finally:
            sys.setprofile(None)
    current = tc.layer_plan_operation_evidence["ssm"][0]
    op_diffs = []
    for name, prev in old_event["return"][0].items():
        for key in ("compute_time", "comm_time", "comm_bytes", "flops", "memory_accesses"):
            before, after = prev["forward"][key], getattr(current[name].forward, key)
            if before != after:
                op_diffs.append(dict(op=name, field=key, old=before, new=after))
    kernel_diffs = []
    for new_key, old_key in dict(in_proj="in_proj_time", conv="conv_time", scan="scan_time",
        gate="gate_time", out_proj="out_proj_time", rms_bytes="rms_comm_bytes",
        rms_participants="participants").items():
        # Old locals only define `participants` if an RMS collective exists.
        before = old_event["kernel_locals"].get(old_key, 1)
        if new_key == "rms_participants":
            before = old_event["kernel_locals"]["dims"]["tp_shard_factor"]
        after = tc.ssm_kernel_evidence[new_key]
        if before != after:
            kernel_diffs.append(dict(kernel=new_key, old=before, new=after))
    old_colls = collective_counts((e["kind"], e["size_bytes"], e["participants"], e["debug_label"])
        for e in events if e["function"] == "collective" and e["size_bytes"] > 0 and e["participants"] > 1)
    new_colls = collective_counts((e["kind"], e["bytes"], e["participants"], e["label"]) for e in collectives)
    old_ir = collective_counts((e["graph_collective"]["extra"]["op_name"],
        e["graph_collective"]["kind"], e["graph_collective"]["size_bytes"],
        e["graph_collective"]["participants"]) for e in events if "graph_collective" in e
        and e["graph_collective"]["interconnect"] == "tp")
    new_ir = collective_counts(ir_collectives)
    old_mem = next(e for e in events if e["function"] == "build_mamba2_memory_data")
    new_mem = tc.layer_plan_memory_evidence[0]
    memory = dict(old_ssm_state=old_mem["cache_locals"]["ssm_state_bytes"],
        new_ssm_state=new_mem["ssm_state_bytes_per_layer"],
        old_conv_state=old_mem["cache_locals"]["conv_state_bytes"],
        new_conv_state=new_mem["conv_state_bytes_per_layer"],
        old_weights=old_mem["return"]["weight_mem_per_layer"],
        new_weights=new_mem["weight_mem_per_layer"])
    old_peak = next(e["per_device_inputs"] for e in events if e["function"] == "simulate_memory")
    peak_comparison, peak_match = compare_peak_memory(old_peak, device_memory,
        memory["old_conv_state"] - memory["new_conv_state"])
    result = dict(case=case, old=old["totals"]["total_inference_time"], new=prefill + decode,
        op_differences=op_diffs, kernel_differences=kernel_diffs,
        old_collectives=old_colls, new_collectives=new_colls,
        collectives_match=old_colls == new_colls and old_ir == new_ir,
        old_ir_collectives=old_ir, new_ir_collectives=new_ir, memory=memory,
        per_device_memory=device_memory,
        per_device_peak_comparison=peak_comparison, per_device_peak_match=peak_match,
        samples=[dict(step=s.step_id, seconds=s.execution_time) for s in samples],
        wall_seconds=time.perf_counter()-start)
    result["status"] = "MATCH" if result["old"] == result["new"] else "DIFFERENCE"
    pins = json.loads((ROOT / "tests/ssm_gpu_pins.json").read_text())
    if case in pins:
        pin = pins[case]
        checks = ("old", "new", "op_differences", "kernel_differences")
        if all(result[k] == pin[k] for k in checks) and result["collectives_match"]:
            result["status"] = "EXPLAINED"
            result["explanation"] = pin["explanation"]
            # A causal check of the exception: change the kernel backend only,
            # retain the Program builder, collectives, schedule and hardware.
            from llm_execution import LLMExecutionDispatcher
            with (output / "native_backend_ablation.log").open("w") as log, redirect_stdout(log):
                tc._extended_gemm_backend = None
                (native_spec, _), _ = tc.prepare_decode_graphs(
                    batch_size=model.model_config.global_batch_size,
                    total_seq_len=model.model_config.seq_len)
                token = LLMExecutionDispatcher(tc, native_spec).run(tc.execution_mode).total_time
            native_total = token + token * (model.model_config.decode_len - 1)
            result["native_backend_ablation"] = native_total
            if native_total != result["old"]:
                result["status"] = "UNEXPLAINED"
    result["memory_match"] = (memory["old_ssm_state"] == memory["new_ssm_state"]
        and memory["old_weights"] == memory["new_weights"]
        and memory["new_conv_state"] == memory["old_conv_state"] / model.model_config.ssm.d_conv
            * (model.model_config.ssm.d_conv - 1))
    (output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=ROOT.parent / "Rapid-LLM-results/ssm_old_goldens")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/ssm-comparison")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "tmp"):
        parser.error("--output must be inside this worktree's tmp/")
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(runtime)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    sys.dont_write_bytecode = True
    os.environ["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"
    os.environ["ASTRA_CACHE_DIR"] = str(output / "astra-cache")
    rows = []
    for ref in sorted(args.references.glob("mamba*.json")):
        old = json.loads(ref.read_text())
        if old["status"] != "ok" or old["model"] not in ("mamba", "mamba_hf"):
            continue
        if args.cases and ref.stem not in args.cases:
            continue
        try:
            row = compare_case(ref, output / ref.stem)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row = dict(case=ref.stem, status="ERROR", error=str(exc))
        rows.append(row)
        print(f"{ref.stem:65} {row['status']:10} {row.get('old')!r:22} {row.get('new')!r:22} "
              f"ops={len(row.get('op_differences', []))} kernels={len(row.get('kernel_differences', []))} "
              f"collectives={row.get('collectives_match')} peaks={row.get('per_device_peak_match')}", flush=True)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return int(not rows or (args.cases is not None and set(args.cases) != {r["case"] for r in rows})
        or any(r["status"] not in ("MATCH", "EXPLAINED")
        or (r["status"] == "MATCH" and (r.get("op_differences") or r.get("kernel_differences")))
        or not r.get("collectives_match") or not r.get("memory_match")
        or not r.get("per_device_peak_match") for r in rows)
        or (not args.cases and len(rows) != 23))


if __name__ == "__main__":
    raise SystemExit(main())
