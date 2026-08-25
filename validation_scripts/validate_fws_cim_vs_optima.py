#!/usr/bin/env python
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
"""Validate the fws_cim device class against the recorded OPTIMA targets.

Runs the three DESIGN section-5 tiers (T1/T2/T3) through ``run_perf.py``
exactly as a user would, reads ``output/VIT/fws_cim_report.json``, and checks
every target in the section-5 table:

- cycle counts and array counts: EXACT;
- times, areas, and fps: <= 0.1 % relative;
- config inputs (model dims, seq, adc_mux geometry, E_vec, area/array,
  fabric clock): EXACT, read back from the shipped YAML templates so a
  template edit cannot silently drift off the recorded reference.

Block-latency convention (pinned by tests/test_fws_cim.py): the OPTIMA
reference recorded block latency over SIX stages — S1..S5 plus a trailing
peripherals stage that ran at exactly one analog stage time. The report's
``block_latency_us`` is sum(S1..S5) per DESIGN 4.3, so this script compares
the recorded targets against ``block_latency_us + S1_time`` (the analog
stage time at the run's seq_len).

Also runs the two DESIGN section-5 regression gates (a100_80GB + vit_base,
a100_80GB + Llama2-7B; the Llama run takes about a minute) against saved
baseline result files: the timing lines must match exactly and the whole
results file must stay bit-identical. Then prints an A100-vs-FWS-CIM
comparison on the seq-196 ViT workload (demo output, not a gated check).

After the pass-1 checks, the DESIGN2 section-4 pass-2A rows run (appended;
the pass-1 rows above stay byte-for-byte the same checks):

- dense-LLM smoke (fws_cim_llama7b + llama2_7b_fws_inf): exit 0, report in
  ``output/LLM``, decode section monotone in context, KV section fields,
  and the sustained-vs-fabric-ceiling reconciliation (wavefront counts and
  the sustained tok/s recomputed independently here);
- cim_dram variant (fws_cim_llama7b_kvdram): KV story fields, kv_read-bound
  decode S2, KV-traffic energy, side capacity check in the capacity text;
- MoE smoke (fws_cim_moe + moe_small_fws_inf): expert arrays census,
  dispatch/combine boundary entries on the ep link, serialized disclosure;
- law spot-checks recomputed INDEPENDENTLY in this script (imports
  cim_timing in-process): decode attention cycles at two contexts from the
  closed form, the MoE dense-reduction identity (E=1/top_k=1/shared=0), and
  the lm_head arrays law.

After the pass-2A checks, the DESIGN2 section-5 pass-2B rows run (appended;
every earlier row stays byte-for-byte the same check). They exercise the DSE
tool (``tools/fws_cim_dse.py``) as a subprocess, exactly as a user would:

- T1 sweep: a 3-variant sweep containing the recorded T1 array point must
  select that point (period/fps/arrays equal to the T1 targets, and equal to
  the T1 run_perf report captured earlier in this script) and keep it on the
  throughput-vs-area Pareto front;
- llama7b sweep: a 2-variant sweep (the shipped Llama point + an infeasible
  tiny array) with ``--emit-config --verify`` must emit a runnable YAML,
  round-trip through run_perf within 0.1 % on period, tok/s (fabric
  ceiling), and sustained tok/s (the decode selection key), and record
  the infeasible candidate with a stage tag + message;
- MoE expert-parallel: sweeping ``cim.dse.moe_expert_parallel: [1, 2]`` on
  the MoE smoke pair must scale the expert pool (0 chips at k=1,
  num_moe_layers * k at k=2) and exactly halve the one-way dispatch time.

The DSE runs write only under ``output/fws_cim_dse/validation_*`` (the
``--verify`` run_perf executes with cwd inside that directory), so the
output/VIT and output/LLM artifacts the earlier rows read stay untouched.

Exit code 0 only if every gated check passes.

Usage (from the repo root):
    .venv/bin/python validation_scripts/validate_fws_cim_vs_optima.py

Baseline locations can be overridden with RAPID_A100_VIT_BASELINE and
RAPID_A100_LLAMA2_7B_BASELINE.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_DIR = os.path.join(REPO_ROOT, "configs", "hardware-config")
MODEL_DIR = os.path.join(REPO_ROOT, "configs", "model-config")
VIT_RESULTS = os.path.join(REPO_ROOT, "output", "VIT", "LLM_inference_results.txt")
LLM_RESULTS = os.path.join(REPO_ROOT, "output", "LLM", "LLM_inference_results.txt")
FWS_REPORT = os.path.join(REPO_ROOT, "output", "VIT", "fws_cim_report.json")

# Pass-2A artifacts (LLM-mode runs land in output/LLM, never output/VIT).
LLM_FWS_REPORT = os.path.join(REPO_ROOT, "output", "LLM", "fws_cim_report.json")
LLM_MEM_CAPACITY = os.path.join(
    REPO_ROOT, "output", "LLM", "memory-summary", "memory_capacity_comparison.txt"
)

# Bit-identical GPU baselines committed under tests/baselines (durable, so
# the gates cannot silently rot); the env vars compare against a different
# recording.
_BASELINE_DIR = os.path.join(REPO_ROOT, "tests", "baselines")
BASELINE_A100_VIT = os.environ.get(
    "RAPID_A100_VIT_BASELINE",
    os.path.join(_BASELINE_DIR, "baseline_a100_vit_results.txt"),
)
BASELINE_A100_LLAMA = os.environ.get(
    "RAPID_A100_LLAMA2_7B_BASELINE",
    os.path.join(_BASELINE_DIR, "baseline_a100_llama2_7b_results.txt"),
)

REL_TOL = 1e-3  # <= 0.1 % relative for times / areas / fps

# Lines in LLM_inference_results.txt that carry timing numbers (the gate's
# stated match criterion; the full file is additionally checked bit-identical).
TIMING_LINE_PREFIXES = (
    "Inference Time for batch:",
    "Prefill Time:",
    "Decode Time:",
    "Time to First Token:",
    "Decode Generations per Second:",
    "Aggregate Decode Throughput",
)

# DESIGN section 5 table, verbatim. Cycle/array counts exact; times/areas/fps
# checked at REL_TOL. "block_latency_recorded_us" is the OPTIMA six-stage
# figure (see module docstring). Config-input rows are exact.
TIERS = [
    {
        "name": "T1 (A_2D)",
        "hw": "fws_cim_optima_t1.yaml",
        "model": "vit_huge_story_64_inf.yaml",
        "inputs": {
            "hidden_dim": 1280,
            "intermediate_size": 5120,
            "num_layers": 32,
            "num_heads": 16,
            "seq_len": 64,
            "adc_mux": 4,
            "rows": 1280,
            "cols_adc": 320,
            "energy_per_vec_pj": 9534.9389,
            "area_mm2_per_array": 1.403065,
            "fabric_clock_ghz": 0.95,
        },
        "targets": {
            "qk_cycles": 2747,
            "sv_cycles": 4471,
            "period_us": 5.12,
            "bottleneck_stage": "S1_qkv",
            "block_latency_recorded_us": 30.407368,
            "fps": 195312.5,
            "arrays_per_layer": 12,
            "ctt_area_mm2": 538.7768,
        },
    },
    {
        "name": "T2 (B_3D)",
        "hw": "fws_cim_optima_t2.yaml",
        "model": "vit_g_64_inf.yaml",
        "inputs": {
            "hidden_dim": 1408,
            "intermediate_size": 6144,
            "num_layers": 40,
            "num_heads": 16,
            "seq_len": 64,
            "adc_mux": 2,
            "rows": 1408,
            "cols_adc": 704,
            "energy_per_vec_pj": 10138.5868,
            "area_mm2_per_array": 2.014071,
            "fabric_clock_ghz": 1.0,
        },
        "targets": {
            "qk_cycles": 3003,
            "sv_cycles": 4471,
            "period_us": 4.567,
            "bottleneck_stage": "S2_attention",
            "block_latency_recorded_us": 17.367,
            "fps": 218962.12,
            "arrays_per_layer": 14,
            "ctt_area_mm2": 1127.8796,
        },
    },
    {
        "name": "T3 (seq196)",
        "hw": "fws_cim_optima_t3.yaml",
        "model": "vit_huge_story_196_inf.yaml",
        "inputs": {
            "hidden_dim": 1280,
            "intermediate_size": 5120,
            "num_layers": 32,
            "num_heads": 16,
            "seq_len": 196,
            "adc_mux": 16,
            "rows": 1280,
            "cols_adc": 80,
            "energy_per_vec_pj": 12470.0063,
            "area_mm2_per_array": 1.135296,
            "fabric_clock_ghz": 0.95,
        },
        "targets": {
            "qk_cycles": 38471,
            "sv_cycles": 45219,
            "period_us": 62.72,
            "bottleneck_stage": "S1_qkv",
            "block_latency_recorded_us": 361.3,
            "fps": 15943.8776,
            "arrays_per_layer": 12,
            "ctt_area_mm2": 435.9537,
        },
    },
]


class CheckTable:
    """Collect (name, expected, measured, tolerance, pass) rows and render them."""

    def __init__(self):
        self.rows = []

    def exact(self, name, expected, measured):
        ok = measured == expected
        self.rows.append((name, str(expected), str(measured), "exact", ok))
        return ok

    def close(self, name, expected, measured, rel_tol=REL_TOL):
        try:
            rel = abs(float(measured) - float(expected)) / abs(float(expected))
            ok = rel <= rel_tol
            note = "<=0.1% (got {:.2e})".format(rel)
        except (TypeError, ValueError, ZeroDivisionError):
            ok = False
            note = "<=0.1% (not comparable)"
        self.rows.append((name, repr(expected), repr(measured), note, ok))
        return ok

    def boolean(self, name, description, ok, detail=""):
        self.rows.append((name, description, detail or ("yes" if ok else "NO"), "bool", ok))
        return ok

    @property
    def all_pass(self):
        return all(row[4] for row in self.rows)

    def render(self):
        headers = ("check", "expected", "measured", "tolerance", "status")
        widths = [len(h) for h in headers]
        printable = []
        for name, exp, meas, tol, ok in self.rows:
            row = (name, exp, meas, tol, "PASS" if ok else "FAIL")
            printable.append(row)
            widths = [max(w, len(c)) for w, c in zip(widths, row)]
        fmt = "  ".join("{:<%d}" % w for w in widths)
        lines = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
        lines += [fmt.format(*row) for row in printable]
        return "\n".join(lines)


def run_perf(hw_yaml, model_yaml):
    """Run run_perf.py as a subprocess from the repo root; return the process."""
    return subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "run_perf.py"),
            "--hardware_config",
            hw_yaml,
            "--model_config",
            model_yaml,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=600,
    )


def load_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def check_tier(tier, table):
    """Run one tier and check every section-5 target for it."""
    hw_path = os.path.join(HW_DIR, tier["hw"])
    model_path = os.path.join(MODEL_DIR, tier["model"])
    name = tier["name"]

    proc = run_perf(hw_path, model_path)
    if not table.boolean(
        "%s: run_perf exit code" % name,
        "0",
        proc.returncode == 0,
        str(proc.returncode),
    ):
        print(proc.stdout[-3000:])
        return
    if not table.boolean(
        "%s: fws_cim_report.json written" % name,
        "exists",
        os.path.exists(FWS_REPORT),
    ):
        return
    with open(FWS_REPORT) as handle:
        report = json.load(handle)

    # Config inputs: model dims from the model YAML, device geometry from the
    # hardware YAML, derived seq from the report. All exact.
    ins = tier["inputs"]
    model_cfg = load_yaml(model_path)["model_param"]
    hw_cfg = load_yaml(hw_path)
    analog = hw_cfg["cim"]["analog"]
    fabric = hw_cfg["cim"]["fabric"]
    table.exact("%s: hidden_dim" % name, ins["hidden_dim"], model_cfg["hidden_dim"])
    table.exact(
        "%s: intermediate_size" % name,
        ins["intermediate_size"],
        model_cfg["intermediate_size"],
    )
    table.exact("%s: num_layers" % name, ins["num_layers"], model_cfg["num_layers"])
    table.exact(
        "%s: num_heads" % name, ins["num_heads"], model_cfg["attention"]["num_heads"]
    )
    table.exact("%s: seq_len (report)" % name, ins["seq_len"], report["seq_len"])
    table.exact("%s: adc_mux" % name, ins["adc_mux"], analog["adc_mux"])
    table.exact("%s: analog rows" % name, ins["rows"], analog["rows"])
    table.exact("%s: cols_adc" % name, ins["cols_adc"], analog["cols_adc"])
    table.exact(
        "%s: energy_per_vec_pj" % name,
        ins["energy_per_vec_pj"],
        analog["energy_per_vec_pj"],
    )
    table.exact(
        "%s: area_mm2_per_array" % name,
        ins["area_mm2_per_array"],
        analog["area_mm2_per_array"],
    )
    table.exact(
        "%s: fabric clock_ghz" % name, ins["fabric_clock_ghz"], fabric["clock_ghz"]
    )

    # Recorded outputs.
    tgt = tier["targets"]
    table.exact("%s: QK cycles" % name, tgt["qk_cycles"], report["qk_cycles"])
    table.exact("%s: PV cycles" % name, tgt["sv_cycles"], report["sv_cycles"])
    table.close("%s: period_us" % name, tgt["period_us"], report["period_us"])
    table.exact(
        "%s: bottleneck stage" % name,
        tgt["bottleneck_stage"],
        report["bottleneck_stage"],
    )
    # OPTIMA recorded six stages: S1..S5 + trailing peripherals stage of one
    # analog stage time (== S1 time). See module docstring.
    recorded_block = report["block_latency_us"] + report["stages"]["S1_qkv"]["time_us"]
    table.close(
        "%s: block latency (recorded, 6-stage)" % name,
        tgt["block_latency_recorded_us"],
        recorded_block,
    )
    table.close("%s: fps" % name, tgt["fps"], report["fps"])
    table.exact(
        "%s: arrays/layer" % name, tgt["arrays_per_layer"], report["arrays_per_layer"]
    )
    table.close(
        "%s: CTT area mm2 (transformer stack)" % name,
        tgt["ctt_area_mm2"],
        report["area_mm2"]["transformer_stack"],
    )


def timing_lines(text):
    return [
        line
        for line in text.splitlines()
        if line.strip().startswith(TIMING_LINE_PREFIXES)
    ]


def check_regression_gate(name, hw_yaml, model_yaml, results_path, baseline_path, table):
    """One DESIGN section-5 regression gate: timing lines + bit-identity."""
    if not table.boolean(
        "%s: baseline file present" % name,
        baseline_path,
        os.path.exists(baseline_path),
    ):
        return
    proc = run_perf(os.path.join(HW_DIR, hw_yaml), os.path.join(MODEL_DIR, model_yaml))
    if not table.boolean(
        "%s: run_perf exit code" % name, "0", proc.returncode == 0, str(proc.returncode)
    ):
        print(proc.stdout[-3000:])
        return
    with open(results_path, "rb") as handle:
        current = handle.read()
    with open(baseline_path, "rb") as handle:
        baseline = handle.read()

    cur_lines = timing_lines(current.decode())
    base_lines = timing_lines(baseline.decode())
    lines_ok = cur_lines == base_lines and len(base_lines) > 0
    table.boolean(
        "%s: timing lines match baseline" % name,
        "%d lines" % len(base_lines),
        lines_ok,
        "match" if lines_ok else "MISMATCH",
    )
    bytes_ok = current == baseline
    table.boolean(
        "%s: results file bit-identical" % name,
        "no tolerance",
        bytes_ok,
        "identical" if bytes_ok else "DIFFERS",
    )
    if not lines_ok or not bytes_ok:
        print("--- %s: baseline timing lines ---" % name)
        print("\n".join(base_lines))
        print("--- %s: current timing lines ---" % name)
        print("\n".join(cur_lines))


def a100_vit196_prefill_seconds():
    """Full-precision A100 prefill on the seq-196 ViT (demo comparison only).

    run_perf's results file rounds prefill to 3 decimals, too coarse for a
    millisecond-scale ViT run, so this reruns the same pipeline in a child
    process and prints the raw float. Output goes to a temp dir so the
    tier reports in output/VIT stay untouched.
    """
    snippet = (
        "import sys, config\n"
        "from inference_timing import TimeCalculationLLMInference\n"
        "hw = config.parse_config(sys.argv[1], config_type='hardware')\n"
        "model = config.parse_config(sys.argv[2], config_type='VIT')\n"
        "config.validate_configs(hw, model)\n"
        "tc = TimeCalculationLLMInference(hw, model, 'VIT', output_dir=sys.argv[3])\n"
        "timing = tc.calc_total_inference_time()\n"
        "print('PREFILL_S=%r' % (timing['prefill_time'],))\n"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                snippet,
                os.path.join(HW_DIR, "a100_80GB.yaml"),
                os.path.join(MODEL_DIR, "vit_huge_story_196_inf.yaml"),
                tmp_dir,
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=600,
        )
    if proc.returncode != 0:
        return None, proc.stdout[-3000:]
    match = re.search(r"PREFILL_S=([0-9.eE+-]+)", proc.stdout)
    if not match:
        return None, proc.stdout[-3000:]
    return float(match.group(1)), None


def print_comparison(t3_report, table):
    """A100 vs FWS-CIM on the same seq-196 ViT workload (demo, not gated)."""
    prefill_s, err = a100_vit196_prefill_seconds()
    table.boolean(
        "A100-vs-FWS comparison run completes",
        "a100_80GB + vit_huge_story_196_inf",
        prefill_s is not None,
    )
    print()
    print("A100 vs FWS-CIM — ViT-Huge-story, seq 196, batch 1 (demo, not a gated target)")
    print("-" * 78)
    if prefill_s is None:
        print("A100 comparison run FAILED:")
        print(err)
        return
    a100_us = prefill_s * 1e6
    a100_fps = 1.0 / prefill_s
    fws_us = t3_report["end_to_end_latency_us"]
    fws_fps = t3_report["fps"]
    fmt = "{:<34}{:>18}{:>18}"
    print(fmt.format("", "A100 (a100_80GB)", "FWS-CIM (T3)"))
    print(fmt.format("latency (us)", "{:.2f}".format(a100_us), "{:.2f}".format(fws_us)))
    print(fmt.format("throughput (images/s)", "{:.2f}".format(a100_fps), "{:.2f}".format(fws_fps)))
    print(
        fmt.format(
            "latency ratio (A100/FWS)", "", "{:.2f}x".format(a100_us / fws_us)
        )
    )
    print(
        fmt.format(
            "throughput ratio (FWS/A100)", "", "{:.2f}x".format(fws_fps / a100_fps)
        )
    )
    print()
    print("Note: A100 latency is the prefill (single-batch) time on 2 GPUs (tp=2);")
    print("its throughput is 1/prefill. FWS-CIM latency is the non-pipelined")
    print("single-sample end-to-end time on one chip; its throughput is the")
    print("pipelined stage rate (1/pipeline_period).")


# ---------------------------------------------------------------------------
# Pass 2A (DESIGN2 section 4): LLM / MoE smokes + independent law spot-checks
# ---------------------------------------------------------------------------


def _exists_after_nfs_settle(path, attempts=6, delay=0.5):
    """os.path.exists with a bounded retry (P7.9).

    run_perf rmtree()s and recreates output/<mode> under the repo root and the
    caller stats a path inside it the moment the subprocess returns. On this
    NFS-backed checkout the directory intermittently reads back EMPTY even
    though the write succeeded, which produced the misleading pair "run_perf
    exit code: 0 / report written: NO" and turned the standing gate red for an
    infrastructure reason (seen once in eight identical invocations). Re-stat
    a few times, listing the parent to force revalidation of the cached
    directory entry, before believing the absence. This changes no check's
    MEANING: a report that was never written is still absent after the retries.
    """
    import time

    for attempt in range(attempts):
        if os.path.exists(path):
            return True
        try:
            os.listdir(os.path.dirname(path) or ".")
        except OSError:
            pass
        if attempt + 1 < attempts:
            time.sleep(delay)
    return os.path.exists(path)


def _read_llm_report(name, table):
    """Read output/LLM/fws_cim_report.json right after a run (runs clobber it)."""
    if not table.boolean(
        "%s: fws_cim_report.json written (output/LLM)" % name,
        "exists",
        _exists_after_nfs_settle(LLM_FWS_REPORT),
    ):
        return None
    with open(LLM_FWS_REPORT) as handle:
        return json.load(handle)


def _strictly_increasing(values):
    return all(a < b for a, b in zip(values, values[1:]))


def _llm_results_text():
    """Read output/LLM/LLM_inference_results.txt right after a run."""
    with open(LLM_RESULTS) as handle:
        return handle.read()


def check_llm_dense_smoke(table):
    """Dense-LLM smoke: fws_cim_llama7b + llama2_7b_fws_inf (cim_sram story).

    Expected values are recomputed inline from the config dims (B=4,
    prefill 1792, decode 256, L=32, MHA kv=32, d=128, bf16 KV):
    kv_bytes_per_stream = L * 2 * kv * d * ctx_final * 2 B = 1 GiB;
    stub tier 8 GB => max_streams 8, max_context(B=4) 4096.
    """
    name = "2A dense-LLM"
    proc = run_perf(
        os.path.join(HW_DIR, "fws_cim_llama7b.yaml"),
        os.path.join(MODEL_DIR, "llama2_7b_fws_inf.yaml"),
    )
    if not table.boolean(
        "%s: run_perf exit code" % name, "0", proc.returncode == 0, str(proc.returncode)
    ):
        print(proc.stdout[-3000:])
        return
    report = _read_llm_report(name, table)
    if report is None:
        return

    table.exact(
        "%s: layer classes" % name, {"dense": 32, "moe": 0}, report["layer_classes"]
    )

    # KV section (cim_sram): the sharded per-stream law at the final context.
    kv = report["kv"]
    table.exact("%s: KV story" % name, "cim_sram", kv["story"])
    table.exact(
        "%s: KV bytes/stream (sharded law)" % name,
        32 * 2 * 32 * 128 * 2048 * 2,  # L * 2(K+V) * kv_heads * d * ctx * bf16
        kv["bytes_per_stream"],
    )
    table.exact("%s: KV max streams" % name, 8, kv["max_streams"])
    table.exact("%s: KV max context at B=4" % name, 4096, kv["max_context_at_batch"])
    table.exact("%s: KV fits" % name, True, kv["fits"])

    # Decode section: direct law evaluation at {prefill+1, midpoint, final}.
    decode = report["decode"]
    contexts = [entry["context"] for entry in decode["contexts"]]
    table.exact("%s: decode contexts" % name, [1793, 1920, 2048], contexts)
    sa_times = [entry["s2"]["sa_time_us"] for entry in decode["contexts"]]
    table.boolean(
        "%s: decode attention grows with context" % name,
        "sa strictly increasing",
        _strictly_increasing(sa_times),
        " < ".join("%.4f" % t for t in sa_times),
    )
    kv_reads = [entry["s2"]["kv_read_time_us"] for entry in decode["contexts"]]
    table.boolean(
        "%s: decode KV read grows with context" % name,
        "kv_read strictly increasing",
        _strictly_increasing(kv_reads),
        " < ".join("%.4f" % t for t in kv_reads),
    )
    final = decode["contexts"][-1]
    table.close(
        "%s: aggregate tok/s = B / period(final)" % name,
        4 / (final["period_us"] * 1e-6),
        decode["aggregate_tokens_per_s_final"],
        rel_tol=1e-9,
    )
    # Sustained-vs-ceiling reconciliation (CimDeviceModel law), recomputed
    # independently from the report's own step latency / period / KV
    # figures: W_full = ceil(4564.77/141.24) = 33 wavefronts to fill the
    # pipeline, W_kv = floor(8 streams / B=4) = 2 wavefronts resident, so
    # KV capacity binds and sustained = 2 * 4 / step latency.
    table.exact(
        "%s: wavefronts_full = ceil(step latency / period)" % name,
        int(math.ceil(final["step_latency_us"] / final["period_us"])),
        decode["wavefronts_full"],
    )
    table.exact(
        "%s: wavefronts_kv = floor(KV max streams / B) = 2" % name,
        kv["max_streams"] // 4,
        decode["wavefronts_kv"],
    )
    table.exact(
        "%s: decode throughput limit (2 < 33 wavefronts)" % name,
        "kv_capacity",
        decode["decode_throughput_limit"],
    )
    table.close(
        "%s: sustained tok/s = W_kv * B / step latency" % name,
        2 * 4 / (final["step_latency_us"] * 1e-6),
        decode["sustained_tokens_per_s"],
        rel_tol=1e-9,
    )
    table.boolean(
        "%s: fabric ceiling relabeled in the results text" % name,
        "'fabric ceiling' + 'sustained at KV capacity' lines present",
        "fabric ceiling" in _llm_results_text()
        and "sustained at KV capacity" in _llm_results_text(),
    )
    table.boolean(
        "%s: staircase caveat disclosed" % name,
        "integration_note mentions sample_every",
        "sample_every" in decode["integration_note"],
    )

    # Model-shaped endpoints: lm_head on the last chip.
    lm_head = report["endpoint_stages"]["lm_head"]
    table.exact("%s: lm_head arrays" % name, 8, lm_head["arrays"])
    table.exact("%s: lm_head chip (last)" % name, 3, lm_head["chip"])


def check_llm_kvdram_smoke(table):
    """cim_dram variant: KV tier fields, kv_read-bound S2, side capacity check."""
    name = "2A kvdram"
    proc = run_perf(
        os.path.join(HW_DIR, "fws_cim_llama7b_kvdram.yaml"),
        os.path.join(MODEL_DIR, "llama2_7b_fws_inf.yaml"),
    )
    if not table.boolean(
        "%s: run_perf exit code" % name, "0", proc.returncode == 0, str(proc.returncode)
    ):
        print(proc.stdout[-3000:])
        return
    report = _read_llm_report(name, table)
    if report is None:
        return

    kv = report["kv"]
    table.exact("%s: KV story" % name, "cim_dram", kv["story"])
    table.exact("%s: KV tier capacity (bytes)" % name, 8589934592, kv["capacity_bytes"])
    table.exact("%s: KV fits (4 GiB in 8 GiB)" % name, True, kv["fits"])
    # The 100 GB/s tier binds decode S2 (kv_read >> sa on this geometry).
    table.exact(
        "%s: decode S2 bound (final context)" % name,
        "kv_read",
        report["decode"]["contexts"][-1]["s2"]["bound"],
    )
    table.boolean(
        "%s: KV traffic joins PARTIAL energy" % name,
        "kv_dram_traffic > 0",
        report["energy_partial_pj"]["kv_dram_traffic"] > 0,
        repr(report["energy_partial_pj"]["kv_dram_traffic"]),
    )
    with open(LLM_MEM_CAPACITY) as handle:
        mem_text = handle.read()
    table.boolean(
        "%s: side capacity check in capacity report" % name,
        "'KV DRAM headroom' present",
        "KV DRAM headroom" in mem_text,
    )


def check_moe_smoke(table):
    """MoE smoke: fws_cim_moe + moe_small_fws_inf (E=16, top_k=2, shared=1)."""
    name = "2A MoE"
    proc = run_perf(
        os.path.join(HW_DIR, "fws_cim_moe.yaml"),
        os.path.join(MODEL_DIR, "moe_small_fws_inf.yaml"),
    )
    if not table.boolean(
        "%s: run_perf exit code" % name, "0", proc.returncode == 0, str(proc.returncode)
    ):
        print(proc.stdout[-3000:])
        return
    report = _read_llm_report(name, table)
    if report is None:
        return

    table.exact(
        "%s: layer classes" % name, {"dense": 1, "moe": 11}, report["layer_classes"]
    )
    # Expert arrays census: E=16 routed experts, one array each per FFN stage
    # (H=1024, I_moe=512, gated fused: ceil(2*512/1024)=1; ceil(512/1024)=1).
    moe = report["moe"]
    table.exact(
        "%s: routed FFN1 expert arrays" % name, 16, moe["stage_arrays"]["ffn1_routed"]
    )
    table.exact(
        "%s: routed FFN2 expert arrays" % name, 16, moe["stage_arrays"]["ffn2_routed"]
    )
    table.exact("%s: arrays per MoE layer" % name, 38, moe["arrays_per_moe_layer"])
    # Dispatch/combine: 11 MoE layers x 2 boundary entries on the ep link;
    # bytes each way = tokens_owner * top_k * H * act = (4*480) * 2 * 1024 * 2.
    table.exact("%s: dispatch/combine entries" % name, 22, moe["dispatch"]["count"])
    table.exact("%s: dispatch link" % name, "ep", moe["dispatch"]["link"])
    table.exact(
        "%s: dispatch bytes each way" % name,
        4 * 480 * 2 * 1024 * 2,
        moe["dispatch"]["bytes_each_way"],
    )
    table.boolean(
        "%s: serialized-experts disclosure" % name,
        "'serialized' in sequential note",
        "serialized" in (report["sequential_latency_note"] or ""),
    )


def _sa_cycles_closed_form(m, n, k, r, c):
    """Independent systolic closed form: ceil(M/R)*ceil(N/C)*(K+R+C-2) - 1."""
    return -(-m // r) * -(-n // c) * (k + r + c - 2) - 1


def check_law_spot_checks(table):
    """Law spot-checks recomputed independently in this script (in-process).

    Imports cim_timing/config from the repo and compares the device model's
    numbers against closed forms evaluated here from the shipped YAML values
    — never against the device model's own helpers.
    """
    name = "2A law"
    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import cim_timing
        import config as rapid_config

        # --- Decode attention cycles at two contexts (closed form) --------
        hw_llama = rapid_config.parse_config(
            os.path.join(HW_DIR, "fws_cim_llama7b.yaml"), "hardware"
        )
        model_llama = rapid_config.parse_config(
            os.path.join(MODEL_DIR, "llama2_7b_fws_inf.yaml"), "VIT"
        ).model_config
        dev_llama = cim_timing.CimDeviceModel(hw_llama, model_llama)

        hw_yaml = load_yaml(os.path.join(HW_DIR, "fws_cim_llama7b.yaml"))
        fabric = hw_yaml["cim"]["fabric"]
        r, c = int(fabric["rows"]), int(fabric["cols"])
        fill_drain = int(fabric.get("fill_drain_penalty_cycles", 3 * r))
        replicas = int(fabric.get("replicas", 1))
        model_yaml = load_yaml(os.path.join(MODEL_DIR, "llama2_7b_fws_inf.yaml"))
        mp = model_yaml["model_param"]
        num_heads = int(mp["attention"]["num_heads"])
        kv_heads = num_heads  # attention_type mha
        head_dim = int(mp["hidden_dim"]) // num_heads
        shared_heads = num_heads // kv_heads
        h_fold = -(-kv_heads // replicas)  # heads_chip at tp=1, per replica
        for ctx in (512, 4096):
            qk = _sa_cycles_closed_form(shared_heads, ctx, head_dim * h_fold, r, c)
            pv = _sa_cycles_closed_form(shared_heads, head_dim, ctx * h_fold, r, c)
            expected_total = max(qk, pv) + fill_drain
            att = dev_llama.decode_attention_timing(ctx, batch_size=1)
            table.exact(
                "%s: decode attention cycles @ctx=%d" % (name, ctx),
                expected_total,
                att.total_cycles,
            )

        # --- MoE dense-reduction identity (E=1, top_k=1, shared=0) --------
        hw_moe = rapid_config.parse_config(
            os.path.join(HW_DIR, "fws_cim_moe.yaml"), "hardware"
        )
        params = cim_timing.CimModelParams(
            hidden_dim=1024,
            intermediate_size=2048,
            num_layers=4,
            num_heads=16,
            kv_heads=4,
            head_dim=64,
            seq_len=256,
            batch_size=2,
            gated_mlp=True,
            patch_dim=0,
            num_classes=0,
            num_experts=1,
            top_k=1,
            moe_intermediate_size=512,
            n_shared_experts=0,
            expert_imbalance_factor=1.0,
            moe_layer_mask=(False, True, True, True),
            vocab_size=32000,
        )
        dev_identity = cim_timing.CimDeviceModel(hw_moe, params)
        tokens = 512
        table.exact(
            "%s: MoE FFN stage == dense analog law (T=%d)" % (name, tokens),
            dev_identity.analog_gemm_time(tokens),
            dev_identity.moe_ffn_stage_time(tokens),
        )
        moe_yaml = load_yaml(os.path.join(HW_DIR, "fws_cim_moe.yaml"))
        analog = moe_yaml["cim"]["analog"]
        vec_latency = (
            int(analog["adc_mux"])
            * int(analog["slice_cycles"])
            / (float(analog["analog_clock_mhz"]) * 1e6)
        )
        table.close(
            "%s: identity value = T * vec_latency" % name,
            tokens * vec_latency,
            dev_identity.moe_ffn_stage_time(tokens),
            rel_tol=1e-12,
        )

        # --- lm_head arrays law -------------------------------------------
        llama_analog = hw_yaml["cim"]["analog"]
        hidden = int(mp["hidden_dim"])
        vocab = int(mp["vocab_size"])
        rows = int(llama_analog["rows"])
        stored_cols = int(llama_analog["cols_adc"]) * int(llama_analog["adc_mux"])
        expected_lm_head = -(-hidden // rows) * -(-vocab // stored_cols)
        table.exact(
            "%s: lm_head arrays (H=%d, V=%d)" % (name, hidden, vocab),
            expected_lm_head,
            dev_llama.endpoint_arrays()["lm_head"],
        )
    except Exception as exc:  # pragma: no cover - defensive: fail the table
        table.boolean(
            "%s: spot-checks completed" % name,
            "no exception",
            False,
            "%s: %s" % (type(exc).__name__, exc),
        )


# ---------------------------------------------------------------------------
# Pass 2B (DESIGN2 section 5): DSE tool checks (tools/fws_cim_dse.py)
# ---------------------------------------------------------------------------

DSE_TOOL = os.path.join(REPO_ROOT, "tools", "fws_cim_dse.py")
DSE_OUT_ROOT = os.path.join(REPO_ROOT, "output", "fws_cim_dse")

# QIF P3.7: the MAPPED-path sweep is a DIFFERENT tool with a different
# evaluator (fws_mapping -> fws_build -> fws_eval, never a closed form), so it
# gets its own tool path and its own output root.
QIF_DSE_TOOL = os.path.join(REPO_ROOT, "tools", "fws_qif_dse.py")
QIF_DSE_OUT_ROOT = os.path.join(REPO_ROOT, "output", "fws_qif_dse")
QIF_DSE_DEMO = os.path.join(
    REPO_ROOT, "docs", "qif", "dse", "granite_capacity_banks", "dse_report.json"
)
#: The Wave-D sweep the one above replaced (ADJ-9). FROZEN and labelled: its
#: config declared `vector_lanes` as an axis and tools/fws_qif_dse.py now
#: refuses that axis by name, so it can never be regenerated. It is kept
#: because it is the measurement ADJ-9 was adjudicated on.
QIF_DSE_DEMO_RETIRED = os.path.join(
    REPO_ROOT, "docs", "qif", "dse", "granite_lanes_banks_retired", "dse_report.json"
)
#: QIF P7.4/P7.5 — the regime-v2 FRONTIER curves (D29-D32). Read, not rerun:
#: `tests/test_qif_frontier.py` re-walks each ladder and compares it, so the
#: rows here are about WHAT THE FRONTIER FOUND.
QIF_FRONTIER_CURVES = (
    ("Granite-4.0-H-Tiny", "granite_4_0_h_tiny_frontier"),
    ("Qwen3.5-4B", "qwen3_5_4b_frontier"),
)
QIF_DSE_DEMO_SELECTED = os.path.join(
    HW_DIR, "fws_cim_granite_tiny_dse_selected.yaml"
)

# The recorded OPTIMA T1 and T3 array points as cim.dse.variants entries
# (rows / slice_cycles / analog_clock_mhz inherit cim.analog = the T1 point).
DSE_T1_VARIANT = {
    "adc_mux": 4,
    "cols_adc": 320,
    "energy_per_vec_pj": 9534.9389,
    "area_mm2_per_array": 1.403065,
}
DSE_T3_VARIANT = {
    "adc_mux": 16,
    "cols_adc": 80,
    "energy_per_vec_pj": 12470.0063,
    "area_mm2_per_array": 1.135296,
}
# Slower (mux 8) AND larger per-array area: dominated by the T1 point on
# both Pareto axes, so it must stay off the front.
DSE_DOMINATED_VARIANT = {
    "adc_mux": 8,
    "cols_adc": 320,
    "energy_per_vec_pj": 9534.9389,
    "area_mm2_per_array": 5.0,
}
# The shipped fws_cim_llama7b analog point as a variant (feasible)...
DSE_LLAMA_VARIANT = {
    "adc_mux": 4,
    "cols_adc": 1024,
    "energy_per_vec_pj": 97637.774,
    "area_mm2_per_array": 14.367386,
}
# ...and a tiny array that cannot hold even one Llama layer on a chip
# (placement-stage infeasible; the reason must be recorded).
DSE_TINY_VARIANT = {
    "adc_mux": 1,
    "cols_adc": 8,
    "rows": 64,
    "energy_per_vec_pj": 1.0,
    "area_mm2_per_array": 0.001,
}
# The shipped fws_cim_moe analog point as a variant.
DSE_MOE_VARIANT = {
    "adc_mux": 4,
    "cols_adc": 256,
    "energy_per_vec_pj": 6102.361,
    "area_mm2_per_array": 0.897962,
}


def _write_dse_hw_yaml(base_hw_name, dse_block, out_path):
    """Shipped template + a cim.dse block, written next to the DSE artifacts."""
    hw_dict = load_yaml(os.path.join(HW_DIR, base_hw_name))
    hw_dict["cim"]["dse"] = dse_block
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        yaml.safe_dump(hw_dict, handle, sort_keys=False)
    return out_path


def _run_dse_tool(hw_yaml_path, model_yaml_name, out_dir, extra_args=()):
    """Run tools/fws_cim_dse.py as a subprocess; return (proc, payload|None).

    The tool writes only under out_dir (and --verify runs run_perf with cwd
    = out_dir), so output/VIT and output/LLM stay untouched.
    """
    proc = subprocess.run(
        [
            sys.executable,
            DSE_TOOL,
            "--hardware_config",
            hw_yaml_path,
            "--model_config",
            os.path.join(MODEL_DIR, model_yaml_name),
            "--output-dir",
            out_dir,
        ]
        + list(extra_args),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=1200,
    )
    report_path = os.path.join(out_dir, "dse_report.json")
    payload = None
    if os.path.exists(report_path):
        with open(report_path) as handle:
            payload = json.load(handle)
    return proc, payload


def check_dse_t1_sweep(t1_report, table):
    """(a) T1 + a 3-variant sweep containing the T1 point.

    The selected point must match the recorded T1 targets (period 5.12 us,
    fps 195312.5 at REL_TOL — the tier-row convention; arrays/chips exact)
    AND the T1 run_perf report captured earlier in this script (tight
    1e-9, since the DSE and run_perf share CimDeviceModel), and the T1
    point must sit on the throughput-vs-area Pareto front.
    """
    name = "2B DSE T1"
    out_dir = os.path.join(DSE_OUT_ROOT, "validation_t1")
    # The T1 point is deliberately LAST: all three variants are feasible, so
    # a broken selection that ignored the objective and returned the first
    # feasible candidate would pick the dominated mux-8 point and fail the
    # selected-variant row below. Never put the expected winner first.
    hw_path = _write_dse_hw_yaml(
        "fws_cim_optima_t1.yaml",
        {
            "variants": [DSE_DOMINATED_VARIANT, DSE_T3_VARIANT, DSE_T1_VARIANT],
            "tp_candidates": [1],
        },
        os.path.join(out_dir, "fws_cim_t1_dse.yaml"),
    )
    proc, payload = _run_dse_tool(hw_path, "vit_huge_story_64_inf.yaml", out_dir)
    if not table.boolean(
        "%s: dse exit code" % name, "0", proc.returncode == 0, str(proc.returncode)
    ):
        print(proc.stdout[-3000:])
        return
    if not table.boolean(
        "%s: dse_report.json written" % name, "exists", payload is not None
    ):
        return

    table.exact("%s: candidates swept" % name, 3, payload["num_candidates"])
    table.exact("%s: workload" % name, "prefill", payload["workload"])
    selected = payload["selected"]
    if not table.boolean(
        "%s: a candidate is selected" % name, "selected != None", selected is not None
    ):
        return
    knobs = selected["knobs"]
    table.exact(
        "%s: selected variant is the T1 point (mux, cols_adc)" % name,
        (4, 320),
        (knobs["adc_mux"], knobs["cols_adc"]),
    )
    table.exact("%s: selected tp" % name, 1, knobs["tp"])

    metrics = selected["metrics"]
    table.close("%s: selected period_us (T1 target)" % name, 5.12, metrics["period_us"])
    table.close("%s: selected fps (T1 target)" % name, 195312.5, metrics["fps"])
    table.exact("%s: selected arrays_total" % name, 386, metrics["arrays_total"])
    table.exact("%s: selected chips" % name, 1, metrics["chips"])
    table.exact(
        "%s: derived layers_per_chip" % name, [32], metrics["derived_layers_per_chip"]
    )
    if table.boolean(
        "%s: T1 run report captured" % name, "from the tier run", t1_report is not None
    ):
        table.close(
            "%s: period_us == T1 run_perf report" % name,
            t1_report["period_us"],
            metrics["period_us"],
            rel_tol=1e-9,
        )
        table.close(
            "%s: fps == T1 run_perf report" % name,
            t1_report["fps"],
            metrics["fps"],
            rel_tol=1e-9,
        )

    front = payload["front_ids"]
    table.boolean(
        "%s: T1 point on the Pareto front" % name,
        "selected_id in front_ids",
        selected["id"] in front,
        "front=%s" % front,
    )
    dominated = next(
        (c for c in payload["candidates"] if c["knobs"]["adc_mux"] == 8), None
    )
    table.boolean(
        "%s: dominated mux-8 variant off the front" % name,
        "feasible but dominated",
        dominated is not None and dominated["ok"] and dominated["id"] not in front,
        "-" if dominated is None else "id=%s ok=%s" % (dominated["id"], dominated["ok"]),
    )


def check_dse_llama_verify(table):
    """(b) llama7b + a small sweep with --emit-config --verify.

    The emitted config must exist, the run_perf round trip must match the
    closed form within 0.1 % on period, tok/s (fabric ceiling), and
    sustained tok/s (the decode selection key), and the infeasible tiny
    variant must carry a stage tag + message.
    """
    name = "2B DSE llama7b"
    out_dir = os.path.join(DSE_OUT_ROOT, "validation_llama7b")
    hw_path = _write_dse_hw_yaml(
        "fws_cim_llama7b.yaml",
        {"variants": [DSE_LLAMA_VARIANT, DSE_TINY_VARIANT], "tp_candidates": [1]},
        os.path.join(out_dir, "fws_cim_llama7b_dse.yaml"),
    )
    emitted_path = os.path.join(out_dir, "selected_config.yaml")
    proc, payload = _run_dse_tool(
        hw_path,
        "llama2_7b_fws_inf.yaml",
        out_dir,
        extra_args=["--emit-config", emitted_path, "--verify"],
    )
    if not table.boolean(
        "%s: dse exit code (incl. verify)" % name,
        "0",
        proc.returncode == 0,
        str(proc.returncode),
    ):
        print(proc.stdout[-3000:])
        return
    if not table.boolean(
        "%s: dse_report.json written" % name, "exists", payload is not None
    ):
        return

    table.exact("%s: workload" % name, "decode", payload["workload"])
    table.exact(
        "%s: valid/total candidates" % name,
        (1, 2),
        (payload["num_valid"], payload["num_candidates"]),
    )
    table.boolean(
        "%s: emitted config exists" % name, emitted_path, os.path.exists(emitted_path)
    )

    verify = payload.get("verify")
    if table.boolean(
        "%s: --verify ran and passed" % name,
        "run_perf round trip <= 0.1%",
        bool(verify) and verify.get("ran") and verify.get("pass"),
    ):
        for check in verify["checks"]:
            # expected = the simulator's number, measured = the closed form.
            table.close(
                "%s: verify %s (run_perf vs closed form)" % (name, check["name"]),
                check["simulator"],
                check["dse"],
            )

    infeasible = [c for c in payload["candidates"] if not c["ok"]]
    table.boolean(
        "%s: every infeasible candidate carries a reason" % name,
        "stage tag + message, >= 1 infeasible",
        len(infeasible) >= 1
        and all(c["fail_stage"] and c["fail_message"] for c in infeasible),
        "; ".join("%s [%s]" % (c["id"], c["fail_stage"]) for c in infeasible) or "none",
    )
    tiny = next((c for c in infeasible if c["knobs"]["rows"] == 64), None)
    table.boolean(
        "%s: tiny variant fails at the placement stage" % name,
        "fail_stage == placement, names the layer class",
        tiny is not None
        and tiny["fail_stage"] == "placement"
        and "cannot place layer" in tiny["fail_message"],
        "-" if tiny is None else tiny["fail_message"][:60],
    )


def check_dse_moe_expert_parallel(table):
    """(c) MoE sweep over moe_expert_parallel [1, 2].

    At k=2 each of the 11 MoE layers' routed experts spread over 2 dedicated
    chips (expert pool = 22) and the one-way dispatch time is exactly half
    the k=1 figure (k parallel ep links divide the whole p2p time).
    """
    name = "2B DSE MoE"
    out_dir = os.path.join(DSE_OUT_ROOT, "validation_moe")
    hw_path = _write_dse_hw_yaml(
        "fws_cim_moe.yaml",
        {
            "variants": [DSE_MOE_VARIANT],
            "tp_candidates": [1],
            "moe_expert_parallel": [1, 2],
        },
        os.path.join(out_dir, "fws_cim_moe_dse.yaml"),
    )
    proc, payload = _run_dse_tool(hw_path, "moe_small_fws_inf.yaml", out_dir)
    if not table.boolean(
        "%s: dse exit code" % name, "0", proc.returncode == 0, str(proc.returncode)
    ):
        print(proc.stdout[-3000:])
        return
    if not table.boolean(
        "%s: dse_report.json written" % name, "exists", payload is not None
    ):
        return

    table.exact(
        "%s: moe_expert_parallel candidates" % name,
        [1, 2],
        payload["moe_expert_parallel_candidates"],
    )
    by_k = {c["knobs"]["moe_expert_parallel"]: c for c in payload["candidates"]}
    if not table.boolean(
        "%s: both k=1 and k=2 feasible" % name,
        "2 valid candidates",
        len(by_k) == 2 and all(c["ok"] for c in by_k.values()),
        "num_valid=%d" % payload["num_valid"],
    ):
        return
    m1 = by_k[1]["metrics"]
    m2 = by_k[2]["metrics"]

    # Expert-pool chips scale with k: 0 at k=1 (experts stay on the layer
    # chips), num_moe_layers * k = 11 * 2 at k=2.
    table.exact("%s: expert-pool chips at k=1" % name, 0, m1["expert_pool_chips"])
    table.exact("%s: expert-pool chips at k=2 (11 MoE layers x 2)" % name, 22, m2["expert_pool_chips"])
    table.exact(
        "%s: k=2 total chips = backbone + pool" % name,
        m2["backbone_chips"] + 22,
        m2["chips"],
    )

    # Dispatch time halves at k=2 (one-way figure in the DSE metrics).
    table.boolean(
        "%s: dispatch time present and positive at k=1" % name,
        "dispatch_time_us > 0",
        m1["dispatch_time_us"] is not None and m1["dispatch_time_us"] > 0,
        repr(m1["dispatch_time_us"]),
    )
    table.close(
        "%s: dispatch time halves at k=2" % name,
        m1["dispatch_time_us"] / 2.0,
        m2["dispatch_time_us"],
        rel_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# Pass-3 rows: the bridge gate (QIF P6.3, ADJ-8)
# ---------------------------------------------------------------------------

#: The degenerate points both paths express: one model, one owner per macro, no
#: column sharing, no bit-slicing, no PD split, tp = 1. ADJ-8's set exactly —
#: the three ViT tiers, the dense-LLM point and the MoE smoke.
BRIDGE_POINTS = (
    ("bridge T1", "fws_cim_optima_t1.yaml", "vit_huge_story_64_inf.yaml", "VIT"),
    ("bridge T2", "fws_cim_optima_t2.yaml", "vit_g_64_inf.yaml", "VIT"),
    ("bridge T3", "fws_cim_optima_t3.yaml", "vit_huge_story_196_inf.yaml", "VIT"),
    ("bridge Llama7B", "fws_cim_llama7b.yaml", "llama2_7b_fws_inf.yaml", "LLM"),
    ("bridge MoE", "fws_cim_moe.yaml", "moe_small_fws_inf.yaml", "LLM"),
)


def check_bridge_gate(table):
    """Closed form vs placed DAG on the degenerate case, one config at a time.

    The closed-form side is the REAL artifact: run_perf writes
    ``fws_cim_report.json`` from the hardware YAML with no ``mapping:`` block.
    The DAG side is built in-process from the SAME two YAMLs, so nothing but
    the evaluation path differs. Every exclusion the gate makes is named in
    ``fws_bridge.EXCLUSIONS`` and printed below the table.

    These runs overwrite output/VIT and output/LLM, so they go LAST: every
    earlier row has already read the artifact it needed.
    """
    sys.path.insert(0, REPO_ROOT)
    import config as _config
    import fws_bridge as _fws_bridge
    import fws_eval as _fws_eval
    import fws_mapping as _fws_mapping
    from program.fws_build import build_fws_program as _build_fws_program

    for label, hw_yaml, model_yaml, mode in BRIDGE_POINTS:
        hw_path = os.path.join(HW_DIR, hw_yaml)
        model_path = os.path.join(MODEL_DIR, model_yaml)
        proc = run_perf(hw_path, model_path)
        if not table.boolean(
            "%s: closed-form run exit code" % label,
            "0",
            proc.returncode == 0,
            str(proc.returncode),
        ):
            print(proc.stdout[-3000:])
            continue
        report_path = os.path.join(REPO_ROOT, "output", mode, "fws_cim_report.json")
        if not table.boolean(
            "%s: closed-form report written" % label,
            "exists",
            _exists_after_nfs_settle(report_path),
        ):
            continue
        with open(report_path) as handle:
            closed_form = json.load(handle)

        raw = load_yaml(hw_path)
        _config.convert(raw)
        bridged_hw = _config.HWConfig.from_dict(raw)
        # ADJ-10 PIN. The mapped path DERIVES the attention fabric's array count
        # and the softmax pipeline's lane count; the frozen closed form has no
        # timeline to derive them from and prices the DECLARED cim.fabric seed.
        # Pinning the DAG side to that same seed is what makes the two sides one
        # machine, and it uses the ordinary card override so the run discloses
        # it. No row is dropped and no tolerance moves — every attention cycle
        # count below is still compared EXACTLY. Named in
        # fws_bridge.EXCLUSIONS as derived_fabric_pinned_to_the_declared_seed.
        _digital = bridged_hw.cim_config.cards.digital_card
        _digital.fabric_num_arrays = int(_digital.fabric.num_arrays)
        _digital.fabric_softmax_lanes = int(_digital.fabric.softmax_lanes)
        mapping = _fws_mapping.build_mapping(
            bridged_hw, _config.parse_config(model_path, mode)
        )
        evaluation = _fws_eval.evaluate_fws(_build_fws_program(mapping))
        device = mapping.device
        table.boolean(
            "%s: the DAG side is pinned to the closed form's own fabric (ADJ-10)" % label,
            "num_arrays == %d and softmax_lanes == %d, DECLARED on both sides"
            % (int(_digital.fabric.num_arrays), int(_digital.fabric.softmax_lanes)),
            (
                device.fabric_num_arrays == int(_digital.fabric.num_arrays)
                and device.fabric_softmax_lanes == int(_digital.fabric.softmax_lanes)
                and device.derived_fabric is None
                and len(device.fabric_sizing_disclosures()) == 1
            ),
            "num_arrays %d, softmax_lanes %d, derivation skipped, %d disclosure(s)"
            % (
                device.fabric_num_arrays,
                device.fabric_softmax_lanes,
                len(device.fabric_sizing_disclosures()),
            ),
        )
        for row in _fws_bridge.bridge_rows(evaluation, closed_form, label):
            if row.kind == "exact":
                table.exact(row.name, row.expected, row.measured)
            elif row.kind == "close":
                table.close(row.name, row.expected, row.measured)
            else:
                table.boolean(row.name, row.expected, row.ok, str(row.measured))

    # D21: a relaxation that is not disclosed in the artifact is the AUDIT's
    # finding 5. The gate's exclusions are part of its output, not a footnote.
    table.boolean(
        "bridge: exclusions are disclosed by name",
        "%d named exclusions printed" % len(_fws_bridge.EXCLUSIONS),
        len(_fws_bridge.EXCLUSIONS) > 0,
        ", ".join(name for name, _reason in _fws_bridge.EXCLUSIONS),
    )
    print()
    print("Bridge gate (P6.3 / ADJ-8) — what it deliberately does NOT compare:")
    for name, reason in _fws_bridge.EXCLUSIONS:
        print("  * %s: %s" % (name, reason))


def _run_qif_dse_tool(hw_yaml_path, model_yaml_name, out_dir, extra_args=()):
    """Run tools/fws_qif_dse.py as a subprocess; return (proc, payload|None).

    Like the closed-form rows, the tool writes only under out_dir (and
    --verify runs run_perf with cwd = out_dir), so output/VIT and output/LLM
    stay untouched.
    """
    proc = subprocess.run(
        [
            sys.executable,
            QIF_DSE_TOOL,
            "--hardware_config",
            hw_yaml_path,
            "--model_config",
            os.path.join(MODEL_DIR, model_yaml_name),
            "--output-dir",
            out_dir,
            "--quiet",
        ]
        + list(extra_args),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=1800,
    )
    report_path = os.path.join(out_dir, "dse_report.json")
    payload = None
    if os.path.exists(report_path):
        with open(report_path) as handle:
            payload = json.load(handle)
    return proc, payload


def check_qif_dse_selection(table):
    """(P3.7 a) The CHECKED-IN demo sweep: Granite-4.0-H-Tiny over
    arrays_per_chip {604,640,965,1207} x bank_depth {1,2}.

    Read, not rerun: `tests/test_qif_dse_allocation.py` regenerates this
    artifact and compares it, so the rows here are about WHAT THE SWEEP FOUND.

    ADJ-9 REWRITE. OLD AXES: `vector_lanes x bank_depth`, and the finding was
    that the headline rises with the DECLARED lane count. D31-v2 derives the
    engine to the analog floor and the tool now refuses the lane axis by name,
    so the sweep's first axis is CHIP CAPACITY and the finding inverts: at a
    fixed bank depth capacity moves no throughput at all, only silicon, which
    is Invariant W (D27) on a cross product. The old artifact is frozen at
    docs/qif/dse/granite_lanes_banks_retired and checked below under its own
    rows.
    """
    name = "P3.7 demo sweep"
    if not table.boolean(
        "%s: checked-in artifact exists" % name,
        "docs/qif/dse/granite_lanes_banks/dse_report.json",
        os.path.exists(QIF_DSE_DEMO),
    ):
        return
    with open(QIF_DSE_DEMO) as handle:
        payload = json.load(handle)

    table.exact(
        "%s: axes swept" % name,
        {"arrays_per_chip": [604, 640, 965, 1207], "bank_depth": [1, 2]},
        payload["sweep"]["axes"],
    )
    table.boolean(
        "%s: the engine width is REFUSED as an axis (D31/ADJ-9)" % name,
        "vector_lanes is in sweep.refused_axes and in no axes[] anywhere",
        "vector_lanes" in payload["sweep"].get("refused_axes", {})
        and "vector_lanes" not in payload["sweep"]["axes"],
        "refused: %s" % sorted(payload["sweep"].get("refused_axes", {})),
    )
    table.exact("%s: candidates" % name, 8, payload["num_candidates"])
    table.exact("%s: valid candidates" % name, 8, payload["num_valid"])
    table.boolean(
        "%s: every candidate priced on the mapped path" % name,
        "fws_mapping -> fws_build -> fws_eval",
        payload["evaluation_path"].startswith("fws_mapping.build_mapping"),
        payload["evaluation_path"][:60],
    )

    # THE FINDING (ADJ-9): at a fixed bank depth, CHIP CAPACITY moves the
    # silicon and NOT the headline. Every candidate derives its own engine and
    # the derived width does not depend on how many empty slots a chip has.
    for bank in (1, 2):
        rows = sorted(
            (c for c in payload["candidates"] if c["knobs"]["bank_depth"] == bank),
            key=lambda c: c["knobs"]["arrays_per_chip"],
        )
        rates = {round(c["metrics"]["tokens_per_s"], 9) for c in rows}
        silicon = [c["silicon"]["total_silicon_mm2"] for c in rows]
        table.boolean(
            "%s: capacity buys silicon and NOT throughput (bank_depth %d)" % (name, bank),
            "one tokens/s over the four capacities; total silicon strictly rising",
            len(rows) == 4
            and len(rates) == 1
            and all(b > a for a, b in zip(silicon, silicon[1:])),
            "%.1f tokens/s over %s mm2"
            % (list(rates)[0], ", ".join("%.0f" % value for value in silicon)),
        )
    fastest = max(c["metrics"]["tokens_per_s"] for c in payload["candidates"])
    slowest = min(c["metrics"]["tokens_per_s"] for c in payload["candidates"])
    biggest = max(c["silicon"]["total_silicon_mm2"] for c in payload["candidates"])
    smallest = min(c["silicon"]["total_silicon_mm2"] for c in payload["candidates"])
    table.boolean(
        "%s: 2x the capacity is FLAT in the headline (D27)" % name,
        "silicon spans ~2x while tokens/s spans under 2% (Invariant W)",
        biggest / smallest > 1.9 and 1.0 < fastest / slowest < 1.02,
        "silicon x%.3f, headline x%.4f" % (biggest / smallest, fastest / slowest),
    )
    table.boolean(
        "%s: every candidate derives an ANALOG-BOUND engine (ADJ-9)" % name,
        "derived_digital.sizing_target == analog_stage_time and every stage is bound",
        all(
            c["derived_digital"]["sizing_target"] == "analog_stage_time"
            and c["derived_digital"]["analog_bound_stages"]
            == c["derived_digital"]["stages_sized"]
            for c in payload["candidates"]
        ),
        "%s stage(s) analog-bound on every point"
        % sorted({c["derived_digital"]["stages_sized"] for c in payload["candidates"]}),
    )

    selected = payload["selected"]
    table.exact(
        "%s: selected knobs" % name,
        {"arrays_per_chip": 604, "bank_depth": 1},
        selected["knobs"],
    )
    table.boolean(
        "%s: selection sits on the Pareto front" % name,
        "selected_id in front_ids",
        payload["selected_id"] in payload["front_ids"],
        "front=%s (%s)" % (payload["front_ids"], payload["front_shape"]),
    )
    # WAVE F REWRITE (D32, P7.8). OLD CLAIM: the shape is "flat_area" because
    # every candidate carries the same silicon. That was true while the shared
    # digital card declared area_mm2: 0 — Wave D's own note said so, and P7.6
    # named the missing field. D32 composes the chiplet area from measured
    # synthesis blocks, so a wider engine costs real mm2 and the front SPREADS.
    # The row still checks that the shape is READ OFF THE DATA, which is what it
    # was always for; it just no longer hard-codes which answer the data gives.
    areas = {round(c["silicon"]["total_silicon_mm2"], 9) for c in payload["candidates"]}
    table.boolean(
        "%s: the front's shape is read off the data" % name,
        "spread when the candidates carry different silicon, flat_area when they do not",
        (payload["front_shape"] == "spread" and len(areas) > 1)
        or (payload["front_shape"] == "flat_area" and len(areas) == 1),
        "%s over %d distinct silicon values" % (payload["front_shape"], len(areas)),
    )
    # ADJ-9 REWRITE. OLD CLAIM: the silicon axis moves with the DIGITAL term
    # only, because the analog inventory was constant across a lane sweep. NEW
    # CLAIM: the sweep's own area axis is CHIP CAPACITY, so the ANALOG term is
    # exactly what moves along it — enumerated-but-unowned slots, which is what
    # Invariant W says capacity buys. The digital term is what moves along the
    # BANK-DEPTH axis, because the derived width follows the analog m-pass.
    at_capacity = [
        c for c in payload["candidates"] if c["knobs"]["arrays_per_chip"] == 604
    ]
    table.boolean(
        "%s: capacity moves the ANALOG term, banking moves the DIGITAL one" % name,
        "analog mm2 constant within a capacity and rising across them; the "
        "digital mm2 is what the two bank depths differ by",
        len({round(c["silicon"]["analog_macro_silicon_mm2"], 9) for c in at_capacity}) == 1
        and len(
            {
                round(c["silicon"]["analog_macro_silicon_mm2"], 9)
                for c in payload["candidates"]
            }
        )
        == 4
        and len({round(c["silicon"]["shared_digital_silicon_mm2"], 9) for c in at_capacity})
        == 2
        and all(
            c["silicon"]["shared_digital_area_provenance"] == "composed-measured"
            for c in payload["candidates"]
        ),
        "4 analog terms over 4 capacities; 2 digital terms over 2 bank depths",
    )
    table.boolean(
        "%s: infeasible candidates recorded, not dropped" % name,
        "len(candidates) == num_candidates",
        len(payload["candidates"]) == payload["num_candidates"],
        "%d rows" % len(payload["candidates"]),
    )

    # WAVE D AUDIT: the front's NOTE must be derived from these candidates,
    # not a constant string. The two counts the silicon accounting multiplies
    # are what is actually constant here; the TILE count is not, so a note
    # crediting the flat front to the same tiles being placed either way
    # would be refuted by the table above it in its own report.
    valid_rows = [c for c in payload["candidates"] if c["ok"]]
    note = payload["front_note"]
    slots = {c["silicon"]["analog_macro_slots"] for c in valid_rows}
    chiplets = {c["placement"]["shared_digital_chiplets"] for c in valid_rows}
    # WAVE F REWRITE (D32, P7.8): the slot-count / chiplet-count clause belongs
    # to the FLAT_AREA note, which is the only one Wave D could produce. The
    # note must still be DERIVED from these rows whatever shape they take, so
    # each shape is checked against its own claim.
    if payload["front_shape"] == "flat_area":
        note_ok = (
            len(slots) == 1
            and len(chiplets) == 1
            and ("slot count is %d" % list(slots)[0]) in note
            and ("chiplet count is %d" % list(chiplets)[0]) in note
        )
        note_says = "slots=%s chiplets=%s" % (sorted(slots), sorted(chiplets))
    else:
        corners = {
            (
                round(
                    [c for c in valid_rows if c["id"] == pid][0]["metrics"]["tokens_per_s"],
                    12,
                ),
                round(
                    [c for c in valid_rows if c["id"] == pid][0]["silicon"]["total_silicon_mm2"],
                    12,
                ),
            )
            for pid in payload["front_ids"]
        }
        note_ok = (
            len(corners) > 1
            and ("%d non-dominated points" % len(payload["front_ids"])) in note
            and ("%d distinct" % len(corners)) in note
        )
        note_says = "%d front points over %d corners" % (
            len(payload["front_ids"]),
            len(corners),
        )
    table.boolean(
        "%s: the front's note is DERIVED from its own shape" % name,
        "flat_area names the constant slot and chiplet counts; spread names the "
        "non-dominated points and the distinct corners it actually has",
        note_ok,
        note_says,
    )
    tile_counts = {c["placement"]["tiles"] for c in valid_rows}
    table.boolean(
        "%s: the front's note claims no mechanism the rows refute" % name,
        "tiles differ across candidates, so the note must not claim they do not",
        len(tile_counts) > 1
        and "same tiles are placed either way" not in note
        and "the model's weights" not in note,
        "tiles=%s" % sorted(tile_counts),
    )

    # WAVE D AUDIT: what banking actually moves. ADJ-4 prices ACTIVE column
    # sets; the STRANDED columns do not move at all, so any account of the
    # banking win that credits wasted columns is wrong.
    #
    # P7.9 REWRITE. The old row also asserted the MACRO COUNT is identical
    # across the sweep. That was a fact about the DEDICATED law, where every
    # tensor rounds up to its own whole macros; this sweep now declares D27's
    # DENSE law, under which a finer bank really does recover macros (4828 at
    # bank_depth 1 against 5488 at 2). The claim that matters is unchanged and
    # is the one kept here: the stranded columns and the enumerated SLOT count
    # — the silicon a point buys — do not move, so the ENERGY win belongs to
    # the active-column-set law and not to recovered waste.
    # ADJ-9 REWRITE. OLD SCOPE: "across the sweep", which held while the sweep's
    # other axis was the lane width. The other axis is now chip capacity, and
    # enumerating slots is exactly what it does — so the claim is scoped to one
    # capacity, where it says the same thing about BANKING that it always did.
    at_one_capacity = [c for c in valid_rows if c["knobs"]["arrays_per_chip"] == 604]
    table.boolean(
        "%s: unowned columns and enumerated slots are IDENTICAL across the bank depths" % name,
        "banking changes active column sets, not stranded ones (ADJ-4); the "
        "macro count DOES move, because D27's dense law is in force",
        len({c["placement"]["unowned_columns"] for c in at_one_capacity}) == 1
        and len({c["placement"]["analog_macro_slots"] for c in at_one_capacity}) == 1
        and len({c["placement"]["analog_macro_slots"] for c in valid_rows}) == 4,
        "unowned=%s slots=%s macros=%s"
        % (
            sorted({c["placement"]["unowned_columns"] for c in at_one_capacity}),
            sorted({c["placement"]["analog_macro_slots"] for c in at_one_capacity}),
            sorted({c["placement"]["macros_holding_tiles"] for c in valid_rows}),
        ),
    )

    def _component(candidate, key):
        for entry in candidate["metrics"]["energy_components"]:
            if entry["component"] == key:
                return entry["energy_pj"]
        return None

    bank1 = [c for c in at_one_capacity if c["knobs"]["bank_depth"] == 1]
    bank2 = [c for c in at_one_capacity if c["knobs"]["bank_depth"] == 2]
    if bank1 and bank2:
        total_delta = bank2[0]["metrics"]["total_energy_pj"] - bank1[0]["metrics"]["total_energy_pj"]
        array_delta = _component(bank2[0], "analog_arrays") - _component(bank1[0], "analog_arrays")
        engine_delta = _component(bank2[0], "shared_digital_chiplet") - _component(
            bank1[0], "shared_digital_chiplet"
        )
        # ADJ-9 REWRITE. OLD CLAIM: the WHOLE banking energy delta is the
        # analog arrays. NEW CLAIM: it is two terms of opposite sign. A finer
        # bank shortens the analog m-pass, and under D31-v2 that m-pass IS the
        # engine's sizing target, so bank_depth 1 derives a WIDER engine whose
        # composed power is higher. The analog term is still the dominant half
        # and the links still do not move.
        table.boolean(
            "%s: the banking energy delta is analog_arrays PLUS the derived engine" % name,
            "total delta == analog_arrays delta + shared_digital delta; "
            "link_traffic unmoved; the analog half dominates",
            abs(total_delta - (array_delta + engine_delta)) < 1e-3
            and _component(bank1[0], "link_traffic") == _component(bank2[0], "link_traffic")
            and abs(engine_delta) < 0.1 * abs(array_delta),
            "total %.6g pJ = arrays %.6g pJ + engine %.6g pJ (lanes %d vs %d)"
            % (
                total_delta,
                array_delta,
                engine_delta,
                bank1[0]["derived_digital"]["vector_lanes"],
                bank2[0]["derived_digital"]["vector_lanes"],
            ),
        )

    # WAVE D AUDIT: D21 wants the relaxation disclosed in the ARTIFACT, and
    # the MD is the artifact a human reads.
    # WAVE F REWRITE (D32, P7.8). OLD CLAIM: the accounting always has an
    # uncovered term to name, because the digital card declared no area law.
    # NEW CLAIM: the accounting always states its PROVENANCE — an uncovered
    # term is named as an ABSENT law, a covered one names the measured library
    # it was composed from. A silent number is what stays forbidden, in either
    # direction.
    coverage = payload.get("silicon_coverage") or {}
    uncovered = coverage.get("uncovered_terms") or []
    table.boolean(
        "%s: the silicon accounting states its provenance" % name,
        "uncovered terms are named as an absent law; a covered term names the "
        "measured library it was COMPOSED from (D32)",
        bool(uncovered)
        or ("COMPOSED area" in coverage.get("basis", "") and "D32" in coverage.get("basis", "")),
        "; ".join(uncovered) or "covered: composed from the measured 22nm library",
    )
    demo_md = os.path.join(os.path.dirname(QIF_DSE_DEMO), "dse_report.md")
    if os.path.exists(demo_md):
        with open(demo_md) as handle:
            md_text = handle.read()
        if uncovered:
            md_ok = (
                "silicon_accounting" in md_text
                and "silicon_uncovered" in md_text
                and all(term in md_text for term in uncovered)
            )
        else:
            md_ok = (
                "silicon_accounting" in md_text
                and "silicon_uncovered" not in md_text
                and "COMPOSED area" in md_text
            )
        table.boolean(
            "%s: the MD carries the coverage, not only the JSON" % name,
            "the Disclosures section states the accounting and its provenance",
            md_ok,
            "%d chars" % len(md_text),
        )

    # WAVE D AUDIT: one constraint, one row — per candidate and for the sweep.
    dup_free = all(
        len({d["constraint"] for d in c["disclosures"]}) == len(c["disclosures"])
        for c in payload["candidates"]
    )
    union_keys = [d["constraint"] for d in payload["disclosures"]]
    table.boolean(
        "%s: disclosures are one entry per constraint" % name,
        "no repeated constraint key on a candidate or in the sweep union",
        dup_free and len(union_keys) == len(set(union_keys)),
        "%d union rows" % len(union_keys),
    )
    # A constraint whose value DIFFERS between candidates is recorded beside
    # the union entry, never merged into it: per-candidate utilization (P7.3,
    # D28) is exactly such a constraint, because provisioning is what the
    # sweep moves. What this row checks is that the disagreement is FLAGGED
    # and itemized — every differing candidate named, with its own value and
    # reason — which is the guarantee. (Before P7.3 nothing in this sweep
    # disagreed, and this row asserted the absence of disagreement instead of
    # its disclosure; that was a fact about the demo sweep, not a law.)
    variants_named = all(
        variant.get("id") and variant.get("value") and variant.get("reason")
        for d in payload["disclosures"]
        for variant in d.get("varies_by_candidate", ())
    )
    table.boolean(
        "%s: every sweep disclosure names the candidates that carried it" % name,
        "each union entry lists its candidates and flags any disagreement",
        all(d.get("candidates") for d in payload["disclosures"]) and variants_named,
        "%d rows labelled, %d with a flagged disagreement"
        % (
            len(payload["disclosures"]),
            sum(1 for d in payload["disclosures"] if d.get("varies_by_candidate")),
        ),
    )

    # The emitted machine IS the selection.
    if table.boolean(
        "%s: emitted config exists" % name,
        "configs/hardware-config/fws_cim_granite_tiny_dse_selected.yaml",
        os.path.exists(QIF_DSE_DEMO_SELECTED),
    ):
        emitted = load_yaml(QIF_DSE_DEMO_SELECTED)
        # ADJ-9 REWRITE. OLD ROW: the emitted card declares vector_lanes = 4096.
        # NEW ROW: it declares NO lane count at all — the winning point has no
        # width to write down, because D31-v2 derives it from the machine's own
        # analog m-pass every time it runs.
        table.boolean(
            "%s: emitted config declares NO vector_lanes (D31/ADJ-9)" % name,
            "the winner carries no engine width; the width is derived at run time",
            "vector_lanes" not in emitted["cim"]["cards"]["sa"],
            "card keys: %s" % sorted(emitted["cim"]["cards"]["sa"]),
        )
        table.exact(
            "%s: emitted arrays_per_chip" % name,
            604,
            emitted["cim"]["chip"]["arrays_per_chip"],
        )
        table.exact(
            "%s: emitted bank_depth" % name, 1, emitted["cim"]["cards"]["ctt"]["bank_depth"]
        )
        table.boolean(
            "%s: emitted config carries no sweep block" % name,
            "mapping_dse stripped (a machine, not a candidate space)",
            "mapping_dse" not in emitted,
        )
        table.boolean(
            "%s: emitted config is a MAPPED config" % name,
            "mapping: block present (ADJ-6 DAG report, not the closed form)",
            "mapping" in emitted,
        )


def check_qif_dse_retired(table):
    """(P3.7 c, ADJ-9) The FROZEN Wave-D sweep, kept because it is evidence.

    ADJ-9 was adjudicated on this artifact: declaring 512 -> 4096 scan lanes on
    the shipped Granite machine bought +27% tokens/s for +0.78% silicon, which
    is the measurement that says digital lanes are nearly free against the
    Invariant-W analog floor. Its config declared `vector_lanes` as a sweep
    axis and tools/fws_qif_dse.py now REFUSES that axis by name, so it can
    never be regenerated. A frozen artifact with no label rots silently; these
    rows are the label, checked.
    """
    name = "P3.7 retired sweep"
    if not table.boolean(
        "%s: the frozen artifact exists" % name,
        "docs/qif/dse/granite_lanes_banks_retired/dse_report.json",
        os.path.exists(QIF_DSE_DEMO_RETIRED),
    ):
        return
    with open(QIF_DSE_DEMO_RETIRED) as handle:
        payload = json.load(handle)
    retired = payload.get("retired") or {}
    table.boolean(
        "%s: it declares itself FROZEN and names the decision" % name,
        "retired.decision names ADJ-9; retired.status says why it cannot regenerate",
        str(retired.get("decision", "")).startswith("ADJ-9")
        and "FROZEN" in str(retired.get("status", ""))
        and "REFUSES that axis by name" in str(retired.get("status", "")),
        str(retired.get("decision", "MISSING")),
    )
    table.boolean(
        "%s: it names its successor and how to read it" % name,
        "retired.successor points at the D31-legal sweep; retired.read_it_as "
        "says the digital side was chosen, not derived",
        "granite_capacity_banks" in str(retired.get("successor", ""))
        and "DERIVED" in str(retired.get("read_it_as", "")),
        str(retired.get("successor", "MISSING")),
    )
    table.boolean(
        "%s: the axis it swept is one the tool now refuses" % name,
        "SweepSpec.from_raw raises on this sweep block",
        _refuses_retired_axes(payload["sweep"]),
        "axes: %s" % ", ".join(sorted(payload["sweep"]["axes"])),
    )
    at_depth_one = sorted(
        (c for c in payload["candidates"] if c["knobs"]["bank_depth"] == 1),
        key=lambda c: c["knobs"]["vector_lanes"],
    )
    rates = [c["metrics"]["tokens_per_s"] for c in at_depth_one]
    silicon = [c["silicon"]["total_silicon_mm2"] for c in at_depth_one]
    table.boolean(
        "%s: the comparison ADJ-9 quotes is still readable off it" % name,
        "8x the declared lanes: ~+27% tokens/s for ~+0.8% silicon",
        len(rates) == 4
        and 1.25 < rates[-1] / rates[0] < 1.30
        and 1.005 < silicon[-1] / silicon[0] < 1.010,
        "%.0f -> %.0f tokens/s (x%.3f) for %.1f -> %.1f mm2 (x%.4f)"
        % (
            rates[0],
            rates[-1],
            rates[-1] / rates[0],
            silicon[0],
            silicon[-1],
            silicon[-1] / silicon[0],
        ),
    )


def _refuses_retired_axes(sweep_block):
    """True when tools/fws_qif_dse.py refuses this sweep block by name."""
    module = _load_qif_dse_module()
    try:
        module.SweepSpec.from_raw({module.DSE_BLOCK: sweep_block}, "retired")
    except module.QifDseUsageError:
        return True
    return False


def _load_qif_dse_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fws_qif_dse_validator", QIF_DSE_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_qif_dse_verify(table):
    """(P3.7 b) A LIVE mapped sweep with --emit-config --verify.

    Llama2-7B on the mapped config: bank_depth {4, 1} x arrays_per_chip
    {120, 8}. The 8-slot chip cannot hold a layer, so two candidates die in
    the MAPPER and carry its own message; the two that survive are ranked,
    and the winner is re-run through run_perf from the emitted YAML.
    """
    name = "P3.7 verify"
    out_dir = os.path.join(QIF_DSE_OUT_ROOT, "validation_llama7b_mapped")
    os.makedirs(out_dir, exist_ok=True)
    hw_dict = load_yaml(os.path.join(HW_DIR, "fws_cim_llama7b_mapped.yaml"))
    # Two declarations the shipped file does not carry: the CARDS a card axis
    # needs (the shipped config synthesizes them, and the tool refuses to
    # invent a card block), and an activation tier that holds this workload —
    # the shipped 8 GB tier is VIOLATED by P4.3 on chips 0-3, which the sweep
    # would (correctly) record as infeasible.
    hw_dict["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"},
        "sa": {"kind": "digital_chiplet", "vector_lanes": 1024},
    }
    hw_dict["tech_param"]["DRAM"]["size"] = "32 GB"
    hw_dict["mapping_dse"] = {
        "label": "P3.7 validator: banking x chip slot budget",
        "objective": "throughput",
        "axes": {"bank_depth": [4, 1], "arrays_per_chip": [120, 8]},
    }
    hw_path = os.path.join(out_dir, "fws_cim_llama7b_mapped_dse.yaml")
    with open(hw_path, "w") as handle:
        yaml.safe_dump(hw_dict, handle, sort_keys=False)
    emitted_path = os.path.join(out_dir, "selected_config.yaml")
    proc, payload = _run_qif_dse_tool(
        hw_path,
        "llama2_7b_fws_inf.yaml",
        out_dir,
        extra_args=["--emit-config", emitted_path, "--verify"],
    )
    if not table.boolean(
        "%s: dse exit code" % name, "0", proc.returncode == 0, str(proc.returncode)
    ):
        print(proc.stdout[-3000:])
        return
    if not table.boolean(
        "%s: dse_report.json written" % name, "exists", payload is not None
    ):
        return

    table.exact("%s: candidates swept" % name, 4, payload["num_candidates"])
    table.exact("%s: valid candidates" % name, 2, payload["num_valid"])
    failed = [c for c in payload["candidates"] if not c["ok"]]
    table.boolean(
        "%s: every infeasible candidate carries a stage + message" % name,
        "stage tag and a non-empty message on each",
        len(failed) == 2
        and all(c["fail_stage"] == "mapping" and c["fail_message"] for c in failed),
        ", ".join("%s[%s]" % (c["id"], c["fail_stage"]) for c in failed),
    )
    selected = payload["selected"]
    table.exact(
        "%s: selected knobs" % name,
        {"bank_depth": 1, "arrays_per_chip": 120},
        selected["knobs"],
    )
    table.boolean(
        "%s: selection sits on the Pareto front" % name,
        "selected_id in front_ids",
        payload["selected_id"] in payload["front_ids"],
        "front=%s" % payload["front_ids"],
    )

    verify = payload.get("verify")
    if not table.boolean(
        "%s: --verify ran" % name, "verify block present", verify is not None
    ):
        return
    table.boolean(
        "%s: emitted config reproduces the selection" % name,
        "every check within 0.1% (counts exact)",
        verify["pass"],
        verify.get("message") or "PASS",
    )
    by_name = {check["name"]: check for check in verify["checks"]}
    for metric in ("tokens_per_s", "prefill_latency"):
        check = by_name.get(metric)
        if check is None:
            table.boolean("%s: %s round trip" % (name, metric), "checked", False, "absent")
            continue
        table.close(
            "%s: %s == run_perf" % (name, metric), check["simulator"], check["dse"]
        )
    for count in ("analog_chips", "tiles", "analog_macro_slots"):
        check = by_name.get(count)
        table.exact(
            "%s: %s == run_perf (exact)" % (name, count),
            None if check is None else check["simulator"],
            None if check is None else check["dse"],
        )
    table.boolean(
        "%s: the verify run wrote the MAPPED report" % name,
        "output/LLM/fws_qif_report.json under the sweep's own output dir",
        os.path.exists(os.path.join(out_dir, "output", "LLM", "fws_qif_report.json")),
    )


def check_qif_synthesis_library(table):
    """QIF P7.8 rows: the measured synthesis library (D32) and derived sizing (D31).

    In-process, on the ADJ-1 headline model, and deliberately after every row
    that reads an artifact. Two questions: is the digital silicon a COMPOSITION
    of measured blocks (and does it say what it does not cover), and is the
    engine width DERIVED from the beat rather than declared (and is the derived
    width the smallest one that holds it)?
    """
    name = "P7.8 library"
    sys.path.insert(0, REPO_ROOT)
    import cim_timing as _cim_timing
    import config as _config
    import fws_eval as _fws_eval
    import fws_mapping as _fws_mapping
    from program.fws_build import build_fws_program as _build_fws_program

    lib = _cim_timing.SynthesisLibrary.load("22nm")
    optima_lib = os.path.join(
        "/app/nanocad/projects/cim_ctt_big_optima/perf_model/configs",
        "digital_hw_components_22nm.yaml",
    )
    table.exact(
        "%s: the checked-in library names its synthesis source" % name,
        (optima_lib, True),
        (lib.source, "reports_22nm" in lib.source_reports),
    )
    table.boolean(
        "%s: every block carries a measured provenance" % name,
        "area_um2 + power_W + 'measured via synthesis, 22nm' on all 13 blocks",
        len(lib.blocks) == 13
        and all(
            block.provenance == "measured via synthesis, 22nm"
            and block.area_um2 >= 0
            and block.power_w >= 0
            for block in lib.blocks.values()
        ),
        "%d blocks" % len(lib.blocks),
    )
    refused = ""
    try:
        lib.block("NO_SUCH_BLOCK")
        refused = "a block the library does not name was ANSWERED"
    except _cim_timing.SynthesisLibraryError as exc:
        if "NO_SUCH_BLOCK" not in str(exc) or "no default block" not in str(exc):
            refused = "the refusal does not name the block and the no-default rule"
    table.boolean(
        "%s: an unnamed block is refused BY NAME (D32/ADJ-4)" % name,
        "no default block, no substitute, no zero-fill",
        not refused,
        refused or "refused, naming the block and the library",
    )

    # The compositions, checked against the same hand arithmetic the tests use.
    lane_um2 = (
        lib.block("FP_MULT").area_um2
        + lib.block("FP_ADD").area_um2
        + lib.block("M_REG").area_um2
    )
    table.close(
        "%s: 5479 scan lanes == 5479 x (FP_MULT + FP_ADD + M_REG)" % name,
        5479 * lane_um2 / 1e6,
        _cim_timing.compose_vector_engine(lib, 5479).area_mm2,
    )
    fabric = _cim_timing.compose_sa_fabric(lib, 32, 64, 2, 1)
    table.exact(
        "%s: a 32x64 x2-array fabric is 4 measured GEMMINI blocks" % name,
        {"GEMMINI_SYS_ARRAY": 4, "TRANSPOSER": 1},
        dict(fabric.blocks),
    )
    softmax = _cim_timing.compose_softmax_engine(lib, 4, 2)
    table.exact(
        "%s: the softmax census is OPTIMA's, term for term" % name,
        {"FP_COMP": 6, "FP_ADD": 14, "FP_MULT": 16, "BF16_EXP": 8, "BF16_RECIP": 2},
        dict(softmax.blocks),
    )
    table.boolean(
        "%s: OPTIMA's overhead pads are recorded and NOT applied (D28)" % name,
        "composed area == the bare sum of count x measured area",
        abs(
            softmax.area_mm2
            - sum(unit.count * unit.unit_area_mm2 for unit in softmax.units)
        )
        <= 1e-15
        and _cim_timing.OPTIMA_COLLECTION_OVERHEADS["Softmax_Stage2"] == 0.2,
        "the 0.2 softmax pad is named and dropped",
    )

    # The derivation, on the headline model's own run.
    hw_path = os.path.join(HW_DIR, "fws_cim_granite_tiny.yaml")
    model_path = os.path.join(MODEL_DIR, "granite_4_0_h_tiny_inf.yaml")
    hw_raw = load_yaml(hw_path)
    _config.convert(hw_raw)
    hw = _config.HWConfig.from_dict(hw_raw)
    model = _config.parse_config(model_path, "LLM")
    mapping = _fws_mapping.build_mapping(hw, model)
    evaluation = _fws_eval.evaluate_fws(_build_fws_program(mapping), mapping)
    device = mapping.device
    sizing = device.derived_engine

    table.boolean(
        "%s: the headline card DECLARES no engine, and one is DERIVED (D31)" % name,
        "cim.cards.sa carries no vector_lanes; the width comes from the analog floor",
        device.digital_card.has_vector_engine is False
        and sizing is not None
        and device.vector_lanes_provenance == _cim_timing.PROVENANCE_DERIVED_COUNT,
        "vector_lanes = %s (%s)"
        % (device.vector_lanes, device.vector_lanes_provenance),
    )
    binding = [row for row in sizing.per_stage if row.stage == sizing.binding_stage][0]
    # ADJ-9 REWRITE. OLD ROW: "the derived width HOLDS the analog BEAT" — the
    # binding stage's vector time fits inside the beat the slowest stage sets.
    # NEW ROW: it holds that stage's OWN ANALOG M-PASS, which is 24x tighter on
    # this machine, and EVERY stage does. That is the fixed point D31-v2 asks
    # for, asserted rather than described.
    table.boolean(
        "%s: EVERY stage is ANALOG-BOUND by construction (ADJ-9)" % name,
        "digital per-stage time <= that stage's own analog m-pass, on every stage",
        len(sizing.analog_bound_stages) == len(sizing.per_stage)
        and not sizing.unreachable_stages
        and all(row.time_s <= row.analog_time_s for row in sizing.per_stage),
        "%d of %d stage(s); binding stage %.6g s of digital in a %.6g s analog m-pass"
        % (
            len(sizing.analog_bound_stages),
            len(sizing.per_stage),
            binding.time_s,
            binding.analog_time_s,
        ),
    )
    table.boolean(
        "%s: and it is the SMALLEST width that does (no margin, D28)" % name,
        "one lane fewer overruns the analog m-pass on the binding stage",
        _cim_timing.vector_cycles_at(
            binding.ops, sizing.vector_lanes - 1, sizing.pipeline_depth
        )
        > binding.budget_cycles,
        "%d lanes; %d would not fit" % (sizing.vector_lanes, sizing.vector_lanes - 1),
    )
    # THE OTHER HALF OF ADJ-9, AND IT IS A FINDING: the criterion binds every
    # STAGE and cannot bind the MACHINE. The beat-setting stage's largest term
    # is the attention systolic fabric, whose geometry is DECLARED card
    # geometry that D31 derives no width for, so scan lanes past the point
    # where the scan fits under it buy area and no throughput. Measured here,
    # not argued.
    setter = sizing.beat_setting_stage or {}
    largest = (setter.get("terms") or [{}])[0]
    table.boolean(
        "%s: the beat-setter is NAMED, and on this machine it is not analog" % name,
        "derived_engine_sizing.beat_setting_stage itemizes the longest stage's terms",
        bool(setter)
        and setter.get("analog_is_largest_term") is False
        and largest.get("device_class") == "shared_digital"
        and float(largest.get("busy_s", 0.0)) > 10 * float(setter.get("analog_time_s", 1.0)),
        "stage %s spends %.6g s on %s/%s against %.6g s of analog m-pass"
        % (
            setter.get("stage"),
            float(largest.get("busy_s", 0.0)),
            largest.get("device_class"),
            largest.get("block"),
            float(setter.get("analog_time_s", 0.0)),
        ),
    )
    table.boolean(
        "%s: the reachability finding rides the ARTIFACT, not only stdout" % name,
        "an analog_floor_reachability disclosure names the stage and the term",
        any(
            item.constraint == "analog_floor_reachability"
            for item in evaluation.disclosures
        ),
        "disclosed",
    )
    table.exact(
        "%s: the derivation is the pricing law inverted" % name,
        _cim_timing.vector_cycles_at(
            binding.ops, sizing.vector_lanes, sizing.pipeline_depth
        ),
        binding.used_cycles,
    )
    table.boolean(
        "%s: the derived sizing is REPORTED, not only applied (D31)" % name,
        "a derived_engine_sizing disclosure and an evaluation.digital_silicon block",
        any(
            item.constraint == "derived_engine_sizing" for item in evaluation.disclosures
        )
        and _fws_eval.report_document(evaluation)["evaluation"]["digital_silicon"][
            "vector_lanes"
        ]
        == sizing.vector_lanes,
        "%d lanes, binding stage %d" % (sizing.vector_lanes, sizing.binding_stage),
    )

    silicon = _fws_eval.report_document(evaluation)["evaluation"]["digital_silicon"]
    parts = device.shared_digital_compositions()
    table.close(
        "%s: the chiplet area IS the sum of its measured blocks" % name,
        sum(part.area_mm2 for part in parts),
        silicon["shared_digital_chiplet"]["area_mm2_per_chiplet"],
    )
    table.boolean(
        "%s: digital area and power are labelled per component" % name,
        "every composed row names where its COUNT and its per-unit silicon came from",
        all(
            unit["unit_provenance"].startswith("measured via synthesis")
            and unit["count_provenance"]
            in (
                _cim_timing.PROVENANCE_DERIVED_COUNT,
                _cim_timing.PROVENANCE_DECLARED_COUNT,
            )
            for engine in silicon["shared_digital_chiplet"]["engines"]
            for unit in engine["units"]
        ),
        "%d engines composed" % len(silicon["shared_digital_chiplet"]["engines"]),
    )
    table.boolean(
        "%s: the composed area says what it does NOT cover (D21/D28)" % name,
        "activation SRAM / interconnect / control are named as absent, not padded",
        any("LOWER BOUND" in note for note in silicon["disclosures"])
        and any("activation SRAM" in note for note in silicon["disclosures"]),
    )
    digital_energy = [
        component
        for component in evaluation.energy
        if component.key == "shared_digital_chiplet"
    ][0]
    table.boolean(
        "%s: shared-digital ENERGY stopped being uncovered (D32)" % name,
        "priced from the composed engine's measured power, labelled partial "
        "because the library reports one average power and no leakage split",
        digital_energy.energy_pj > 0 and digital_energy.coverage == "partial",
        "%.6g pJ, %s" % (digital_energy.energy_pj, digital_energy.coverage),
    )


def check_qif_filled_pipeline(table):
    """QIF P7.7 rows: the FILLED PIPELINE serving regime (D29/D30).

    In-process, on the ADJ-1 headline model. Every row is a reading off the one
    timeline or a refusal the surface makes by name; nothing here re-derives a
    number the report already owns.
    """
    name = "P7.7 pipeline"
    sys.path.insert(0, REPO_ROOT)
    import config as _config
    import fws_eval as _fws_eval
    import fws_mapping as _fws_mapping
    from program.fws_build import (
        LAW_ANALOG_GEMM as _LAW_ANALOG_GEMM,
        annotations_of as _annotations_of,
        build_fws_program as _build_fws_program,
    )

    hw_path = os.path.join(HW_DIR, "fws_cim_granite_tiny.yaml")
    model_path = os.path.join(MODEL_DIR, "granite_4_0_h_tiny_inf.yaml")
    hw_raw = load_yaml(hw_path)
    _config.convert(hw_raw)
    hw = _config.HWConfig.from_dict(hw_raw)
    model = _config.parse_config(model_path, "LLM")
    mapping = _fws_mapping.build_mapping(hw, model)
    evaluation = _fws_eval.evaluate_fws(_build_fws_program(mapping), mapping)
    pipeline = evaluation.pipeline
    residency = evaluation.state_residency
    metrics = {metric.key: metric.value for metric in evaluation.metrics}

    table.exact(
        "%s: a mapped config is a filled pipeline (D29)" % name,
        _fws_mapping.REGIME_FILLED,
        mapping.regime,
    )
    table.exact(
        "%s: D = the stage count = the analog chips" % name,
        (len(mapping.analog_chips()), len(mapping.analog_chips())),
        (mapping.resident_streams, len(mapping.stages)),
    )
    beat = float(pipeline["beat_s"])
    table.close(
        "%s: tokens/s == 1 / beat" % name, 1.0 / beat, metrics["sys.fws.tokens_per_s"]
    )
    table.close(
        "%s: per-stream rate == 1 / (D x beat)" % name,
        1.0 / (beat * mapping.resident_streams),
        metrics["sys.fws.per_stream_tokens_per_s"],
    )
    table.exact(
        "%s: one token exits per beat over the steady window" % name,
        # steady + 2 traversals complete: the extra one is what makes the
        # transient interval holdable-out and still leaves `steady` of them.
        int(evaluation.serving.steady_beats) + 2,
        int(pipeline["tokens_exited"]),
    )
    intervals = list(pipeline["steady_beat_intervals_s"])
    spread = (
        max(abs(value - beat) for value in intervals) / beat if intervals else 1.0
    )
    tail = (
        abs(intervals[-1] - intervals[-2]) / beat if len(intervals) >= 2 else 1.0
    )
    # WAVE F REWRITE (D31, P7.8). OLD CLAIM: "every STEADY exit interval IS the
    # beat" to 1e-3. That held while the digital side was fast enough to make
    # every stage's service time nearly equal — Granite's card DECLARED 1024
    # scan lanes. D31 derives the width instead (220 lanes), the digital half of
    # a stage becomes comparable to the analog half, and the UNEVEN stage plan
    # (4-layer and 3-layer stages) then shows up as a longer ramp: the fill
    # allowance of D - 1 beats is exact only for equal stages. NEW CLAIM: the
    # exits CONVERGE — they approach the beat monotonically and the last two
    # agree to 1e-3 — and a run whose sample still contains ramp SAYS SO by
    # name. That is a check on the measurement, not a tolerance widened to fit.
    monotone = all(
        intervals[i] <= intervals[i + 1] * (1 + 1e-12)
        for i in range(len(intervals) - 1)
    ) or all(
        intervals[i] >= intervals[i + 1] * (1 - 1e-12)
        for i in range(len(intervals) - 1)
    )
    table.boolean(
        "%s: the exits CONVERGE to the beat inside the window" % name,
        "the steady intervals approach the beat monotonically and the last two agree "
        "to <= 1e-3; the fill allowance is D - 1 beats, which is exact only when every "
        "stage has the same service time",
        bool(intervals) and monotone and tail <= 1e-3,
        "%d steady intervals, %d held out as the fill transient, spread %.2e, last two "
        "agree to %.2e" % (
            len(intervals),
            len(pipeline["transient_beat_intervals_s"]),
            spread,
            tail,
        ),
    )
    # P7.9 REWRITE. OLD CLAIM: beat_converged is true exactly when the WHOLE
    # steady sample is within 1e-3, and a wider spread rides a
    # pipeline_beat_still_converging disclosure. That pairing existed because
    # the beat was the median of the whole sample, ramp included, which made
    # the headline move 12% with the window size. NEW CLAIM: the beat is the
    # median of the SETTLED TAIL, converged means that tail holds at least two
    # intervals that agree, and every interval dropped as ramp is printed and
    # rides beat_read_from_the_converged_tail. The spread of the whole sample is
    # still reported and is still allowed to be large — it is the ramp, named.
    ramp = list(pipeline["ramp_beat_intervals_s"])
    settled = list(pipeline["converged_tail_intervals_s"])
    table.boolean(
        "%s: the beat is the SETTLED TAIL and every dropped ramp says so BY NAME" % name,
        "beat_converged means >= 2 agreeing intervals; a dropped ramp interval rides a "
        "beat_read_from_the_converged_tail disclosure and is printed in the pipeline block",
        bool(pipeline["beat_converged"]) == (len(settled) >= 2)
        and settled == list(pipeline["steady_beat_intervals_s"])[len(ramp):]
        and bool(ramp)
        == any(
            item.constraint == "beat_read_from_the_converged_tail"
            for item in evaluation.disclosures
        ),
        "converged=%s over %d settled interval(s), %d dropped as ramp, whole-sample "
        "spread %.2e" % (pipeline["beat_converged"], len(settled), len(ramp), spread),
    )
    annotations = _annotations_of(evaluation.program)
    analog_m = {a.tokens for a in annotations if a.law == _LAW_ANALOG_GEMM}
    table.exact("%s: every analog op fires at M = 1" % name, {1.0}, analog_m)
    table.boolean(
        "%s: the streams sit at different contexts in one beat" % name,
        "more than one context inside the last lowered beat (D29)",
        len({a.context for a in annotations if a.beat == evaluation.serving.beats - 1})
        > 1,
    )
    table.exact(
        "%s: every stage holds all D streams (state residency)" % name,
        (True, mapping.resident_streams, len(mapping.stages)),
        (
            bool(residency["measured"]),
            int(residency["resident_streams"]),
            int(residency["stages"]),
        ),
    )
    table.close(
        "%s: total state == D x per-stream model state" % name,
        residency["total_state_bytes"],
        residency["per_stream_model_state_bytes"] * mapping.resident_streams,
    )
    table.boolean(
        "%s: the state verdict is named" % name,
        "fits / VIOLATED / undeclared, per stage and in total",
        residency["verdict"] in ("fits", "VIOLATED", "undeclared")
        and all(
            row["verdict"] in ("fits", "VIOLATED", "undeclared")
            for row in residency["per_stage"]
        ),
        str(residency["verdict"]),
    )
    table.boolean(
        "%s: D30 leaves no endpoint anywhere" % name,
        "no lm_head owner, tile or op",
        not mapping.endpoint_blocks
        and not [t for t in mapping.tiles if t.owner.op == "lm_head"]
        and not [a for a in annotations if a.block == "lm_head"],
    )

    # The refusals, by name.
    refusals = []
    batched = _config.parse_config(model_path, "LLM")
    batched.model_config.global_batch_size = 4
    try:
        _config.validate_model_config(hw, batched)
        refusals.append("global_batch_size NOT refused")
    except ValueError as exc:
        if "D29" not in str(exc) or "global_batch_size" not in str(exc):
            refusals.append("batch refusal does not name the field and D29")
    endpoints = _config.parse_config(model_path, "LLM")
    endpoints.model_config.disable_embedding_unembedding = False
    try:
        _config.validate_model_config(hw, endpoints)
        refusals.append("endpoints NOT refused")
    except ValueError as exc:
        if "D30" not in str(exc) or "disable_embedding_unembedding" not in str(exc):
            refusals.append("endpoint refusal does not name the field and D30")
    try:
        _config.MappingConfig.from_dict({"batch": 4})
        refusals.append("mapping.batch NOT refused")
    except ValueError as exc:
        if "refused BY NAME (D29)" not in str(exc):
            refusals.append("mapping.batch refusal does not cite D29")
    table.boolean(
        "%s: batch and endpoints are refused BY NAME" % name,
        "model_param.global_batch_size (D29), disable_embedding_unembedding (D30), "
        "mapping.batch (D29)",
        not refusals,
        "; ".join(refusals) or "3 refusals, each naming its field",
    )


def check_qif_service_links(table):
    """(P5/P7) The atlas `svc` wires: one per serviced analog chip, MEASURED.

    The atlas drew the `act` chip boundary and nothing at all for the
    relationship between an analog chip and the shared digital chiplet that
    runs its attention and its scan (D13). The results.html pipeline map needs
    that wire tree, and a wire whose bytes are invented is a picture, not a
    result — so these rows check the byte accounting rather than the drawing.
    """
    name = "P5 svc links"
    for model, filename in (
        ("Granite-4.0-H-Tiny", "granite_4_0_h_tiny.json"),
        ("Qwen3.5-4B", "qwen3_5_4b.json"),
    ):
        path = os.path.join(REPO_ROOT, "docs", "qif", "atlas", filename)
        if not table.boolean(
            "%s %s: the priced atlas exists" % (name, model),
            "docs/qif/atlas/%s" % filename,
            os.path.exists(path),
        ):
            continue
        with open(path) as handle:
            document = json.load(handle)
        links = [row for row in document["links"] if row["role"] == "svc"]
        analog = [c for c in document["chips"] if c["pool"] == "analog"]
        digital = {c["id"] for c in document["chips"] if c["pool"] == "digital"}
        table.exact(
            "%s %s: one svc wire per analog chip" % (name, model),
            len(analog),
            len(links),
        )
        table.boolean(
            "%s %s: every wire runs chiplet -> analog chip" % (name, model),
            "from is a digital chip id, to is an analog chip id, and every "
            "analog chip is reached exactly once",
            bool(links)
            and {row["from"] for row in links} <= digital
            and sorted(row["to"] for row in links)
            == sorted(c["id"] for c in analog),
            "%d wire(s) from %d chiplet(s)"
            % (len(links), len({row["from"] for row in links})),
        )
        table.boolean(
            "%s %s: the bytes are MEASURED per beat and name both directions"
            % (name, model),
            "basis prints the measurement slice, operands out + results back, "
            "and the per-block itemization",
            bool(links)
            and all(
                row["per"] == "beat"
                and row["bytes"] >= 0.0
                and "MEASURED on beat" in row["basis"]
                and "operands out" in row["basis"]
                and "results back" in row["basis"]
                for row in links
            ),
            "bytes/beat: %s"
            % sorted({row["bytes"] for row in links}),
        )
        table.boolean(
            "%s %s: an unpriced component is NAMED, never absorbed" % (name, model),
            "a wire carrying only scan work measures 0 B and its basis says "
            "which block lowers no transfer (D21, D28)",
            bool(links)
            and all(
                "UNPRICED COMPONENT" not in row["basis"]
                or "absent op, not a measured zero" in row["basis"]
                for row in links
            )
            and any("UNPRICED COMPONENT" in row["basis"] for row in links),
            "%d of %d wire(s) name an unpriced component"
            % (
                sum(1 for row in links if "UNPRICED COMPONENT" in row["basis"]),
                len(links),
            ),
        )


def check_qif_frontier(table):
    """QIF P7.4/P7.5 rows: the (area, tokens/s) FRONTIER under regime v2.

    Read, not rerun: `tests/test_qif_frontier.py` re-walks each ladder end to
    end and compares the artifact, so the rows here are about WHAT THE SWEEP
    FOUND — that the walk was a ladder and not a cross product, that every
    point carries D29's own quantities (D, the beat, the per-stage state bill),
    that the silicon accounting closes over its three terms with the digital
    ones composed from measured synthesis, that the front is a front, that the
    knee is the bend in it, and that a point refused for residency is refused
    with its stage and its bytes rather than with a word.
    """
    for model_id, folder in QIF_FRONTIER_CURVES:
        name = "P7.4 frontier %s" % model_id
        path = os.path.join(REPO_ROOT, "docs", "qif", "dse", folder, "dse_report.json")
        if not table.boolean(
            "%s: checked-in curve exists" % name,
            "docs/qif/dse/%s/dse_report.json" % folder,
            os.path.exists(path),
        ):
            continue
        with open(path) as handle:
            payload = json.load(handle)
        sweep = payload["sweep"]
        valid = [c for c in payload["candidates"] if c["ok"]]

        # (1) The walk. A ladder, from a declared initializer, that visited
        # strictly fewer points than the cross product it did NOT enumerate.
        table.exact("%s: search mode" % name, "ladder", sweep["search"])
        table.boolean(
            "%s: the ladder is directional, not a cross product" % name,
            "visited < the declared cross product, and every visit priced in full",
            len(payload["candidates"]) < _cross_product_size(sweep["axes"]),
            "%d visited of %d declared points"
            % (len(payload["candidates"]), _cross_product_size(sweep["axes"])),
        )
        table.boolean(
            "%s: every accepted step moves ONE axis ONE rung" % name,
            "|direction| == 1 on every accepted trail entry",
            all(
                abs(int(entry["direction"])) == 1
                for entry in payload["ladder_trail"]
                if entry.get("accepted") and entry.get("axis")
            ),
            "%d accepted steps" % sum(
                1 for entry in payload["ladder_trail"] if entry.get("accepted")
            ),
        )

        # (2) D31: digital provisioning is NOT an axis, and the engine is
        # derived at every point.
        table.boolean(
            "%s: no refused axis is swept (D31/ADJ-9)" % name,
            "the scan/vector engine is derived per point, never declared",
            not set(sweep["axes"]) & set(sweep.get("refused_axes", {})),
            "axes: %s (refused: %s)"
            % (
                ", ".join(sorted(sweep["axes"])),
                ", ".join(sorted(sweep.get("refused_axes", {}))),
            ),
        )
        table.boolean(
            "%s: every point's engine is sized to the ANALOG FLOOR (ADJ-9)" % name,
            "derived_digital.sizing_target == analog_stage_time and every stage "
            "of every valid point is analog-bound",
            bool(valid)
            and all(
                c["derived_digital"]["sizing_target"] == "analog_stage_time"
                and c["derived_digital"]["analog_bound_stages"]
                == c["derived_digital"]["stages_sized"]
                and c["derived_digital"]["stages_sized"] > 0
                for c in valid
            ),
            "stage counts: %s, all analog-bound"
            % sorted({c["derived_digital"]["stages_sized"] for c in valid}),
        )
        table.boolean(
            "%s: the engine width is DERIVED at every point (D31)" % name,
            "vector_lanes_provenance is derived on every valid candidate",
            bool(valid)
            and all(
                "derived" in str(c["derived_digital"]["vector_lanes_provenance"])
                for c in valid
            ),
            ", ".join(
                sorted({str(c["derived_digital"]["vector_lanes"]) for c in valid})
            )
            + " lanes over the front",
        )

        # (3) D29 on every row: tokens/s IS 1/beat, and D IS the stage count.
        table.boolean(
            "%s: tokens/s == 1/beat on every point (D29)" % name,
            "one token exits per beat",
            bool(valid)
            and all(
                abs(c["metrics"]["tokens_per_s"] * c["pipeline"]["beat_s"] - 1.0) < 1e-9
                for c in valid
            ),
            "%d points" % len(valid),
        )
        table.boolean(
            "%s: D is the stage count, derived from the stage plan" % name,
            "resident_streams == stages on every point",
            bool(valid)
            and all(
                c["state_bill"]["resident_streams"] == c["state_bill"]["stages"]
                for c in valid
            ),
            "D over the sweep: %s"
            % ", ".join(str(d) for d in sorted({c["state_bill"]["resident_streams"] for c in valid})),
        )

        # (4) The state bill, and the FINDING it carries: a finer stage plan
        # holds more resident state, because every stage holds all D streams'.
        bills = sorted(
            (c["state_bill"]["resident_streams"], c["state_bill"]["max_stage_state_bytes"])
            for c in payload["candidates"]
            if (c.get("state_bill") or {}).get("measured")
        )
        table.boolean(
            "%s: the per-stage state bill RISES with D (D29)" % name,
            "more resident streams -> more state per stage",
            bool(bills) and bills[0][1] < bills[-1][1],
            "%.1f MiB at D=%d to %.1f MiB at D=%d"
            % (
                bills[0][1] / 1024.0 ** 2,
                bills[0][0],
                bills[-1][1] / 1024.0 ** 2,
                bills[-1][0],
            )
            if bills
            else "-",
        )

        # (5) The silicon accounting: three terms, and they close.
        table.boolean(
            "%s: total silicon == analog + chiplets + pools" % name,
            "one accounting, three named terms (D21/D32)",
            bool(valid)
            and all(
                abs(
                    c["silicon"]["total_silicon_mm2"]
                    - (
                        c["silicon"]["analog_macro_silicon_mm2"]
                        + c["silicon"]["shared_digital_silicon_mm2"]
                        + c["silicon"]["macro_pool_silicon_mm2"]
                    )
                )
                <= 1e-9 * max(1.0, c["silicon"]["total_silicon_mm2"])
                for c in valid
            ),
            "%d points" % len(valid),
        )
        table.boolean(
            "%s: both digital terms are COMPOSED from measured synthesis (D32)" % name,
            "shared chiplet and per-macro pool, unit counts x measured area",
            bool(valid)
            and all(
                c["silicon"]["shared_digital_area_provenance"] == "composed-measured"
                and c["silicon"]["macro_pool_area_provenance"] == "composed-measured"
                for c in valid
            ),
            "no declared placeholder anywhere on the front",
        )
        # THE GATE THAT WAS MISSING (P7.9). Nothing here compared an AREA: the
        # rows above check the sum and the provenance STRING, so the sweep could
        # (and did) compose the shared chiplet BEFORE pricing — dropping the
        # D31-derived scan engine from every row while still calling itself
        # composed-measured. This row compares the sweep's own per-chiplet area
        # against the simulator's composition for the SAME point, which is the
        # only reading that can catch it (D21: one accounting per metric).
        mismatched = [
            c["id"]
            for c in valid
            if (c.get("derived_digital") or {}).get("shared_digital_chiplet_area_mm2")
            is not None
            and abs(
                c["silicon"]["shared_digital_area_mm2_per_chiplet"]
                - c["derived_digital"]["shared_digital_chiplet_area_mm2"]
            )
            > 1e-9
            * max(1.0, abs(c["derived_digital"]["shared_digital_chiplet_area_mm2"]))
        ]
        table.boolean(
            "%s: the sweep's chiplet area IS the evaluator's composition (D21/D32)" % name,
            "silicon.shared_digital_area_mm2_per_chiplet == "
            "derived_digital.shared_digital_chiplet_area_mm2, so the D31 scan "
            "engine cannot be missing from one of them",
            bool(valid) and not mismatched,
            "all %d points agree" % len(valid)
            if not mismatched
            else "%d point(s) disagree: %s" % (len(mismatched), ", ".join(mismatched[:5])),
        )
        table.boolean(
            "%s: D27's CELL census rides every priced point" % name,
            "packing.macros, cell_floor_macros and waste_pct, measured by the "
            "dense packer at the point's own bank_depth",
            bool(valid)
            and all(
                (c.get("packing") or {}).get("law") == "dense"
                and c["packing"]["macros"] >= c["packing"]["cell_floor_macros"] > 0
                and 0.0 <= c["packing"]["waste_pct"] < 100.0
                for c in valid
            ),
            "%d points carry a cell census" % len(
                [c for c in valid if (c.get("packing") or {}).get("macros")]
            ),
        )
        table.boolean(
            "%s: Invariant W pins the analog floor per macro (D27)" % name,
            "one macro footprint over the whole sweep; area moves by SLOT COUNT",
            len({round(c["silicon"]["macro_footprint_mm2"], 12) for c in valid}) == 1,
            "%d distinct footprints" % len(
                {round(c["silicon"]["macro_footprint_mm2"], 12) for c in valid}
            ),
        )

        # (6) The front is a front: sorted by area, throughput is monotone and
        # no point on it dominates another.
        front = [c for c in payload["candidates"] if c["id"] in payload["front_ids"]]
        front.sort(key=lambda c: c["silicon"]["total_silicon_mm2"])
        rates = [c["metrics"]["tokens_per_s"] for c in front]
        table.boolean(
            "%s: the front is monotone in (area, tokens/s)" % name,
            "sorted by area, throughput never falls",
            len(front) >= 2 and rates == sorted(rates),
            "%d points: %s"
            % (len(front), " -> ".join("%.0f" % rate for rate in rates)),
        )
        table.exact(
            "%s: front shape read off the data" % name,
            True,
            payload["front_shape"] in ("spread", "flat_area", "dominated_chain", "tied"),
        )

        # (7) The knee, recomputed here from the front's two axes only.
        knee = payload["knee"]
        if len(front) >= 3:
            areas = [c["silicon"]["total_silicon_mm2"] for c in front]
            span_a = areas[-1] - areas[0]
            span_r = max(rates) - min(rates)
            if span_a > 0 and span_r > 0:
                distances = dict(
                    (
                        c["id"],
                        (c["metrics"]["tokens_per_s"] - min(rates)) / span_r
                        - (c["silicon"]["total_silicon_mm2"] - areas[0]) / span_a,
                    )
                    for c in front
                )
                expected = max(distances, key=lambda cid: distances[cid])
                table.exact(
                    "%s: the knee is the front point furthest above its chord" % name,
                    expected if distances[expected] > 1e-9 else None,
                    knee["point"],
                )
        table.boolean(
            "%s: the knee is named or refused with a reason" % name,
            "a point id, or None with the reason it has no bend",
            knee["point"] is not None or bool(knee["basis"]),
            str(knee["point"] or knee["basis"][:60]),
        )

        # (8) Infeasible by residency: a verdict with a stage and bytes.
        summary = payload["state_bill_summary"]
        refused = summary["infeasible_by_state"]
        if summary["budget_bytes"] is None:
            # No budget declared on this sweep. That is a CHOICE with a reason,
            # and the row says which choice it is rather than passing a check
            # that was never made (D21).
            table.boolean(
                "%s: no residency budget declared, and the bill is reported anyway" % name,
                "max_stage_state_bytes absent -> no point refused for residency, "
                "every point's measured bill still on its row",
                not refused and summary["points_measured"] > 0,
                "%d point(s) with a measured bill, 0 refused"
                % summary["points_measured"],
            )
        else:
            table.boolean(
                "%s: residency refusals name the stage and the bytes (D29)" % name,
                "every refused point lists the violating stage(s) over the budget",
                bool(refused)
                and all(
                    row["violating_stages"]
                    and all(
                        stage["state_bytes"] > summary["budget_bytes"]
                        for stage in row["violating_stages"]
                    )
                    for row in refused
                ),
                "%d point(s) refused by residency, %d by memory"
                % (len(refused), len(summary["infeasible_by_memory"])),
            )
            table.boolean(
                "%s: a refused point was PRICED before it was refused" % name,
                "the state check is last, so an infeasible point still reports its "
                "throughput and its bill",
                bool(refused)
                and all(
                    row["max_stage_state_bytes"] and row["resident_streams"]
                    for row in refused
                ),
                "%.1f MiB at D=%d on the worst refusal"
                % (
                    max(row["max_stage_state_bytes"] for row in refused) / 1024.0 ** 2,
                    max(row["resident_streams"] for row in refused),
                )
                if refused
                else "NO point was refused, although a budget is declared",
            )

        # (9) D28: utilization at every point, idle devices inside the mean.
        table.boolean(
            "%s: per-device-class utilization at every point (D28)" % name,
            "every valid point carries a row per instantiated class",
            bool(valid) and all(c["utilization"] for c in valid),
            "classes: %s"
            % ", ".join(sorted({row["device_class"] for row in valid[0]["utilization"]})),
        )

        # (10) The round trip: the emitted machine reproduces the selection.
        verify = payload["verify"]
        table.boolean(
            "%s: --emit-config + --verify round trip" % name,
            "run_perf on the emitted config reproduces the selected point (<=0.1%)",
            bool(verify) and verify["pass"],
            "%d checks" % len(verify["checks"]) if verify else "no verify block",
        )


def _cross_product_size(axes):
    total = 1
    for values in axes.values():
        total *= len(values)
    return total


def main():
    table = CheckTable()

    # T1 / T2 / T3 tiers. Each run overwrites output/VIT, so the T1 report
    # (pass-2B DSE cross-check) and the T3 report (A100 comparison) are
    # captured before later runs clobber them.
    t1_report = None
    t3_report = None
    for tier in TIERS:
        check_tier(tier, table)
        if tier["name"].startswith("T1") and os.path.exists(FWS_REPORT):
            with open(FWS_REPORT) as handle:
                t1_report = json.load(handle)
        if tier["name"].startswith("T3") and os.path.exists(FWS_REPORT):
            with open(FWS_REPORT) as handle:
                t3_report = json.load(handle)

    # Regression gates (DESIGN section 5: bit-identical, no tolerance).
    check_regression_gate(
        "gate A100+vit_base",
        "a100_80GB.yaml",
        "vit_base_inf.yaml",
        VIT_RESULTS,
        BASELINE_A100_VIT,
        table,
    )
    print("Running the A100 + Llama2-7B gate (takes about a minute)...")
    check_regression_gate(
        "gate A100+Llama2-7B",
        "a100_80GB.yaml",
        "Llama2-7B_inf.yaml",
        LLM_RESULTS,
        BASELINE_A100_LLAMA,
        table,
    )

    # Cross-architecture demo (gated only on "the run completes").
    if t3_report is not None:
        print_comparison(t3_report, table)
    else:
        table.boolean(
            "A100-vs-FWS comparison run completes",
            "requires T3 report",
            False,
            "T3 report missing",
        )

    # Pass-2A rows (DESIGN2 section 4), appended after the pass-1 rows so the
    # original check list stays intact and in order. These runs write to
    # output/LLM; each report is read back immediately after its run.
    print()
    print("Running the pass-2A LLM/MoE smokes (three run_perf subprocesses)...")
    check_llm_dense_smoke(table)
    check_llm_kvdram_smoke(table)
    check_moe_smoke(table)
    check_law_spot_checks(table)

    # Pass-2B rows (DESIGN2 section 5), appended after the pass-2A rows.
    # These write only under output/fws_cim_dse/validation_* (the --verify
    # run_perf executes with cwd inside that tree), so the output/VIT and
    # output/LLM artifacts the earlier rows read stay untouched.
    print()
    print("Running the pass-2B DSE checks (three DSE sweeps; one --verify run_perf)...")
    check_dse_t1_sweep(t1_report, table)
    check_dse_llama_verify(table)
    check_dse_moe_expert_parallel(table)

    # Pass-3 rows (QIF P6.3, ADJ-8): the bridge from the frozen closed form to
    # the placed DAG. Appended last so the 152 rows above keep their order and
    # their artifacts — these runs rewrite output/VIT and output/LLM.
    print()
    print("Running the bridge gate (five run_perf subprocesses + five DAG builds)...")
    check_bridge_gate(table)

    # QIF P3.7 rows: the MAPPED-path sweep (D10's "swept by the DSE"). The
    # demo rows read the checked-in artifact (the pytest suite regenerates and
    # compares it); the verify rows run a live sweep whose run_perf executes
    # with cwd inside output/fws_qif_dse, so output/VIT and output/LLM stay as
    # the rows above left them.
    print()
    print("Running the QIF P3.7 mapped-DSE rows (one live sweep + one --verify run_perf)...")
    check_qif_dse_selection(table)
    check_qif_dse_retired(table)
    check_qif_dse_verify(table)

    # QIF P7.7 rows: the filled-pipeline regime (D29/D30). In-process and last,
    # so no artifact any earlier row reads is touched.
    print()
    print("Running the QIF P7.7 filled-pipeline rows (one in-process Granite run)...")
    check_qif_filled_pipeline(table)

    # QIF P7.8 rows: the measured synthesis library (D32) and the engine width
    # D31 derives from the beat. In-process, after every artifact-reading row.
    print()
    print("Running the QIF P7.8 synthesis-library / derived-sizing rows...")
    check_qif_synthesis_library(table)

    # QIF P7.4/P7.5 rows: the regime-v2 frontier curves. They READ the two
    # checked-in artifacts (the pytest suite re-walks and compares them), so
    # nothing here touches an artifact an earlier row read.
    print()
    print("Running the QIF P7.4/P7.5 frontier rows (two checked-in curves, read)...")
    check_qif_service_links(table)
    check_qif_frontier(table)

    print()
    print("FWS-CIM validation vs OPTIMA — DESIGN section 5")
    print("=" * 78)
    print(table.render())
    print("=" * 78)
    n_pass = sum(1 for row in table.rows if row[4])
    verdict = "ALL CHECKS PASS" if table.all_pass else "FAILURES PRESENT"
    print("{}: {}/{} checks pass".format(verdict, n_pass, len(table.rows)))
    return 0 if table.all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
