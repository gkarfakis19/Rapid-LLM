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


def _read_llm_report(name, table):
    """Read output/LLM/fws_cim_report.json right after a run (runs clobber it)."""
    if not table.boolean(
        "%s: fws_cim_report.json written (output/LLM)" % name,
        "exists",
        os.path.exists(LLM_FWS_REPORT),
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
    REPO_ROOT, "docs", "qif", "dse", "granite_lanes_banks", "dse_report.json"
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
            "%s: closed-form report written" % label, "exists", os.path.exists(report_path)
        ):
            continue
        with open(report_path) as handle:
            closed_form = json.load(handle)

        raw = load_yaml(hw_path)
        _config.convert(raw)
        mapping = _fws_mapping.build_mapping(
            _config.HWConfig.from_dict(raw), _config.parse_config(model_path, mode)
        )
        evaluation = _fws_eval.evaluate_fws(_build_fws_program(mapping))
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
    vector_lanes {512,1024,2048,4096} x bank_depth {1,2}.

    Read, not rerun: `tests/test_qif_dse_allocation.py` regenerates this
    artifact and compares it, so the rows here are about WHAT THE SWEEP
    FOUND — the headline scaling with the declared design point, the pick,
    and the front.
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
        {"vector_lanes": [512, 1024, 2048, 4096], "bank_depth": [1, 2]},
        payload["sweep"]["axes"],
    )
    table.exact("%s: candidates" % name, 8, payload["num_candidates"])
    table.exact("%s: valid candidates" % name, 8, payload["num_valid"])
    table.boolean(
        "%s: every candidate priced on the mapped path" % name,
        "fws_mapping -> fws_build -> fws_eval",
        payload["evaluation_path"].startswith("fws_mapping.build_mapping"),
        payload["evaluation_path"][:60],
    )

    # THE FINDING: at a fixed bank depth the headline rises with every
    # doubling of the declared vector-lane count.
    for bank in (1, 2):
        series = [
            c["metrics"]["tokens_per_s"]
            for c in payload["candidates"]
            if c["knobs"]["bank_depth"] == bank
        ]
        table.boolean(
            "%s: headline rises with vector_lanes (bank_depth %d)" % (name, bank),
            "strictly increasing over 512/1024/2048/4096",
            len(series) == 4 and all(b > a for a, b in zip(series, series[1:])),
            ", ".join("%.1f" % value for value in series),
        )
    fastest = max(c["metrics"]["tokens_per_s"] for c in payload["candidates"])
    slowest = min(c["metrics"]["tokens_per_s"] for c in payload["candidates"])
    table.boolean(
        "%s: 8x the lanes is SUB-linear in the headline" % name,
        "1 < fastest/slowest < 8 (the analog GEMMs do not move)",
        1.0 < fastest / slowest < 8.0,
        "%.3f" % (fastest / slowest),
    )

    selected = payload["selected"]
    table.exact(
        "%s: selected knobs" % name,
        {"vector_lanes": 4096, "bank_depth": 1},
        selected["knobs"],
    )
    table.boolean(
        "%s: selection sits on the Pareto front" % name,
        "selected_id in front_ids",
        payload["selected_id"] in payload["front_ids"],
        "front=%s (%s)" % (payload["front_ids"], payload["front_shape"]),
    )
    table.boolean(
        "%s: the front's shape is read off the data" % name,
        "flat_area: every valid candidate carries the same silicon",
        payload["front_shape"] == "flat_area"
        and len({round(c["silicon"]["total_silicon_mm2"], 9) for c in payload["candidates"]}) == 1,
        payload["front_shape"],
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
    table.boolean(
        "%s: the front's note names the CONSTANT terms" % name,
        "the enumerated slot count and the chiplet count, both read off the rows",
        len(slots) == 1
        and len(chiplets) == 1
        and ("slot count is %d" % list(slots)[0]) in note
        and ("chiplet count is %d" % list(chiplets)[0]) in note,
        "slots=%s chiplets=%s" % (sorted(slots), sorted(chiplets)),
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
    table.boolean(
        "%s: unowned columns are IDENTICAL across the sweep" % name,
        "banking changes active column sets, not stranded ones (ADJ-4)",
        len({c["placement"]["unowned_columns"] for c in valid_rows}) == 1
        and len({c["placement"]["macros_holding_tiles"] for c in valid_rows}) == 1,
        "unowned=%s macros=%s"
        % (
            sorted({c["placement"]["unowned_columns"] for c in valid_rows}),
            sorted({c["placement"]["macros_holding_tiles"] for c in valid_rows}),
        ),
    )

    def _component(candidate, key):
        for entry in candidate["metrics"]["energy_components"]:
            if entry["component"] == key:
                return entry["energy_pj"]
        return None

    bank1 = [c for c in valid_rows if c["knobs"]["bank_depth"] == 1]
    bank2 = [c for c in valid_rows if c["knobs"]["bank_depth"] == 2]
    if bank1 and bank2:
        total_delta = bank2[0]["metrics"]["total_energy_pj"] - bank1[0]["metrics"]["total_energy_pj"]
        array_delta = _component(bank2[0], "analog_arrays") - _component(bank1[0], "analog_arrays")
        table.boolean(
            "%s: the whole banking energy delta is analog_arrays" % name,
            "total delta == analog_arrays delta; link_traffic unmoved",
            abs(total_delta - array_delta) < 1e-3
            and _component(bank1[0], "link_traffic") == _component(bank2[0], "link_traffic"),
            "total %.6g pJ vs arrays %.6g pJ" % (total_delta, array_delta),
        )

    # WAVE D AUDIT: D21 wants the relaxation disclosed in the ARTIFACT, and
    # the MD is the artifact a human reads.
    coverage = payload.get("silicon_coverage") or {}
    table.boolean(
        "%s: the silicon accounting names its uncovered terms" % name,
        "silicon_coverage.uncovered_terms is non-empty and named",
        bool(coverage.get("uncovered_terms")),
        "; ".join(coverage.get("uncovered_terms", [])) or "(none)",
    )
    demo_md = os.path.join(os.path.dirname(QIF_DSE_DEMO), "dse_report.md")
    if os.path.exists(demo_md):
        with open(demo_md) as handle:
            md_text = handle.read()
        table.boolean(
            "%s: the MD carries the coverage, not only the JSON" % name,
            "silicon_accounting + silicon_uncovered in the Disclosures section",
            "silicon_accounting" in md_text
            and "silicon_uncovered" in md_text
            and all(term in md_text for term in coverage.get("uncovered_terms", ["-"])),
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
        table.exact(
            "%s: emitted vector_lanes" % name,
            4096,
            emitted["cim"]["cards"]["sa"]["vector_lanes"],
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
    check_qif_dse_verify(table)

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
