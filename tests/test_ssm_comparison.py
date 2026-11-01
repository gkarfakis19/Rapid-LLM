"""Make duplicate collectives and single-device peak regressions fail fast."""
import copy

from tools.compare_ssm_goldens import collective_counts, compare_peak_memory


def test_collective_counts_preserve_multiplicity():
    rms = ("all_reduce", 8, 2, "ssm_rms_stats_allreduce")
    assert collective_counts([rms, rms]) == [dict(collective=list(rms), count=2)]
    assert collective_counts([rms, rms]) != collective_counts([rms])


def test_peak_comparison_pins_every_device_and_only_declared_corrections():
    old = dict(summary=[dict(gpu_id=i, static_gib=250 / 1024**3,
                            peak_gib=270 / 1024**3) for i in range(2)],
               ssm_layers_per_device=[2, 2],
               transformer_dense_layers_per_device=[2, 2],
               weight_mem_per_layer_dense=50)
    new = [dict(static_bytes=[130, 130], peak_bytes=[150, 150])]
    rows, matches = compare_peak_memory(old, new, conv_bytes_saved=10)
    assert matches
    assert [r["archive_duplicate_weight_bytes"] for r in rows] == [100, 100]
    broken = copy.deepcopy(new)
    broken[0]["peak_bytes"][1] += 1
    assert not compare_peak_memory(old, broken, conv_bytes_saved=10)[1]
