"""The approved CIM KV reporting correction has no pricing consumers."""
from pathlib import Path

import config
from inference_timing import TimeCalculationLLMInference

ROOT = Path(__file__).resolve().parents[1]


def test_granite_cim_kv_diagnostics_count_only_attention_layers(tmp_path, monkeypatch, capsys):
    model = config.parse_config(str(ROOT / "configs/model-config/granite_4_0_h_1b_decode_inf.yaml"), "LLM")
    hw = config.parse_config(str(ROOT / "configs/hardware-config/fws_cim_granite_1b_frontier.yaml"), "hardware")
    tc = TimeCalculationLLMInference(hw, model, "LLM", output_dir=str(tmp_path))
    # Isolate reporting with the auditor's independently captured full-run
    # phase times. The slow CLI test exercises the actual phases and QIF.
    monkeypatch.setattr(tc, "calc_time", lambda: (0.2667176183484531, 0.0))
    monkeypatch.setattr(tc, "calc_decode_time", lambda: (0.40426278471582894, 0.0, []))
    result = tc.calc_total_inference_time()
    layers = len(tc.layer_mixers)
    attention = sum("attention" in kinds for kinds in tc.layer_mixers)
    assert (attention, layers) == (4, 40)
    old_and_corrected = {
        "kv_cache_prefill_store_bytes": (146800640, 14680064),
        "kv_cache_decode_store_bytes": (20971520, 2097152),
        "kv_cache_decode_fetch_bytes": (40265318400, 4026531840),
    }
    for key, (old, corrected) in old_and_corrected.items():
        assert result[key] == corrected == attention / layers * old
    assert result["total_inference_time"] == 0.6709804030642821
    assert "[kv-cache] prefill_store=0.0 GiB, decode_store=0.0 GiB, decode_fetch=3.8 GiB" in capsys.readouterr().out
