# Branches and related trees

`main` is the one working copy of Rapid-LLM. It combines:

- the rewrite (program IR; the old graph flattener is gone),
- the FWS-CIM device class and QIF tooling (`docs/qif/`),
- SSM and hybrid inference on GPU and CIM with one schema, `model_param.layer_plan` (`docs/ssm/SSM.md`).

No SSM path models training.

## Generated results live outside the repo

The QIF atlas, DSE reports and diagnostics are generated files. They are kept out of git history at
`/app/nanocad/projects/ispass_deepflow/Rapid-LLM-results/qif` (see its `README.txt` and `STRIPPED_FILES.txt`).
Tests find them through `RAPID_QIF_RESULTS`. Missing results fail the tests unless `RAPID_QIF_ALLOW_MISSING=1`.

The same folder holds `ssm_old_goldens/`, the reference runs of the pre-rewrite GPU SSM branch that
`tools/compare_ssm_goldens.py` checks against, and `audits/`, the evidence from the consolidation audits.

## Kept branches

| Branch | What it is |
|---|---|
| `origin/heterogeneous` | Thermal / heterogeneous-die work. Kept for reference, not merged. |
| `origin/remote_backend` | Paper validation work (device testbench, baselines). Kept for reference, not merged. |
| `origin/fws_cim_dev` | The CIM branch before its generated results were stripped. Superseded by `main`. |
| `origin/rewrite_astra` | The rewrite branch. Superseded by `main`. |

Local-only `archive/*` branches in the main checkout hold the rest: the CIM branch with its results,
the old GPU SSM branch, and snapshots of uncommitted work and stashes from the retired copies. They are
not pushed.

## Related trees under `/app/nanocad/projects/ispass_deepflow`

| Path | What it is |
|---|---|
| `Rapid-LLM/` | This repository (`main`). |
| `Rapid-LLM-results/` | Generated results, old SSM reference runs, audit evidence. |
| `temp_RAPIDLLM_energy/`, `energy_landscape_explore/` | The energy project. A separate project; left exactly where it was. |
| `external/` | Third-party baseline tools used by the validation scripts: calculon, llm-analysis, stage_new (STAGE), vtrain, op-model. |
| `.archive/` | Retired copies: the old dev tree, the rewrite tree, DeepFlow and DeepFlow_alt. |
| `deepflow_astra_dev`, `deepflow_astra_dev_rewrite` | Symlinks into `.archive/`, kept so that existing remotes and absolute paths (including the energy trees' remotes) still resolve. |
| `overleaf_paper/` | Paper sources. |
