# barrier — shared core (BARRIER)

This package contains the **shared** BARRIER machinery.  Every experiment
directory (`Classification/`, `DDPM/`, `SD/`, `Flux/`) imports from here;
nothing in it is experiment-specific, and no experiment re-defines the core
algorithm.

| Module | What it provides |
|--------|------------------|
| `intact.py` | The core algorithm: `UnlearnIntervalProtection` (`setup_protection`, `compute_protection_loss`, `freeze_non_target_params`, `get_trainable_params`), region-construction primitives (`make_region_boxes`, `box_drift_max`, …), and model forward functions (`ddpm_forward_fn`, `classification_forward_fn`). |
| `cache.py` | Redirects HF/torch/wandb/CLIP caches off the home filesystem. Imported automatically by `import barrier` — must run before any downloading library. |
| `sd_utils.py` | SD-specific helpers previously copy-pasted across the SD experiments: `sd_forward_fn`, `sd_forward_fn_model_schedule`, `setup_intact_protection`, `compact_target_tag`. |
| `ablation_common.py` | Shared ablation machinery: CSV schema, `summarize_ablations`, `microbench_protection`, and the shared ablation mode lists (`REGION_MODES`, `ABLATION_INTERVAL_MODES`, `LAMBDA_SWEEP`, matvec-count tables). |
| `ablation_mechanism.py` | Post-hoc checkpoint analysis (Experiments 1–2): realised-update vs cost scatter, kappa statistics. |
| `expgrid.py` | Deterministic grid definition for the mechanism/design-choice ablations (Experiments 3–9). Consumed by the SLURM ablation arrays. |

## Getting started

- All pipelines already `import barrier` / `from barrier.* import …`; just make
  sure the repo root is on `PYTHONPATH` (every run script does this).
- `import barrier` also runs `barrier.cache` as its first side effect, so it
  is safe to import the package before any torch / transformers / diffusers
  import.
- If you add shared code that two or more experiments would otherwise
  duplicate, put it here.