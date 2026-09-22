# BARRIER: Bounded Activation Regions for Robust Information Erasure

**Jan Miksa @ GMUM JU** &nbsp;·&nbsp; **Patryk Krukowski @ GMUM JU**

BARRIER ("InTAct", Interval-based Task Activation Consolidation) is a machine-learning
unlearning method. Given a *forget set* (samples/concepts to erase) and a *retain set*
(samples to keep), it adds an **interval protection loss** to any base unlearning method
(GA, RL, ESD, …). The protection loss keeps the model's activations inside a "safe zone"
derived from the retain set in the SVD subspace of the forget set — so unlearning succeeds
while the retained behaviour is preserved.

```
total_loss = base_method_loss + lambda_interval * interval_protection_loss
```

## Repository layout

```
BARRIER/
├── barrier/                # SHARED CORE – the algorithm and shared utils.
│   │                         Every experiment imports from here; nothing is
│   │                         duplicated across experiments (see barrier/README.md).
│   ├── intact.py           #   UnlearnIntervalProtection + region primitives + forward fns
│   ├── sd_utils.py         #   SD-specific helpers (sd_forward_fn, setup_intact_protection, …)
│   ├── cache.py            #   cache redirection (auto-run on `import barrier`)
│   ├── ablation_common.py  #   shared ablation CSV machinery + microbenchmark
│   ├── ablation_mechanism.py  # post-hoc checkpoint analysis (Exp 1–2)
│   └── expgrid.py          #   deterministic ablation grids (Exp 3–9)
│
├── Classification/         # ResNet-18 / CIFAR-10  (vendored Sparse-Unlearn/SalUn base, ESC ViT track)
├── DDPM/                   # Conditional DDPM / CIFAR-10, STL-10 (vendored DDIM + Selective-Amnesia base)
├── SD/                     # Stable Diffusion v1.4 (vendored latent-diffusion base + ESD / RECE / SCaPre / STEREO)
├── Flux/                   # FLUX.1-dev (vendored EraseAnything base + HF diffusers/peft)
│
├── benchmark_timing/       # SLURM wall-time/VRAM comparison (SalUn | SEMU | ESC | BARRIER)
├── README.md               # ← this file
├── NOTES.md                # setup details, envs, data prep, sweep/SLURM recipes
├── DDPM.pth                # reference checkpoint (git-lfs, kept intentionally)
└── LICENSE
```

## The core algorithm (`barrier/intact.py`)

`UnlearnIntervalProtection` is used identically in every experiment:

1. `setup_protection(model, forget_dl, device, remain_dl, forward_fn, …)`
   — collects activations on the forget set, runs SVD/PCA, and derives the
   safe interval `[z_min, z_max]` + envelope `[inf_low, inf_high]` per target layer.
2. `compute_protection_loss(model, device)` — penalises parameter drift that
   pushes activations outside the protected intervals/negative space.
3. `freeze_non_target_params(model)` / `get_trainable_params(model)` — restrict
   the optimizer to the unlearning-relevant parameters.

### Protection constants

| Parameter | Default | Description |
|-----------|---------|-------------|
| `targets` | `["fc"]` | Layer-name patterns to protect (e.g. `["to_q","to_k","to_v"]` or `transformer_blocks.12.attn.*`) |
| `lambda_interval` | `10.0` | Weight of the protection loss |
| `lower_percentile` / `upper_percentile` | `0.05` / `0.95` | Percentile bounds of the safe activation zone |
| `reduced_dim` | `32` | SVD dimensions (efficiency) |
| `infinity_scale` | `20.0` | Scale of the outer envelope (negative space) |
| `use_actual_bounds` | `False` | Use actual retain-set min/max instead of scaled bounds |
| `normalize_protection` | `True` | Normalize the loss by the number of layers |

## The experiments

Every experiment follows the same pattern: a **unified pipeline** entrypoint
(`pipeline.py`) that orchestrates **unlearn → generate → evaluate → wandb**,
driven by a single YAML config. All pipelines support wandb sweeps and are
SLURM-ready. See `NOTES.md` for environment setup before running anything.

| # | Experiment | Where | Entrypoint | Config to edit (`configs/`) | What to change |
|---|-----------|-------|-----------|------------------------------|----------------|
| 1 | Classification — class-wise forgetting (ResNet-18/CIFAR-10) | `Classification/` | `python pipeline.py --config configs/pipeline_classwise.yaml` | `pipeline_classwise.yaml` | `paths.pretrained_ckpt`, `paths.data_dir`, `unlearn.class_to_forget`, `intact.*` |
| 2 | Classification — random-sample forgetting (10%/50%) | `Classification/` | `python pipeline.py --config configs/pipeline_random.yaml` | `pipeline_random.yaml` (or `_10pp` / `_50pp`) | same as #1 |
| 3 | Classification — ViT/Tiny-ImageNet benchmark (ESC base) | `Classification/ESC/` | `python unlearn_intact.py --method intact …` | `ESC/configs/sweep_*.yaml` | forget class, targets, HP search |
| 4 | Classification — mechanism/design ablations (Exp 1–9) | `Classification/` | `python ablation_cls.py …` / `barrier/expgrid.py resnet18-classwise <i>` | via `scripts/slurm_ablation_cls_*.sh` | `expgrid.py` grids |
| 5 | DDPM — class forgetting (CIFAR-10/STL-10) | `DDPM/` | `python pipeline.py --config configs/pipeline.yaml` | `pipeline.yaml` (+ `configs/cifar10_intact*.yml` model config) | `paths.pretrained_ckpt_folder`, `paths.classifier_ckpt`, `unlearn.mode/method`, `intact.*` |
| 6 | DDPM — sweeps (GA/RL/KL/ESD + ablations) | `DDPM/` | `./run_sweep.sh sweep` | `sweep*.yaml` | sweep ranges |
| 7 | DDPM — region ablations (Exp 3–9) | `DDPM/` | `python ablation_regions.py …` / `barrier/expgrid.py ddpm <i>` | `scripts/slurm_ablation*.sh` | `expgrid.py` grids |
| 8 | SD — Imagenette class forgetting | `SD/` | `python pipeline.py --config configs/pipeline_class.yaml` | `pipeline_class.yaml` (+ `paths.sd_ckpt` → v1-4 weights) | `paths.*`, `unlearn.*`, `intact.*` |
| 9 | SD — NSFW removal | `SD/` | `python pipeline.py --config configs/pipeline_nsfw.yaml` | `pipeline_nsfw.yaml` | NSFW/not-NSFW data folders, `intact.targets` |
| 10 | SD — artist unlearning | `SD/` | `python train-scripts/train_artists.py` | `configs/pipeline_artists.yaml` | artist prompt CSV, `paths.*` |
| 11 | SD — SCaPre benchmark (ImageNet-Diversi50/Confuse5) | `SD/scapre/` | `python train.py --config configs/train_diversi50.yaml` (+ `evaluate.py`) | `scapre/configs/*.yaml` | dataset root, model paths |
| 12 | Flux — concept (nudity) erasure | `Flux/` | `python intact_pipeline.py --config configs/intact/pipeline_concept.yaml` | `pipeline_concept.yaml` | HF token, `unlearn.*`, `intact.target_blocks/layers` |
| 13 | Flux — Imagenette class forgetting | `Flux/` | `python intact_pipeline.py --config configs/intact/pipeline_class.yaml` | `pipeline_class.yaml` | same as #12 |
| 14 | Flux — NSFW dataset erasure | `Flux/` | `python intact_pipeline.py --config configs/intact/pipeline_nsfw.yaml` | `pipeline_nsfw.yaml` (or `_10k`, `_debug`) | NSFW/not-NSFW folders, `intact.*` |
| 15 | Timing benchmark (SalUn/SEMU/ESC/BARRIER) | `benchmark_timing/` | `sbatch run_timing_benchmark.sh` | env vars (see its header) | `REPEATS`, `BATCH_SIZE`, `FORGET_CLASS` |

**Quick start (fastest, no downloads):** #1 and #2 only need CIFAR-10 + a
trained ResNet checkpoint.

### Common cost of changing an experiment

1. Edit the pipeline YAML (`paths` → your checkpoints/data, `wandb.entity`,
   hyperparameters under `unlearn.*` / `intact.*`).
2. Run the pipeline listed above (it also logs everything to wandb).
3. For sweeps: `./run_sweep.sh <sweep-name>` (see `NOTES.md` §Sweeps) or launch
   agents on SLURM.

## Install

```bash
# everything ships self-contained (vendored repos included); only per-modality
# Python environments are needed, see NOTES.md §Environments
```

| Experiment | Environment |
|-----------|-------------|
| Classification / DDPM | `pip install -r requirements.txt` (DDPM: py3.8 `salun-ddpm` env) |
| SD | `conda env create -f environment.yaml` → `ldm` |
| Flux | `pip install -e ./diffusers[torch]` + `pip install -e ./peft`, `tokenizers==0.20.0`, HF token for `black-forest-labs/FLUX.1-dev` |

Full step-by-step setup (weights, data, wandb entity, cache routing) is in
`NOTES.md`.

## Vendored code

The repository vendors upstream codebases so each experiment is
self-contained (no network installs). Vendored pieces are kept unmodified;
BARRIER additions live beside them:

| Directory | Vendored upstream |
|-----------|-------------------|
| `Classification/models, unlearn, trainer, pruner, evaluation, …` | Sparse-Unlearn / SalUn; `ESC/` = ESC (KHU-VGI) |
| `DDPM/models, functions, datasets` | DDIM / Selective-Amnesia |
| `SD/ldm, configs/latent-diffusion…` | CompVis latent-diffusion |
| `SD/train-scripts/{train-esd,convertModels,generate_mask}.py` | Erased Diffusion (ESD) |
| `SD/rece/` | RECE / Unified Concept Editing benchmark |
| `SD/scapre/datasets/`, `SD/stereo/prompts/` | SCaPre / STEREO benchmark data |
| `Flux/diffusers, peft` + EraseAnything files | HuggingFace diffusers & peft; EraseAnything (ICML 2025) |

## License

MIT License.

Acknowledgements: funded by National Science Centre, Poland, 2023/49/B/ST6/01137.