# InTAct for Unlearning

**InTAct** (Interval-based Task Activation Consolidation) is a continual learning technique preserving activations on previous tasks. Here, we adapt InTAct for machine unlearning. It works by contstraining how activations can change anywhere apart from the forget region.

## How InTAct Works

### Protection Loss

The protection loss has three components:

```
L_protect = L_mean + L_residual + L_interval
```

- **L_mean**: Penalizes shift in mean activation (global drift)
- **L_residual**: Penalizes changes in residual (non-principal) directions  
- **L_interval**: Penalizes activations moving outside the safe zone defined by percentiles

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         InTAct                              │
├─────────────────────────────────────────────────────────────┤
│  1. setup_protection(model, forget_dl, forward_fn)          │
│     - Find target layers (e.g., "to_q", "to_k", "to_v")     │
│     - Collect activations via forward hooks                 │
│     - Compute SVD and define safe intervals                 │
│     - Snapshot initial parameters                           │
│                                                             │
│  2. compute_protection_loss(model, device)                  │
│     - Compare current params to snapshot                    │
│     - Compute drift in SVD space                            │
│     - Return weighted protection loss                       │
└─────────────────────────────────────────────────────────────┘
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `targets` | `["fc"]` | Layer name patterns to protect (e.g., `["to_q", "to_k", "to_v"]`) |
| `lambda_interval` | `10.0` | Weight for protection loss |
| `lower_percentile` | `0.05` | Lower bound of safe activation zone |
| `upper_percentile` | `0.95` | Upper bound of safe activation zone |
| `reduced_dim` | `32` | SVD dimensions (for efficiency) |
| `infinity_scale` | `20.0` | Scale for outer bounds (negative space) |
| `use_actual_bounds` | `False` | Use actual min/max from remain data instead of scaled bounds |
| `normalize_protection` | `True` | Normalize loss by number of layers |

## Installation

Please follow instructions from each subfolder.

## Pipelines (Recommended)

Each setting has a **unified pipeline** that orchestrates unlearning → evaluation → wandb logging via a single YAML config. All pipelines support **wandb sweeps** and are **SLURM-ready**.

### Classification – Class-wise Forgetting

```bash
cd Classification
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"
pip install -r requirements.txt

# Edit configs/pipeline_classwise.yaml (paths, wandb entity, etc.)
python pipeline.py --config configs/pipeline_classwise.yaml
```

### Classification – Random Data Forgetting

```bash
cd Classification
python pipeline.py --config configs/pipeline_random.yaml
```

### DDPM – Conditional Diffusion (CIFAR-10)

```bash
cd DDPM
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"
pip install -r requirements.txt

# Edit configs/pipeline.yaml (paths, wandb entity, etc.)
python pipeline.py --config configs/pipeline.yaml
```

### Stable Diffusion – Class Forgetting

```bash
cd SD
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"
conda env create -f environment.yaml && conda activate ldm
# Download SD v1.4 weights (see SD/README.md)

python pipeline.py --config configs/pipeline_class.yaml
```

### Stable Diffusion – NSFW Removal

```bash
cd SD
python pipeline.py --config configs/pipeline_nsfw.yaml
```

### wandb Sweeps

Sweeps automate hyperparameter search across pipeline runs. Each sweep config
defines which parameters to vary; the pipeline YAML provides defaults for
everything else.

**Quick start:**
```bash
# 1. Create the sweep (returns a sweep-id)
cd SD
wandb sweep configs/sweep_class.yaml

# 2. Launch one or more agents (each runs pipeline.py with different params)
wandb agent <sweep-id>
```

**On SLURM** – launch one agent per job:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --mem=48G --time=48:00:00 --array=0-9
source activate ldm
cd /path/to/InTAct-Unl/SD
export PYTHONPATH="${PYTHONPATH}:/path/to/InTAct-Unl"
wandb agent <sweep-id>
```

**Sweep parameter format** – dotted keys map to nested YAML fields:
```yaml
parameters:
  unlearn.lr:               # → cfg["unlearn"]["lr"]
    values: [1e-5, 5e-5]
  intact.lambda_interval:   # → cfg["intact"]["lambda_interval"]
    values: [1.0, 10.0]
```

**Available sweep configs:**
| Setting | Config | Key parameters |
|---------|--------|----------------|
| Classification class-wise | `Classification/configs/sweep_classwise.yaml` | `unlearn_lr`, `unlearn_epochs`, `lambda_interval`, `base_method` |
| Classification random | `Classification/configs/sweep_random.yaml` | `unlearn_lr`, `unlearn_epochs`, `lambda_interval`, `base_method` |
| DDPM class forgetting | `DDPM/configs/sweep.yaml` | `lr`, `n_iters`, `lambda_interval`, `method` |
| SD class forgetting | `SD/configs/sweep_class.yaml` | `lr`, `epochs`, `lambda_interval`, `base_method`, `targets` |
| SD NSFW removal | `SD/configs/sweep_nsfw.yaml` | `lr`, `epochs`, `lambda_interval`, `targets` |

**Adding parameters:** copy any dotted key from the pipeline YAML into the
sweep's `parameters:` block. Use `values:` for grid, `min:`/`max:` for random,
or `distribution:` for Bayesian. See the
[wandb sweep docs](https://docs.wandb.ai/guides/sweeps) for details.

**Examples:**
```bash
# Classification class-wise sweep
cd Classification
wandb sweep configs/sweep_classwise.yaml
wandb agent <sweep-id>

# DDPM sweep
cd DDPM
wandb sweep configs/sweep.yaml
wandb agent <sweep-id>

# SD class sweep
cd SD
wandb sweep configs/sweep_class.yaml
wandb agent <sweep-id>
```

### SLURM

Wrap any pipeline command in a SLURM script (eg. DDPM):

```bash
#!/bin/bash
#SBATCH --job-name=intact-ddpm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

source activate ldm  # or your env
cd /path/to/InTAct-Unl/DDPM
export PYTHONPATH="${PYTHONPATH}:/path/to/InTAct-Unl"
python pipeline.py --config configs/pipeline.yaml
```

For sweep agents on SLURM, replace the last line with `wandb agent <sweep-id>`.

---

## Direct Script Usage

The original training and evaluation scripts remain available for finer-grained control.

<details>
<summary>Classification (manual)</summary>

```bash
cd Classification
export PYTHONPATH="${PYTHONPATH}:/path/to/InTAct-Unl"

# Forget class 0 (airplane)
python intact_experiment.py --unlearn_classes 0 --lambda_interval 100.0
```
</details>

<details>
<summary>DDPM (manual)</summary>

```bash
cd DDPM
python train.py --config configs/cifar10_intact.yml
```
</details>

<details>
<summary>Stable Diffusion (manual)</summary>

```bash
cd SD

# GA + InTAct
python train-scripts/intact_unlearn.py \
    --base_method ga --class_to_forget 0 \
    --targets to_q to_k to_v --lambda_interval 1.0 --epochs 5

# Generate + evaluate
python eval-scripts/generate-images.py --model_name "..." --prompts_path prompts/imagenette.csv --save_path evaluation/
python eval-scripts/compute-fid.py --folder_path evaluation/
python eval-scripts/imageclassify.py --prompts_path prompts/imagenette.csv --folder_path evaluation/
```
</details>

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `targets` | `["fc"]` | Layer name patterns to protect (e.g., `["to_q", "to_k", "to_v"]`) |
| `lambda_interval` | `10.0` | Weight for protection loss |
| `lower_percentile` | `0.05` | Lower bound of safe activation zone |
| `upper_percentile` | `0.95` | Upper bound of safe activation zone |
| `reduced_dim` | `32` | SVD dimensions (for efficiency) |
| `infinity_scale` | `20.0` | Scale for outer bounds (negative space) |
| `use_actual_bounds` | `False` | Use actual min/max from remain data instead of scaled bounds |
| `normalize_protection` | `True` | Normalize loss by number of layers |

## Project Structure

```
InTAct-Unl/
├── InTAct/
│   └── intact.py                          # Core InTAct implementation
├── Classification/
│   ├── pipeline.py                        # Unified pipeline (classwise + random)
│   ├── configs/
│   │   ├── pipeline_classwise.yaml
│   │   ├── pipeline_random.yaml
│   │   ├── sweep_classwise.yaml
│   │   └── sweep_random.yaml
│   ├── intact_experiment.py               # Standalone InTAct demo
│   ├── main_forget.py                     # Baseline unlearning methods
│   └── ...
├── DDPM/
│   ├── pipeline.py                        # Unified pipeline
│   ├── configs/
│   │   ├── pipeline.yaml
│   │   └── sweep.yaml
│   ├── train.py                           # Original training entry
│   ├── runners/diffusion.py               # InTAct integration
│   └── ...
├── SD/
│   ├── pipeline.py                        # Unified pipeline (class + NSFW)
│   ├── configs/
│   │   ├── pipeline_class.yaml
│   │   ├── pipeline_nsfw.yaml
│   │   ├── sweep_class.yaml
│   │   └── sweep_nsfw.yaml
│   ├── train-scripts/intact_unlearn.py    # SD InTAct (GA, RL, NSFW, ESD)
│   ├── eval-scripts/                      # FID, classify, NudeNet
│   └── ...
└── README.md
```

## License

MIT License