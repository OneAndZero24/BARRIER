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

## Usage

### Classification (CIFAR-10)

```bash
cd Classification

# Forget class 0 (airplane)
python intact_experiment.py --unlearn_classes 0 --lambda_interval 100.0

# Forget multiple classes
python intact_experiment.py --unlearn_classes 0 1 2 --unlearn_epochs 20

# Custom targets (protect specific layers)
python intact_experiment.py --targets fc layer4 --lambda_interval 50.0
```

### DDPM (Diffusion Models)

```bash
cd DDPM

# Forget class 0 with InTAct
python train.py --config configs/cifar10_intact.yml
```

### Stable Diffusion

```bash
cd SD

# Download SD v1.4 weights first (see SD/README.md)

# Forget class 0 with Gradient Ascent + InTAct
python train-scripts/intact_unlearn.py \
    --base_method ga \
    --class_to_forget 0 \
    --train_method xattn \
    --lambda_interval 1.0 \
    --targets to_q to_k to_v

# Random Label + InTAct
python train-scripts/intact_unlearn.py \
    --base_method rl \
    --class_to_forget 0 \
    --train_method xattn

# NSFW removal + InTAct
python train-scripts/intact_unlearn.py \
    --base_method nsfw \
    --train_method xattn

# ESD (prompt-based) + InTAct
python train-scripts/intact_unlearn.py \
    --base_method esd \
    --prompt "nudity" \
    --train_method xattn \
    --iterations 1000
```

**SD Base Methods:**

| Method | Description | Use Case |
|--------|-------------|----------|
| `ga` | Gradient Ascent | Class forgetting |
| `rl` | Random Label | Class forgetting |
| `nsfw` | NSFW removal | Concept removal |
| `esd` | Erased Stable Diffusion | Prompt-based concept removal |

## Project Structure

```
InTAct-Unl/
├── InTAct/
│   ├── intact.py              # Core InTAct implementation
│   └── README.md
├── Classification/
│   ├── intact_experiment.py   # Classification demo (CIFAR-10)
│   ├── main_forget.py         # Other unlearning baselines
│   └── ...
├── DDPM/
│   ├── train.py               # DDPM training with InTAct
│   ├── runners/diffusion.py   # InTAct integration
│   ├── configs/
│   │   ├── cifar10_intact.yml
│   │   └── ...
│   └── ...
├── SD/
│   ├── train-scripts/
│   │   ├── intact_unlearn.py  # SD InTAct (GA, RL, NSFW, ESD)
│   │   └── ...
│   ├── configs/
│   │   └── stable-diffusion/v1-intact.yaml
│   └── ...
└── README.md
```

## License

MIT License