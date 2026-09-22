# NOTES — setup, environments, data, sweeps, SLURM

Companion to `README.md` (which lists what to run). This file tells you how to
prepare everything so a run actually works. Read `README.md` first.

## 0. Global prerequisites

- Python 3.8–3.11 (depending on experiment, see §1).
- A GPU machine for anything beyond tiny smoke runs.
- `PYTHONPATH` must include the **repo root** (so `import barrier`, `import
  ldm`, `import utils`, `import evaluation` resolve). Every run script does
  this for you; for manual runs:

  ```bash
  export PYTHONPATH="$PWD:$PYTHONPATH"          # from the repo root
  # e.g. cd Classification && export PYTHONPATH="$PWD/..:$PYTHONPATH"
  ```

- Cache redirection (`barrier/cache.py`) happens automatically on the first
  `import barrier` in every pipeline. Override the root with `CACHE_ROOT`.

## 1. Environments per experiment

| Experiment | Setup |
|-----------|-------|
| Classification | `cd Classification && pip install -r requirements.txt` |
| DDPM | `conda create -n salun-ddpm python=3.8 && conda activate salun-ddpm && pip install -r requirements.txt` |
| SD | `cd SD && conda env create -f environment.yaml && conda activate ldm` |
| Flux | `pip install -e ./diffusers[torch] && pip install -e ./peft && pip install tokenizers==0.20.0` (+ Rust toolchain); you also need an HF token with access to `black-forest-labs/FLUX.1-dev` |

No environment is shared between experiments on purpose (pinned stacks differ).

## 2. Data & model preparation

### Classification (CIFAR-10)

```bash
cd Classification
# 1. Train a base model (or point paths.pretrained_ckpt at an existing one)
python main_train.py --arch resnet18 --dataset cifar10 --lr 0.1 --epochs 182 --save_dir ./checkpoints
# 2. In configs/pipeline_classwise.yaml / pipeline_random.yaml set:
#    paths.model / paths.pretrained_ckpt, paths.data_dir, wandb.entity
```

### DDPM (CIFAR-10)

```bash
cd DDPM
# 1. Train the base DDPM (saves results/cifar10/<timestamp>/)
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/cifar10_train.yml --mode train
# 2. FID reference data (creates cifar10_without_label_0/)
python save_base_dataset.py --dataset cifar10 --label_to_forget 0
# 3. Evaluation classifier (saves cifar10_resnet34.pth)
CUDA_VISIBLE_DEVICES=0 python train_classifier.py --dataset cifar10
# 4. In configs/pipeline.yaml set:
#    paths.pretrained_ckpt_folder, paths.ref_dataset_dir,
#    paths.classifier_ckpt, model_config: configs/cifar10_intact.yml
```

`cifar10_intact*.yml` are **model/diffusion configs** used by both
`train.py` and the pipeline; they select the target layers (QKV, all-Linear,
+saliency, actual bounds, …).

### SD (Stable Diffusion v1.4)

```bash
cd SD
mkdir -p models/ldm/stable-diffusion-v1
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt \
  -O models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt
wget https://huggingface.co/CompVis/stable-diffusion-v1-4/raw/main/unet/config.json \
  -O diffusers_unet_config.json
```

- **Class forgetting**: Imagenette prompts are bundled (`prompts/imagenette.csv`);
  reference dataset auto-downloads via HuggingFace.
- **NSFW removal**: generate ~800 images with SD v1.4, e.g.
  `"a photo of a nude person"` → `data/nsfw/`, `"a photo of a person wearing clothes"` → `data/not-nsfw/`,
  then set `paths.nsfw_data` / `paths.not_nsfw_data` in `pipeline_nsfw.yaml`.
  The Flux pipelines can point at the **same two folders** (or HF dataset ids).
- Verify paths in `configs/pipeline_class.yaml`: `paths.sd_ckpt`,
  `paths.sd_config` (`configs/stable-diffusion/v1-intact.yaml`),
  `paths.diffusers_config`.
- Artist unlearning needs an artist prompt CSV (see `train-scripts/train_artists.py`).
- **SCaPre** (`SD/scapre/`) additionally needs an ImageNet-1K root; configure
  `scapre/configs/train_*.yaml`.

### Flux (FLUX.1-dev)

```bash
cd Flux
huggingface-cli login            # gated model `black-forest-labs/FLUX.1-dev`
# set $SCRATCH / CACHE_ROOT before running (model + ref data are large)
python intact_pipeline.py --config configs/intact/pipeline_concept.yaml   # concept erasure
python intact_pipeline.py --config configs/intact/pipeline_class.yaml     # class forgetting
python intact_pipeline.py --config configs/intact/pipeline_nsfw.yaml      # NSFW dataset erasure
```

## 3. wandb

Set your entity in every config you run (`entity: your-wandb-username`) or
export `WANDB_ENTITY=…`. Pipelines log metrics + artifacts; set
`--no-wandb` to run locally without logging.

## 4. Sweeps

Each experiment folder has `run_sweep.sh` + sweep YAMLs:

```bash
cd Classification && ./run_sweep.sh sweep_classwise   # also: sweep_random
cd DDPM         && ./run_sweep.sh sweep               # also: sweep_esd_bayes, sweep_kl_bayes, …
cd SD           && ./run_sweep.sh sweep_class         # also: sweep_nsfw, sweep_artists_lpips
cd Flux         && ./run_sweep.sh sweep_concept       # also: sweep_class, sweep_nsfw, sweep_nsfw_big
```

Sweep parameter keys are **dotted paths into the pipeline YAML**:

```yaml
parameters:
  unlearn.lr:              # → cfg["unlearn"]["lr"]
    values: [1e-5, 5e-5]
  intact.lambda_interval:  # → cfg["intact"]["lambda_interval"]
    values: [1.0, 10.0]
```

## 5. SLURM

Every `scripts/` / `scripts/slurm_*.sh` follows the same pattern:

```bash
#SBATCH --gres=gpu:1 --mem=48G --time=48:00:00
source activate <env>              # ldm / salun-ddpm / …
cd /path/to/BARRIER/<Experiment>
export PYTHONPATH="${PYTHONPATH}:/path/to/BARRIER"
python pipeline.py --config configs/<name>.yaml
```

Sweep agents on SLURM: replace the last line with `wandb agent <sweep-id>`
and launch one array job per agent. Note: several SLURM scripts hard-code the
cluster checkout path (`/home/miksa/InTAct-Unl/…`) — adjust to your machines.

## 6. Testing commands (first-run smoke)

| Test | Command | Watch for |
|------|---------|-----------|
| Classification class-wise | `cd Classification && python pipeline.py --config configs/pipeline_classwise.yaml` | `acc/forget` ↓, `acc/retain`+`acc/test` high, UA, MIA |
| Classification random | `python pipeline.py --config configs/pipeline_random.yaml` | same metrics |
| DDPM class forgetting | `cd DDPM && python pipeline.py --config configs/pipeline.yaml` | `fid` low for remain, classifier entropy high on forgotten class |
| SD class forgetting | `cd SD && python pipeline.py --config configs/pipeline_class.yaml` | `fid`, `classify/accuracy` |
| SD NSFW removal | `python pipeline.py --config configs/pipeline_nsfw.yaml` | `nudenet/nude_ratio` low |
| Flux concept erasure | `cd Flux && python intact_pipeline.py --config configs/intact/pipeline_concept.yaml` | NudeNet I2P counts ↓ |

## 7. Ablation experiments (Exp 1–9)

The mechanism/design-choice ablations live in the shared `barrier/` package:

```bash
# Exp 3–9 grids are defined in barrier/expgrid.py
python barrier/expgrid.py ddpm --count                 # DDPM grid size
python barrier/expgrid.py resnet18-classwise <i>       # single grid row → CLI flags

# Runners (headless, no wandb):
python DDPM/ablation_regions.py --help
python Classification/ablation_cls.py --help
# Post-hoc analysis (Exp 1–2):
python barrier/ablation_mechanism.py --run_dir <runs/<suffix>>
```

Existing ablation-results notes live in `Classification/results/README.md` and
`DDPM/results/README.md`. `benchmark_timing/` is a separate wall-time/VRAM
comparison harness (`sbatch benchmark_timing/run_timing_benchmark.sh`).

## 8. Troubleshooting

- **`ModuleNotFoundError: ldm / utils / evaluation`** → the repo root is not on
  `PYTHONPATH` (see §0).
- **Caches fill home dir** → every entrypoint imports `barrier` (which runs
  `barrier.cache`); set `CACHE_ROOT`/`SCRATCH` for shared storage.
- **torchmetrics compat error on checkpoint load** → `barrier.cache` installs
  a shim automatically; if it disappears, re-check that `import barrier`
  happens before any `torchmetrics`/`pytorch-lightning` import.
- **HF gated-model access denied (Flux)** → `huggingface-cli login`, accept
  the license on the model page, token with `read` scope.
- **Git ignored weights**: `*.pth` / `*.safetensors` are ignored for new files;
  existing tracked ones (e.g. `DDPM.pth`, Flux reference LoRA) stay in git.