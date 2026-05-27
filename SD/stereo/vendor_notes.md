# Vendored STEREO Notes (robust-concept-erasing)

Source snapshot: https://github.com/koushiksrivats/robust-concept-erasing (main branch, tarball fetched on 2026-05-27)

## What was vendored

- `train.py` -> `sd/stereo/scripts/train_vendor.py`
- `utils/stereo.py` -> `sd/stereo/attacks/stereo_vendor.py`
- `utils/utils.py` -> `sd/stereo/vendor/utils_vendor.py`
- `utils/dataset.py` -> `sd/stereo/vendor/dataset_vendor.py`
- `utils/apg.py` -> `sd/stereo/vendor/apg_vendor.py`
- `utils/anchor_prompts.json` -> `sd/stereo/configs/anchor_prompts.json`
- `generate_images.py` -> `sd/stereo/evaluation/generate_images_vendor.py`

Only import/path fixes were applied. Attack logic is unchanged.

## Compatibility assumptions (observed)

1. Stable Diffusion base version
- Uses SD v1.4 components from HuggingFace IDs:
  - `CompVis/stable-diffusion-v1-4` (UNet, VAE, scheduler, feature extractor, safety checker)
  - `openai/clip-vit-large-patch14` (tokenizer, text encoder)

2. Checkpoint loading model
- Core assumption is a diffusers-compatible UNet `state_dict`.
- Attack code does direct `diffuser.unet.load_state_dict(torch.load(path))`.
- No BARRIER-specific checkpoint logic exists in upstream attack code.

3. UNet structure assumptions
- Fine-tuning target selection is pattern-based in `FineTunedModel` and expects SD1.x UNet naming (`attn1`, `attn2`, `to_q`, `to_k`, `to_v`, etc.).
- `train_method` switches module subsets (`xattn`, `noxattn`, `selfattn`, `full_unet`, ...).

4. Text encoder assumptions
- Textual inversion attack optimizes token embeddings in CLIP text encoder (`clip-vit-large-patch14`).
- Placeholder tokens are inserted into tokenizer and corresponding embedding rows are trained.

5. Attack entrypoints
- Main script arguments in `sd/stereo/scripts/train_vendor.py`.
- Attack-only mode calls `attack_stereo(args, diffuser)`.
- `attack_stereo` requires:
  - `output_dir`
  - `unet_ckpt_to_attack` (filename relative to `output_dir`)
  - `attack_eval_images` (gallery image directory)
  - `initializer_token`, `ci_lr`, `ti_max_train_steps`, `learnable_property`

6. Config flow
- Upstream is argparse-driven, no yaml config loader.
- BARRIER integration wraps this with experiment-level configs in `SD/experiments/table2`.

7. Output format from vendored attack path
- Attack textual inversion checkpoint: `eval_ci_attack_on_stereo_text_encoder.pt`
- Generated images: `<output_dir>/eval_ci_iteration/eval_ci_attack_image_placeholder_<token>_<i>.png`

## Important Table 2 caveat

Upstream STEREO repository does **not** include implementations of UD/RAB/CCE attacks.
It explicitly points to external repositories for those evaluations.

For publication-level Table 2 reproducibility in BARRIER, we should:
- Keep this vendored STEREO attack path as-is.
- Add wrappers/scripts that call pinned commits of UD, RAB, and CCE (vendored or git-subtree in future step).

## External attack code now vendored

Additional repositories are vendored under `SD/stereo/attacks/vendors/`:
- `unlearndiffatk` (OPTML-Group/Diffusion-MU-Attack)
- `ring-a-bell` (chiayi-hsu/Ring-A-Bell)
- `cce` (NYU-DICE-Lab/circumventing-concept-erasure)

Bootstrap/update script:
- `SD/stereo/attacks/vendors/fetch_external_attacks.sh`

Unified launcher:
- `SD/stereo/scripts/run_external_attacks.py`
