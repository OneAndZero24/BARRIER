#!/usr/bin/env python3
"""
Artist Unlearning + LPIPS Evaluation Pipeline
=============================================
Delegates per-artist InTAct training to train_artists.py via subprocess,
then computes:
  LPIPS_e  - LPIPS on erased artist (baseline vs trained model)
  LPIPS_u  - LPIPS on unerased artists (should stay low)
  LPIPS_d  = LPIPS_e - LPIPS_u (unlearning success metric)

Usage:
  python scripts/lpips_eval_pipeline.py --config configs/pipeline_artists.yaml
"""

from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
rece_path = root / 'rece'
train_scripts_path = root / 'train-scripts'
sys.path.insert(0, str(root))
sys.path.insert(0, str(rece_path))
sys.path.insert(0, str(train_scripts_path))

import argparse
import time
import os
import subprocess
import json
import yaml
import pandas as pd
import numpy as np
import torch
import lpips
import cv2
from tqdm import tqdm
import tempfile
import shutil

# Use isolated temp dirs for wandb to avoid cleanup conflicts.
temp_wandb_dir = tempfile.mkdtemp(prefix='wandb_tmp_')
os.environ['TMPDIR'] = temp_wandb_dir
wandb_dir = tempfile.mkdtemp(prefix='wandb_dir_')
os.environ['WANDB_DIR'] = wandb_dir

import wandb


# ---- helpers ----------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='Train + LPIPS eval pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='path to config YAML')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    return parser.parse_known_args()[0]


def merge_wandb_config(args, config):
    """Override YAML defaults with explicit CLI flags."""
    if args.config is None:
        return config
    explicit = set()
    for a in sys.argv[1:]:
        if a.startswith('--'):
            k = a.lstrip('-').replace('-', '_')
            if '=' in k:
                k = k.split('=')[0]
            explicit.add(k)
    if config is None:
        return None
    for section in ['unlearn', 'intact', 'paths', 'pipeline']:
        if section in config and config[section]:
            for k in list(config[section].keys()):
                if k in explicit:
                    config[section].pop(k, None)
    return config


def get_img_pairs(d_baseline: str, d_train: str, files: list):
    """Return list of (baseline_path, trained_path, filename) tuples."""
    pairs = []
    base_dir = os.path.join(d_baseline, 'imgs')
    train_dir = os.path.join(d_train, 'imgs')
    for fn in files:
        fp_b  = os.path.join(base_dir,  fn)
        fp_t  = os.path.join(train_dir, fn)
        if os.path.exists(fp_b) and os.path.exists(fp_t):
            pairs.append((fp_b, fp_t, fn))
    return pairs


def compute_lpips_between_dirs(dir0: str, dir1: str, loss_fn, filter_files=None):
    """Compare images in dir0 vs dir1. Returns dict {filename: lpips_value}."""
    imgs0 = os.path.join(dir0, 'imgs') if os.path.isdir(os.path.join(dir0, 'imgs')) else dir0
    imgs1 = os.path.join(dir1, 'imgs') if os.path.isdir(os.path.join(dir1, 'imgs')) else dir1

    files = sorted([f for f in os.listdir(imgs0) if f.endswith('.png')])
    if filter_files is not None:
        files = [f for f in files if f in filter_files]

    dists = {}
    for f in tqdm(files, desc=f'LPIPS {os.path.basename(dir1)}'):
        p0 = os.path.join(imgs0, f)
        p1 = os.path.join(imgs1, f)
        if not os.path.exists(p1):
            continue
        img0 = lpips.load_image(p0)
        img0 = cv2.resize(img0, (64, 64))
        img1 = lpips.load_image(p1)
        img1 = cv2.resize(img1, (64, 64))
        t0 = lpips.im2tensor(img0).cuda()
        t1 = lpips.im2tensor(img1).cuda()
        dists[f] = loss_fn.forward(t0, t1).item()
    return dists


def filter_prompts_by_artist(df_csv: str, artist_name: str):
    """Return list of filenames that belong to *artist_name*."""
    df = pd.read_csv(df_csv)
    artist_lower = artist_name.lower()
    matches = df[df['artist'].str.lower().str.contains(artist_lower)]
    if matches.empty:
        tokens = artist_lower.replace('-', ' ').split()
        candidate = df[df['artist'].apply(lambda a: any(t in a.lower() for t in tokens if len(t) > 2))]
        matches = candidate
    base_nums = sorted(set(matches['case_number']))
    return sorted([f'{cn}_0.png' for cn in base_nums])


def build_train_cmd(cfg, artist, sweep_overrides):
    """Build the command list for a single train_artists.py invocation."""
    un_cfg = cfg.get('unlearn', {}) or {}
    path_cfg = cfg.get('paths', {}) or {}
    int_cfg = cfg.get('intact', {}) or {}
    pipe_cfg = cfg.get('pipeline', {}) or {}

    cmd = [
        'python', 'train-scripts/train_artists.py',
        '--config', str(Path(__file__).resolve().parents[1] / 'configs' / 'pipeline_artists.yaml'),
        '--concepts', artist,
        '--concept_type', un_cfg.get('concept_type', 'art'),
        '--seed', str(pipe_cfg.get('seed', 42)),
        '--epochs', str(sweep_overrides.get('unlearn.epochs', un_cfg.get('epochs', 100))),
        '--lr', str(sweep_overrides.get('unlearn.lr', un_cfg.get('lr', 1e-5))),
        '--test_csv_path', path_cfg.get('test_csv_path', 'rece/dataset/artists1734_prompts.csv'),
    ]

    # Guided concepts
    gc = un_cfg.get('guided_concepts')
    if gc is not None:
        if isinstance(gc, list):
            gc = ','.join(gc)
        cmd += ['--guided_concepts', gc]

    # Preserve concepts
    pc = un_cfg.get('preserve_concepts')
    if pc is not None:
        if isinstance(pc, list):
            pc = ','.join(pc)
        cmd += ['--preserve_concepts', pc]

    cmd += [
        '--technique', un_cfg.get('technique', 'replace'),
        '--preserve_scale', str(un_cfg.get('preserve_scale', 0.1)),
        '--erase_scale', str(un_cfg.get('erase_scale', 1.0)),
        '--lamb', str(un_cfg.get('lamb', 0.1)),
        '--regular_scale', str(sweep_overrides.get('unlearn.regular_scale', un_cfg.get('regular_scale', 1e-3))),
        '--num_samples', str(un_cfg.get('num_samples', 1)),
        '--ddim_steps', str(un_cfg.get('ddim_steps', 50)),
    ]

    # Model paths
    sd_ckpt = path_cfg.get('sd_ckpt')
    if sd_ckpt:
        cmd += ['--sd_ckpt', sd_ckpt]
    sd_config_p = path_cfg.get('sd_config')
    if sd_config_p:
        cmd += ['--sd_config', sd_config_p]
    target_ckpt = path_cfg.get('target_ckpt', '')
    if target_ckpt:
        cmd += ['--target_ckpt', target_ckpt]
    save_path = path_cfg.get('save_path')
    if save_path:
        cmd += ['--save_path', save_path]

    # InTAct flags
    cmd.append('--intact')
    lambda_interval = sweep_overrides.get('intact.lambda_interval', int_cfg.get('lambda_interval', 1.0))
    cmd += ['--lambda_interval', str(lambda_interval)]

    for flag, key in [
        ('--lower_percentile', 'lower_percentile'),
        ('--upper_percentile', 'upper_percentile'),
        ('--infinity_scale', 'infinity_scale'),
    ]:
        val = int_cfg.get(key)
        if val is not None:
            cmd += [flag, str(val)]

    reduced_dim = sweep_overrides.get('intact.reduced_dim', int_cfg.get('reduced_dim'))
    if reduced_dim is not None:
        cmd += ['--reduced_dim', str(int(reduced_dim))]

    bool_flags = ['use_actual_bounds', 'normalize_protection']
    for key in bool_flags:
        if int_cfg.get(key, True):
            cmd.append(f'--{key}')

    # Target blocks (from YAML or sweep, never overridden by k)
    tb = sweep_overrides.get('intact.target_blocks', int_cfg.get('target_blocks'))
    if tb is not None:
        if isinstance(tb, str):
            try:
                import ast
                tb = ast.literal_eval(tb)
            except Exception:
                tb = [b.strip() for b in tb.split(',')]
        cmd += ['--target_blocks'] + [str(b) for b in tb]

    # k controls which cross-attention layers are targeted:
    #   k=1 → to_q only,  k=2 → to_q,to_k,  k=3 → to_q,to_k,to_v
    LAYER_POOL = ['attn2.to_q', 'attn2.to_k', 'attn2.to_v']
    k = sweep_overrides.get('intact.k')
    if k is not None:
        selected_layers = LAYER_POOL[:min(int(k), len(LAYER_POOL))]
        cmd += ['--target_layers'] + [str(l) for l in selected_layers]
    elif int_cfg.get('target_layers') is not None:
        cmd += ['--target_layers'] + [str(l) for l in int_cfg['target_layers']]

    return cmd


# ---- main -------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()

    # Load config or use defaults
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        cfg = merge_wandb_config(args, cfg) or {}
    else:
        cfg = {
            'unlearn': {},
            'intact': {},
            'paths': {},
            'pipeline': {},
        }

    un_cfg   = cfg.get('unlearn',  {}) or {}
    path_cfg = cfg.get('paths',   {}) or {}
    int_cfg  = cfg.get('intact',   {}) or {}
    pipe_cfg = cfg.get('pipeline', {}) or {}
    wb_cfg   = cfg.get('wandb',    {}) or {}

    concepts       = un_cfg.get('concepts', ['Kelly McKernan', 'Van Gogh'])
    concept_type   = un_cfg.get('concept_type', 'art')
    epochs_yaml    = un_cfg.get('epochs', 100)
    lr_yaml        = un_cfg.get('lr', 1e-5)
    regular_scale_yaml = un_cfg.get('regular_scale', 1e-3)
    eval_prompts_sample = un_cfg.get('eval_prompts_per_artist', None)

    test_csv       = path_cfg.get('test_csv_path', 'rece/dataset/artists1734_prompts.csv')
    save_root      = path_cfg.get('save_path', '/shared/results/common/miksa/intact/SD/lpips_eval')

    seed           = pipe_cfg.get('seed', 42)

    # ---- wandb init -------------------------------------------------------
    run_name = args.wandb_run_name or f'lpips-{"-".join(c.lower().replace(" ","_") for c in concepts)}'
    run = wandb.init(
        project=wb_cfg.get('project', 'intact-sd'),
        entity=wb_cfg.get('entity', 'oneandzero24'),
        name=run_name,
        group=wb_cfg.get('group', 'artist-lpips'),
        tags=list(set(wb_cfg.get('tags', ['sd','artist','lpips'])).union(['lpips'])),
        config={
            'concepts': concepts,
            'concept_type': concept_type,
            'epochs': epochs_yaml,
            'lr': lr_yaml,
            'regular_scale': regular_scale_yaml,
            **int_cfg,
        },
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- override with sweep hyperparameters --------------------------------
    sweep_params = {}
    unlearn_sweep = run.config.get('unlearn', {})
    intact_sweep = run.config.get('intact', {})
    if isinstance(unlearn_sweep, dict) and 'epochs' in unlearn_sweep:
        sweep_params['unlearn.epochs'] = unlearn_sweep['epochs']
    elif run.config.get('unlearn.epochs') is not None:
        sweep_params['unlearn.epochs'] = run.config['unlearn.epochs']
    if isinstance(unlearn_sweep, dict) and 'lr' in unlearn_sweep:
        sweep_params['unlearn.lr'] = unlearn_sweep['lr']
    elif run.config.get('unlearn.lr') is not None:
        sweep_params['unlearn.lr'] = run.config['unlearn.lr']
    if isinstance(unlearn_sweep, dict) and 'regular_scale' in unlearn_sweep:
        sweep_params['unlearn.regular_scale'] = unlearn_sweep['regular_scale']
    elif run.config.get('unlearn.regular_scale') is not None:
        sweep_params['unlearn.regular_scale'] = run.config['unlearn.regular_scale']
    if isinstance(intact_sweep, dict) and 'lambda_interval' in intact_sweep:
        sweep_params['intact.lambda_interval'] = intact_sweep['lambda_interval']
    elif run.config.get('intact.lambda_interval') is not None:
        sweep_params['intact.lambda_interval'] = run.config['intact.lambda_interval']
    if isinstance(intact_sweep, dict) and 'k' in intact_sweep:
        sweep_params['intact.k'] = int(intact_sweep['k'])
    elif run.config.get('intact.k') is not None:
        sweep_params['intact.k'] = int(run.config['intact.k'])
    if isinstance(intact_sweep, dict) and 'reduced_dim' in intact_sweep:
        sweep_params['intact.reduced_dim'] = int(intact_sweep['reduced_dim'])
    elif run.config.get('intact.reduced_dim') is not None:
        sweep_params['intact.reduced_dim'] = int(run.config['intact.reduced_dim'])

    effective_epochs = sweep_params.get('unlearn.epochs', epochs_yaml)
    effective_lr = sweep_params.get('unlearn.lr', lr_yaml)
    effective_regular_scale = sweep_params.get('unlearn.regular_scale', regular_scale_yaml)

    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.to(device)

    dev_df = pd.read_csv(test_csv)
    all_files = sorted([f for f in os.listdir(os.path.join(save_root, 'baseline_imgs')) if f.endswith('.png')]) if os.path.isdir(os.path.join(save_root, 'baseline_imgs')) else None

    # ---- per-artist training + eval loop -----------------------------------
    results_rows = []

    for artist in concepts:
        t0_full = time.time()
        safe_name = artist.replace(' ', '_').lower()
        print(f'\n========== Training on {artist} ==========')

        run_path = os.path.join(save_root, 'lpips_eval', concept_type, safe_name)
        os.makedirs(run_path, exist_ok=True)

        # --- Build and run training command ---------------------------------
        cmd = build_train_cmd(cfg, artist, sweep_params)

        # Override save_path for this artist so that train_artists.py writes into our slot
        # The --save_path flag was already set; we need to point to our run_path.
        # Find the index of --save_path in the command and replace the value after it.
        for i, elem in enumerate(cmd):
            if elem == '--save_path':
                cmd[i + 1] = save_root

        # Also append a custom tag to the sub-command's output paths by setting env var
        env_override = os.environ.copy()
        cwd = str(Path(__file__).resolve().parents[1])

        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env_override,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if proc.returncode != 0:
            print(f'Training failed for {artist} (exit code {proc.returncode})')
            print(proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
            continue

        # Find the latest train output dir containing both "before" and "final".
        # train_artists.py writes to: save_path/{concept_type}/intact/...
        trainer_base = str(Path(save_root) / concept_type / 'intact')
        if os.path.isdir(trainer_base):
            candidates = []
            for d in Path(trainer_base).rglob('*'):
                if (d / 'before').is_dir() and (d / 'final').is_dir():
                    candidates.append(d)
            if not candidates:
                print(f'Could not locate train output for {artist} under {trainer_base}')
                continue
            # Pick the most recently modified
            train_result_dir = str(max(candidates, key=lambda p: p.stat().st_mtime))
        else:
            print(f'Trainer base dir missing: {trainer_base}')
            continue

        # Ensure baseline images exist (generate once from fresh model if needed)
        before_dir = os.path.join(train_result_dir, 'before')
        final_dir  = os.path.join(train_result_dir, 'final')

        total_time = time.time() - t0_full

        # ---- LPIPS evaluation -----------------------------------------------
        erased_matched = filter_prompts_by_artist(test_csv, artist)
        if eval_prompts_sample is not None and len(erased_matched) > eval_prompts_sample:
            import random as _rnd
            _rnd.seed(seed)
            erased_matched = _rnd.sample(erased_matched, eval_prompts_sample)
        
        all_before_files = sorted([f for f in os.listdir(os.path.join(before_dir, 'imgs')) if f.endswith('.png')])
        unerased_matched = [f for f in all_before_files if f not in erased_matched]
        if eval_prompts_sample is not None and len(unerased_matched) > eval_prompts_sample:
            import random as _rnd2
            _rnd2.seed(seed)
            unerased_matched = _rnd2.sample(unerased_matched, eval_prompts_sample)

        lpips_e = compute_lpips_between_dirs(before_dir, final_dir, loss_fn, filter_files=erased_matched)
        lpips_u = compute_lpips_between_dirs(before_dir, final_dir, loss_fn, filter_files=unerased_matched)

        avg_e = np.mean(list(lpips_e.values())) if lpips_e else 0.0
        avg_u = np.mean(list(lpips_u.values())) if lpips_u else 0.0
        delta = avg_e - avg_u

        print(f'{artist}: LPIPS_e={avg_e:.6f}  LPIPS_u={avg_u:.6f}  LPIPS_d={delta:.6f}')

        # ---- wandb logging ---------------------------------------------------
        run.log({
            f'LPIPS_e/{artist}': avg_e,
            f'LPIPS_u/{artist}': avg_u,
            f'LPIPS_d/{artist}': delta,
            'hyperparams/lr': effective_lr,
            'hyperparams/epochs': effective_epochs,
            'hyperparams/regular_scale': effective_regular_scale,
            'time/total_sec': total_time,
        })

        e_rows = [[fn, f'{v:.6f}'] for fn, v in lpips_e.items()]
        u_rows = [[fn, f'{v:.6f}'] for fn, v in lpips_u.items()]
        run.log({
            f'lpips_e_table/{artist}': wandb.Table(data=e_rows, columns=['file', 'LPIPS']),
            f'lpips_u_table/{artist}': wandb.Table(data=u_rows, columns=['file', 'LPIPS']),
        })

        def get_img_pairs(d_baseline: str, d_train: str, files: list):
            pairs = []
            base_dir = os.path.join(d_baseline, 'imgs')
            train_dir = os.path.join(d_train, 'imgs')
            for fn in files:
                fp_b  = os.path.join(base_dir,  fn)
                fp_t  = os.path.join(train_dir, fn)
                if os.path.exists(fp_b) and os.path.exists(fp_t):
                    pairs.append((fp_b, fp_t, fn))
            return pairs

        erased_pairs = get_img_pairs(before_dir, final_dir, erased_matched)
        if erased_pairs:
            tb_e = wandb.Table(columns=['index', 'prompt', 'baseline', 'unlearned'])
            for idx, (fp_b, fp_t, fn) in enumerate(erased_pairs):
                tb_e.add_data(idx, fn, wandb.Image(fp_b), wandb.Image(fp_t))
            run.log({f'comparison_erased/{artist}': tb_e})

        unerased_pairs = get_img_pairs(before_dir, final_dir, unerased_matched)
        if unerased_pairs:
            tb_u = wandb.Table(columns=['index', 'prompt', 'baseline', 'unlearned'])
            for idx, (fp_b, fp_t, fn) in enumerate(unerased_pairs):
                tb_u.add_data(idx, fn, wandb.Image(fp_b), wandb.Image(fp_t))
            run.log({f'comparison_uerased/{artist}': tb_u})

        results_rows.append({
            'artist': artist,
            'LPIPS_e': round(avg_e, 6),
            'LPIPS_u': round(avg_u, 6),
            'LPIPS_d': round(delta, 6),
            'lr': effective_lr,
            'epochs': effective_epochs,
            'regular_scale': effective_regular_scale,
        })

    # ---- summary --------------------------------------------------------
    if results_rows:
        mean_d = np.mean([r['LPIPS_d'] for r in results_rows])
        run.log({'mean_LPIPS_d': mean_d})
        run.summary['mean_LPIPS_d'] = mean_d
    else:
        run.summary['mean_LPIPS_d'] = float('nan')

    # Save results to CSV for offline inspection
    if results_rows:
        results_df = pd.DataFrame(results_rows)
        csv_path = os.path.join(wandb_dir, 'lpips_results.csv')
        results_df.to_csv(csv_path, index=False)
        run.save(csv_path)

    run.finish()
    shutil.rmtree(temp_wandb_dir, ignore_errors=True)
    shutil.rmtree(wandb_dir, ignore_errors=True)
    print('\nPipeline complete. Results logged to wandb.')
