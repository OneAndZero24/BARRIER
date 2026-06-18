#!/usr/bin/env python3
"""
Artist Unlearning + LPIPS Evaluation Pipeline
=============================================
Trains unlearning for each artist separately, computes:
  LPIPS_e  - LPIPS on erased artist (baseline vs erased model)
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
import os
# Use a unique temporary directory for wandb to avoid cleanup conflicts.
# This avoids shared TMPDIR usage that can cause "Directory not empty" errors.
import tempfile
temp_wandb_dir = tempfile.mkdtemp(prefix='wandb_tmp_')
os.environ['TMPDIR'] = temp_wandb_dir
# No need to manually create; mkdtemp already does.

import argparse
import time
import os
import shutil
import json
import pandas as pd
import numpy as np
import torch
import lpips
import cv2
os.environ.setdefault('WANDB_DISABLE_SERVICE', 'true')
import wandb
from tqdm import tqdm

# ---- helpers ----------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='Train + LPIPS eval pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='path to config YAML')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    return parser.parse_args()

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



def compute_lpips_between_dirs(dir0: str, dir1: str, loss_fn, filter_files=None):
    """
    Replicate lpips_score.py logic: compare images in dir0 vs dir1.
    Returns dict {filename: lpips_value}.
    """
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
    """Return list of filenames that belong to *artist_name*.
    Uses substring matching in both directions since the CSV may store
    full names ('Vincent Van Gogh') while config uses short forms ('Van Gogh')."""
    df = pd.read_csv(df_csv)
    artist_lower = artist_name.lower()
    matches = df[df['artist'].str.lower().str.contains(artist_lower)]
    if matches.empty:
        # Fallback: search for any CSV artist name that contains a token from artist_name
        tokens = artist_lower.replace('-', ' ').split()
        candidate = df[df['artist'].apply(lambda a: any(t in a.lower() for t in tokens if len(t) > 2))]
        matches = candidate
    base_nums = sorted(set(matches['case_number']))
    return sorted([f'{cn}_0.png' for cn in base_nums])

# ---- main -------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()

    # Load config or use defaults
    import yaml
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

    un_cfg  = cfg.get('unlearn',  {}) or {}
    path_cfg = cfg.get('paths',   {}) or {}
    int_cfg = cfg.get('intact',   {}) or {}
    pipe_cfg = cfg.get('pipeline',{}) or {}
    wb_cfg  = cfg.get('wandb',    {}) or {}

    concepts       = un_cfg.get('concepts', ['Kelly McKernan', 'Van Gogh'])
    concept_type   = un_cfg.get('concept_type', 'art')
    guided_concepts = un_cfg.get('guided_concepts', None)
    technique      = un_cfg.get('technique', 'replace')
    preserve_scale = un_cfg.get('preserve_scale', 0.1)
    erase_scale    = un_cfg.get('erase_scale', 1.0)
    lamb           = un_cfg.get('lamb', 0.1)
    emb_computing  = un_cfg.get('emb_computing', 'close_regzero')
    reg_item       = un_cfg.get('reg_item', '1st')
    regular_scale  = un_cfg.get('regular_scale', 1e-3)
    num_samples            = un_cfg.get('num_samples', 1)
    ddim_steps             = un_cfg.get('ddim_steps', 50)
    epochs                 = un_cfg.get('epochs', 3)
    lr                     = un_cfg.get('lr', 1e-5)
    use_intact             = un_cfg.get('intact', True)
    inner_iters            = un_cfg.get('inner_iterations', 250)
    eval_prompts_sample    = un_cfg.get('eval_prompts_per_artist', None)

    sd_ckpt        = path_cfg.get('sd_ckpt', 'models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    sd_config      = path_cfg.get('sd_config', 'configs/stable-diffusion/v1-intact.yaml')
    target_ckpt    = path_cfg.get('target_ckpt', '')
    test_csv       = path_cfg.get('test_csv_path', 'rece/dataset/artists1734_prompts.csv')
    save_root      = path_cfg.get('save_path', '/shared/results/common/miksa/intact/SD/lpips_eval')

    seed           = pipe_cfg.get('seed', 42)
    device         = f'cuda:{pipe_cfg.get("device", "0")}'

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
            'guided_concepts': guided_concepts,
            'technique': technique,
            'emb_computing': emb_computing,
            'epochs': epochs,
            'inner_iterations': inner_iters,
            'lr': lr,
            'regular_scale': regular_scale,
            'use_intact': use_intact,
            **int_cfg,
        },
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.to(device)

    # ---- load base pipeline once ------------------------------------------
    from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer
    from omegaconf import OmegaConf

    def load_pipeline(ckpt_p):
        if ckpt_p.endswith('.ckpt'):
            checkpoint = torch.load(ckpt_p, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            orig_cfg = OmegaConf.load(sd_config)
            from convertModels import (create_vae_diffusers_config, create_unet_diffusers_config,
                                      convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint,
                                      convert_ldm_clip_checkpoint)
            vae_cfg = create_vae_diffusers_config(orig_cfg, image_size=512)
            vae = AutoencoderKL(**vae_cfg)
            vae.load_state_dict(convert_ldm_vae_checkpoint(checkpoint, vae_cfg))
            unet_cfg = create_unet_diffusers_config(orig_cfg, image_size=512)
            unet_cfg['upcast_attention'] = False
            unet = UNet2DConditionModel(**unet_cfg)
            unet.load_state_dict(convert_ldm_unet_checkpoint(checkpoint, unet_cfg))
            tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
            text_encoder = convert_ldm_clip_checkpoint(checkpoint)
            scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                              beta_schedule='scaled_linear', num_train_timesteps=1000)
            pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                           unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=None)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(ckpt_p)
        pipe.to(device)
        return pipe

    base_pipe  = load_pipeline(sd_ckpt)
    tokenizer  = base_pipe.tokenizer

    dev_df = pd.read_csv(test_csv)

    # ---- per-artist training + eval loop -----------------------------------
    results_rows = []

    for artist in concepts:
        t0_full = time.time()
        print(f'\n========== Training on {artist} ==========')

        # Save path for this artist
        run_path = os.path.join(save_root, 'lpips_eval', concept_type, artist.replace(' ','_').lower())
        os.makedirs(run_path, exist_ok=True)

        from execs import generate_images
        from erase_methods import edit_model_adversarial
        from utils.embedding_calculation import close_form_emb, close_form_emb_regzero
        from InTAct.intact import UnlearnIntervalProtection

        pipe = load_pipeline(sd_ckpt)
        pipe_copy = load_pipeline(sd_ckpt)

        # Generate baseline images before training
        generate_images(pipe, dev_df, f'{run_path}/before', ddim_steps=ddim_steps, num_samples=num_samples)

        # Load UCE checkpoint if provided
        if target_ckpt:
            pipe.unet.load_state_dict(torch.load(target_ckpt))
            pipe.to(device)
            generate_images(pipe, dev_df, f'{run_path}/uce', ddim_steps=ddim_steps, num_samples=num_samples)

        # Build old/new text pairs (single concept this iteration)
        old_texts = [artist]
        if guided_concepts is None:
            new_texts = [' ']
        else:
            gc = [c.strip() for c in guided_concepts.split(',')] if isinstance(guided_concepts, str) else guided_concepts
            new_texts = [gc[0]] if len(gc) == 1 else gc

        # Preserve all other artists
        all_artists = list(pd.read_csv('rece/dataset/artists1734_prompts.csv')['artist'].unique())
        old_lower = [t.lower() for t in old_texts]
        preserve_concepts = [a for a in all_artists if a.lower() not in old_lower]
        retain_texts = [''] + preserve_concepts

        # ---- training loop --------------------------------------------------
        start = time.time()
        adv_df_i = pd.DataFrame([{'prompt': artist, 'evaluation_seed': seed}])

        if use_intact:
            int_targets     = int_cfg.get('targets', ['attn2.to_q','attn2.to_k','attn2.to_v'])
            lambda_interval = int_cfg.get('lambda_interval', 1.0)
            lower_pct       = int_cfg.get('lower_percentile', 0.05)
            upper_pct       = int_cfg.get('upper_percentile', 0.95)
            reduced_dim     = int_cfg.get('reduced_dim', 64)
            inf_scale       = int_cfg.get('infinity_scale', 18.0)
            use_actual      = int_cfg.get('use_actual_bounds', True)
            norm_prot       = int_cfg.get('normalize_protection', True)

            protect = UnlearnIntervalProtection(
                targets=int_targets, lambda_interval=lambda_interval,
                lower_percentile=lower_pct, upper_percentile=upper_pct,
                reduced_dim=reduced_dim, infinity_scale=inf_scale,
                use_actual_bounds=use_actual, normalize_protection=norm_prot,
            )

            def gen_batches(prompts, n=50):
                import random
                batches = []
                for _ in range(n):
                    p = random.choice(prompts)
                    ids = tokenizer(p, padding='max_length', max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt').input_ids.to(device)
                    with torch.no_grad():
                        emb = pipe_copy.text_encoder(ids)[0]
                    z = torch.randn((1,4,64,64)).to(device)
                    batches.append((z, emb))
                return batches

            forget_b = gen_batches(old_texts, 50)
            remain_b = gen_batches(retain_texts, 50) if use_actual else None

            def fwd(unet, batch, dev, **kw):
                z, c = batch
                n = z.size(0)
                t_ = torch.randint(0,1000,(n,),device=dev).long()
                unet(z, t_, encoder_hidden_states=c)

            protect.setup_protection(pipe.unet, forget_b, device,
                                     remain_dataloader=remain_b, forward_fn=fwd)
            protect.freeze_non_target_params(pipe.unet)
            trainable = protect.get_trainable_params(pipe.unet)
            optimizer = torch.optim.Adam(trainable, lr=lr)
            criteria = torch.nn.MSELoss()
            pipe.unet.train()

            for epoch in tqdm(range(epochs), desc=f'Epoch ({artist})'):
                prog_bar = tqdm(total=inner_iters * len(old_texts), desc=f'Steps ep {epoch}', leave=False)
                for step in range(inner_iters):
                    for i_idx in range(len(old_texts)):
                        bc = old_texts[i_idx]
                        bn = new_texts[i_idx]
                        id_c = tokenizer(bc, padding='max_length', max_length=tokenizer.model_max_length,
                                         truncation=True, return_tensors='pt').input_ids.to(device)
                        id_n = tokenizer(bn, padding='max_length', max_length=tokenizer.model_max_length,
                                         truncation=True, return_tensors='pt').input_ids.to(device)
                        with torch.no_grad():
                            inp_emb = pipe_copy.text_encoder(id_c)[0]
                            new_emb = pipe_copy.text_encoder(id_n)[0]
                        optimizer.zero_grad()
                        z = torch.randn((1,4,64,64), device=device)
                        noise = torch.randn_like(z)
                        t_ = torch.randint(0,1000,(1,),device=device).long()
                        z_noisy = pipe.scheduler.add_noise(z, noise, t_)
                        with torch.no_grad():
                            tgt = pipe_copy.unet(z_noisy, t_, encoder_hidden_states=new_emb).sample
                        pred = pipe.unet(z_noisy, t_, encoder_hidden_states=inp_emb).sample
                        loss = criteria(pred, tgt) + protect.compute_protection_loss(pipe.unet, device)
                        loss.backward()
                        optimizer.step()
                        prog_bar.update(1)
                prog_bar.close()
                torch.save(pipe.unet.state_dict(), f'{run_path}/epoch_{epoch}.pt')

        else:
            for epoch in tqdm(range(epochs), desc=f'Epoch ({artist})'):
                adv_embs = []
                new_embs = []
                for i_idx in range(len(old_texts)):
                    bc = old_texts[i_idx]
                    bn = new_texts[i_idx]
                    id_c = tokenizer(bc, padding='max_length', max_length=tokenizer.model_max_length,
                                     truncation=True, return_tensors='pt').input_ids.to(device)
                    id_n = tokenizer(bn, padding='max_length', max_length=tokenizer.model_max_length,
                                     truncation=True, return_tensors='pt').input_ids.to(device)
                    with torch.no_grad():
                        inp_emb = pipe_copy.text_encoder(id_c)[0]
                        new_emb = pipe_copy.text_encoder(id_n)[0]
                    new_embs.append(new_emb[0])
                    if 'regzero' in emb_computing:
                        _, adv_e = close_form_emb_regzero(pipe, pipe_copy, bc, with_to_k=True,
                                                          save_path=run_path, regeular_scale=regular_scale,
                                                          seed=seed, save_name=f'{epoch}-{i_idx}')
                    elif 'surrogate' in emb_computing:
                        _, adv_e = close_form_emb(pipe, pipe_copy, bc, with_to_k=True, save_path=run_path,
                                                  old_target_concept=None, regeular_scale=regular_scale,
                                                  seed=seed, save_name=f'{epoch}-{i_idx}', reg_item=reg_item)
                    else:
                        _, adv_e = close_form_emb(pipe, pipe_copy, bc, with_to_k=True, save_path=run_path,
                                                  old_target_concept=None, regeular_scale=regular_scale,
                                                  seed=seed, save_name=f'{epoch}-{i_idx}')
                    adv_embs.append(adv_e[0])
                pipe = edit_model_adversarial(pipe, adv_embs, new_embs, retain_texts,
                                              technique=technique, preserve_scale=preserve_scale,
                                              erase_scale=erase_scale, lamb=lamb)
                torch.save(pipe.unet.state_dict(), f'{run_path}/epoch_{epoch}.pt')

        # Only generate eval images for final checkpoint (8670 prompts is expensive per epoch)
        pipe.unet.load_state_dict(torch.load(f'{run_path}/epoch_{epochs-1}.pt'))
        pipe.to(device)
        generate_images(pipe, dev_df, f'{run_path}/final', ddim_steps=ddim_steps, num_samples=num_samples)

        total_time = time.time() - start

        # ---- LPIPS evaluation -----------------------------------------------
        erased_dir = f'{run_path}/final'

        all_files = sorted([f for f in os.listdir(os.path.join(f'{run_path}/before/imgs')) if f.endswith('.png')])
        erased_matched = filter_prompts_by_artist(test_csv, artist)
        if eval_prompts_sample is not None and len(erased_matched) > eval_prompts_sample:
            import random as _rnd
            _rnd.seed(seed)
            erased_matched = _rnd.sample(erased_matched, eval_prompts_sample)
        unerased_matched = [f for f in all_files if f not in erased_matched]

        lpips_e = compute_lpips_between_dirs(f'{run_path}/before', erased_dir, loss_fn, filter_files=erased_matched)
        lpips_u = compute_lpips_between_dirs(f'{run_path}/before', erased_dir, loss_fn, filter_files=unerased_matched)

        avg_e = np.mean(list(lpips_e.values())) if lpips_e else 0.0
        avg_u = np.mean(list(lpips_u.values())) if lpips_u else 0.0
        delta = avg_e - avg_u

        print(f'{artist}: LPIPS_e={avg_e:.6f}  LPIPS_u={avg_u:.6f}  LPIPS_d={delta:.6f}')

        # ---- wandb logging ---------------------------------------------------
        run.log({
            f'LPIPS_e/{artist}': avg_e,
            f'LPIPS_u/{artist}': avg_u,
            f'LPIPS_d/{artist}': delta,
            'hyperparams/lr': lr,
            'hyperparams/epochs': epochs,
            'hyperparams/regular_scale': regular_scale,
            'time/total_sec': total_time,
        }, step=epochs)

        e_rows = [[fn, f'{v:.6f}'] for fn,v in lpips_e.items()]
        u_rows = [[fn, f'{v:.6f}'] for fn,v in lpips_u.items()]
        run.log({
            f'lpips_e_table/{artist}': wandb.Table(data=e_rows, columns=['file','LPIPS']),
            f'lpips_u_table/{artist}': wandb.Table(data=u_rows, columns=['file','LPIPS']),
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

        erased_pairs = get_img_pairs(f'{run_path}/before', erased_dir, erased_matched)
        if erased_pairs:
            tb_e = wandb.Table(columns=['index', 'prompt', 'baseline', 'unlearned'])
            for idx, (fp_b, fp_t, fn) in enumerate(erased_pairs):
                tb_e.add_data(idx, fn, wandb.Image(fp_b), wandb.Image(fp_t))
            run.log({f'comparison_erased/{artist}': tb_e})

        unerased_pairs = get_img_pairs(f'{run_path}/before', erased_dir, unerased_matched)
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
            'lr': lr,
            'epochs': epochs,
            'regular_scale': regular_scale,
            'use_intact': use_intact,
        })

    # ---- summary table -----------------------------------------------------
    summary_table = wandb.Table(data=results_rows, columns=list(results_rows[0].keys()))
    mean_d = np.mean([r['LPIPS_d'] for r in results_rows])
    run.log({'summary_lpips': summary_table, 'mean_LPIPS_d': mean_d})
    run.summary['mean_LPIPS_d'] = mean_d
    run.finish()
    # Clean up temporary wandb directory to avoid leftover files
    import shutil
    shutil.rmtree(temp_wandb_dir, ignore_errors=True)
    print('\nPipeline complete. Results logged to wandb.')
