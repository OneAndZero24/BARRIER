import sys
from pathlib import Path
# Add project root and rece directory to sys.path for module imports
root = Path(__file__).resolve().parents[2]
rece_path = root / 'SD' / 'rece'
sys.path.insert(0, str(root))
sys.path.insert(0, str(rece_path))

import logging
import torch
import random
import pandas as pd
import argparse
import os
from functools import reduce
import operator

log = logging.getLogger(__name__)
import time
import tqdm
import json
import numpy as np
import pickle
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer
else:
    StableDiffusionPipeline = LMSDiscreteScheduler = AutoencoderKL = UNet2DConditionModel = None  # type: ignore
    CLIPTokenizer = None


# Lazy imports for optional heavy dependencies – they are only required at runtime on the HPC where the packages are installed.
# Heavy imports (diffusers, transformers) are imported inside load_sd_pipeline at runtime.
# They are not required for static analysis and will be available on the HPC.



from erase_methods import edit_model_adversarial  # type: ignore
from execs import generate_images, compute_nudity_rate  # type: ignore
from utils.embedding_calculation import close_form_emb, close_form_emb_regzero  # type: ignore
from InTAct.intact import UnlearnIntervalProtection

def resolve_intact_targets(intact_cfg: dict) -> list[str]:
    """Resolve InTAct target layers from config.

    If ``target_blocks`` and ``target_layers`` are both present, expand them to
    fully-qualified SD UNet target names (e.g. ``output_blocks.4.1.transformer_blocks.0.attn2.to_q``).
    Otherwise fall back to explicit ``targets`` (default: [to_q, to_k, to_v]).
    """
    import re
    pattern = re.compile(r"^output_blocks\.(\d+)\.1\.transformer_blocks\.0\.(.+)$")

    target_blocks = intact_cfg.get("target_blocks")
    target_layers = intact_cfg.get("target_layers")

    if target_blocks is not None and target_layers is not None:
        return [
            f"output_blocks.{block}.1.transformer_blocks.0.{layer}"
            for block in target_blocks
            for layer in target_layers
        ]

    targets = intact_cfg.get("targets", ["to_q", "to_k", "to_v"])
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    return targets


# Helper to load Stable Diffusion pipeline from either a diffusers directory or a .ckpt file + config
def load_sd_pipeline(ckpt_path: str, config_path: Optional[str], device: str):
    """Load a StableDiffusionPipeline.
    If ``ckpt_path`` ends with ``.ckpt`` we convert the checkpoint using the project's
    conversion utilities (convertModels). Otherwise we assume it is a diffusers-format
    directory or a HuggingFace hub identifier.
    ``device`` is the torch device string (e.g., ``'cuda'`` or ``'cpu'``).
    """
    # Import heavy dependencies lazily at runtime.
    from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTokenizer

    if ckpt_path.endswith('.ckpt'):
        # Convert legacy checkpoint to diffusers components
        from convertModels import (
            create_vae_diffusers_config,
            create_unet_diffusers_config,
            convert_ldm_vae_checkpoint,
            convert_ldm_unet_checkpoint,
            convert_ldm_clip_checkpoint,
        )
        from omegaconf import OmegaConf
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        # Load original config
        if config_path is None:
            raise ValueError("sd_config path is required when loading from a .ckpt file")
        original_config = OmegaConf.load(config_path)
        # VAE
        vae_cfg = create_vae_diffusers_config(original_config, image_size=512)
        vae = AutoencoderKL(**vae_cfg)
        vae.load_state_dict(convert_ldm_vae_checkpoint(checkpoint, vae_cfg))
        # UNet
        unet_cfg = create_unet_diffusers_config(original_config, image_size=512)
        unet_cfg["upcast_attention"] = False
        unet = UNet2DConditionModel(**unet_cfg)
        unet.load_state_dict(convert_ldm_unet_checkpoint(checkpoint, unet_cfg))
        # Text encoder & tokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = convert_ldm_clip_checkpoint(checkpoint)
        # Scheduler (matches the default used elsewhere)
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        # Assemble pipeline
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        pipeline.to(device)
        return pipeline
    else:
        # Assume diffusers format or hub identifier
        pipeline = StableDiffusionPipeline.from_pretrained(ckpt_path)
        pipeline.to(device)
        return pipeline


def setup_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUSD',
                    description = 'Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--config', help='path to config YAML', type=str, default=None)
    parser.add_argument('--concepts', help='concepts to erase', type=str, required=False, default=None)
    parser.add_argument('--old_target_concept', help='old target concept ever used in UCE', type=str, required=False, default=None)
    parser.add_argument('--seed', help='random seed', type=int, required=False, default=42)
    parser.add_argument('--epochs', help='epochs to train', type=int, required=False, default=1)
    parser.add_argument('--test_csv_path', help='path to csv file with prompts', type=str, default='rece/dataset/validation_niche_artists.csv')
    parser.add_argument('--guided_concepts', help='whether to use old prompts to guide', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='whether to preserve old prompts', type=str, default=None)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False, default='replace')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--sd_ckpt', help='path to stable diffusion checkpoint or diffusers directory', type=str, required=False, default=None)
    parser.add_argument('--sd_config', help='path to stable diffusion config (required if sd_ckpt is .ckpt)', type=str, required=False, default=None)
    parser.add_argument('--target_ckpt', help='target checkpoint to load, UCE', type=str, required=False, default='')
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=0.1)
    parser.add_argument('--preserve_number', help='number of preserve concepts', type=int, required=False, default=None)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--lamb', help='scale for init', type=float, required=False, default=0.1)
    parser.add_argument('--save_path', help='path to save the model', type=str, required=False, default='ckpt2/SD_adv_train')
    parser.add_argument('--concept_type', help='type of concept being erased', type=str, required=False, default=None)
    parser.add_argument('--emb_computing', help='close-form or gradient-descent, standard regularization or surrogate regularization', type=str, required=False, default='close_standardreg', choices=['close_standardreg', 'close_surrogatereg', 'close_regzero'])
    parser.add_argument('--reg_item', help='use 1st, 2nd or both items in surrogate regularization', type=str, required=False, default='1st', choices=['1st', '2nd','both'])
    parser.add_argument('--regular_scale', help='scale for regularization', type=float, required=False, default=1e-3)
    parser.add_argument('--num_samples', help='number of samples for gradient descent', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='number of steps for ddim', type=int, required=False, default=50)

    # InTAct arguments
    parser.add_argument('--intact', help='whether to use intact unlearning', action='store_true')
    parser.add_argument('--lambda_interval', help='InTAct protection loss weight', type=float, default=1.0)
    parser.add_argument('--lr', help='learning rate for intact fine-tuning', type=float, default=1e-5)
    parser.add_argument('--targets', help='explicit target layer patterns (overrides target_blocks/target_layers)', type=str, nargs="+", default=None)
    parser.add_argument('--target_blocks', help='SD UNet output block indices to protect', type=int, nargs="+", default=None)
    parser.add_argument('--target_layers', help='layer names within each block (e.g. attn2.to_q)', type=str, nargs="+", default=None)
    parser.add_argument('--lower_percentile', help='lower percentile for bounds', type=float, default=0.05)
    parser.add_argument('--upper_percentile', help='upper percentile for bounds', type=float, default=0.95)
    parser.add_argument('--reduced_dim', help='reduced dimension for PCA', type=int, default=32)
    parser.add_argument('--infinity_scale', help='infinity scale for bounds', type=float, default=20.0)
    parser.add_argument('--use_actual_bounds', help='use actual bounds from remain data', action='store_true')
    parser.add_argument('--normalize_protection', help='normalize protection loss by layer count', action='store_true', default=True)

    args = parser.parse_args()

    # Load config file if specified
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check which CLI args were explicitly passed
        explicit_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                key = arg.lstrip('-').replace('-', '_')
                if '=' in key:
                    key = key.split('=')[0]
                explicit_args.add(key)
                
        # Merge config parameters into args namespace
        for section in ['unlearn', 'intact', 'paths', 'pipeline']:
            if section in config and config[section] is not None:
                for k, v in config[section].items():
                    if k not in explicit_args:
                        setattr(args, k, v)

    # Verification checks
    if args.concepts is None:
        raise ValueError("--concepts is required (either via CLI or config file)")
    if args.concept_type is None:
        raise ValueError("--concept_type is required (either via CLI or config file)")
    seed_shuffle=123
    setup_seed(seed_shuffle)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # `args.concepts` may come from CLI (string) or from a config file (list).
    # Normalise it to a list of stripped strings.
    if isinstance(args.concepts, str):
        concepts = [c.strip() for c in args.concepts.split(',')]
    else:
        concepts = [str(c).strip() for c in args.concepts]


    if args.old_target_concept is None:
        old_target_concept = [None for _ in concepts]
    else:
        old_target_concept = args.old_target_concept.split(',')
        old_target_concept = [con.strip() for con in old_target_concept]
        for idx, con in enumerate(old_target_concept):
            if con == 'none':
                old_target_concept[idx] = None
            if con == '':
                old_target_concept[idx] = ' '
    assert len(old_target_concept) == len(concepts), f'length of old_target_concept {len(old_target_concept)} should be the same as concepts {len(concepts)}'

    seed = args.seed
    epochs = args.epochs
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    technique = args.technique
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    lamb = args.lamb
    preserve_number = args.preserve_number
    concept_type = args.concept_type
    emb_computing = args.emb_computing
    reg_item = args.reg_item
    regular_scale = args.regular_scale
    num_samples = args.num_samples
    ddim_steps = args.ddim_steps
    # Load base Stable Diffusion model using configuration paths.
    # ``args.sd_ckpt`` points to the checkpoint (either .ckpt or a diffusers directory).
    # ``args.sd_config`` is required when loading from a legacy .ckpt file.
    # ``load_sd_pipeline`` handles both cases and returns a ready‑to‑use pipeline.
    ldm_stable = load_sd_pipeline(args.sd_ckpt, getattr(args, 'sd_config', None), device)
    ldm_stable_copy = load_sd_pipeline(args.sd_ckpt, getattr(args, 'sd_config', None), device)
    # ``StableDiffusionPipeline`` provides a ``tokenizer`` attribute.
    tokenizer = ldm_stable.tokenizer

    target_ckpt = args.target_ckpt
    dev_df = pd.read_csv(args.test_csv_path)

    print_text=''
    for concept in concepts:
        print_text+=f'{concept}_'

    # PROMPT CLEANING
    if concepts[0] == 'allartist':
        concepts = ["Kelly Mckernan", "Thomas Kinkade", "Pablo Picasso", "Tyler Edlin", "Kilian Eng"]
    if concepts[0] == '10artists':
        concepts = ["Asger Jorn", "Eric Fischl", "Johannes Vermeer", "Apollinary Vasnetsov", "Naoki Urasawa", "Nicolas Mignard", "John Whitcomb", "John Constable", "Warwick Globe", "Albert Marquet"]

    if 'artists' in concepts[0]:
        df = pd.read_csv('rece/dataset/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = int(concepts[0].replace('artists', ''))
        concepts = random.sample(artists,number) 

    # create a new df similar to prompts_df, using concepts and seed
    # It should contain prompt, evaluation_seed
    adv_df = pd.DataFrame(columns=['prompt', 'evaluation_seed'])
    for concept in concepts:
        adv_df = pd.concat([adv_df, pd.DataFrame([{'prompt': concept, 'evaluation_seed': args.seed}])], ignore_index=True)


    old_texts = []
    for concept in concepts:
        old_texts.append(f'{concept}')
    
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text+=f'-towards_uncond'
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
            print_text+=f'-towards_{guided_concepts[0]}'
        else:
            new_texts = [[con] for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts)
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'
            
    assert len(new_texts) == len(old_texts)
    
    
    if preserve_concepts is None:
        if concept_type == 'art':
            prompts_df = pd.read_csv('rece/dataset/artists1734_prompts.csv')

            retain_texts = list(prompts_df.artist.unique())
            old_texts_lower = [text.lower() for text in old_texts]
            preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
            if preserve_number is not None:
                print_text+=f'-preserving_{len(old_texts)}artists'
                preserve_concepts = random.sample(preserve_concepts, len(old_texts))
        else:
            preserve_concepts = []
    if type(preserve_concepts) == str:
        preserve_concepts = [con.strip() for con in preserve_concepts.split(',')]
    retain_texts = ['']+preserve_concepts
    if len(retain_texts) > 1:
        print_text+=f'-preserve_true'     
    else:
        print_text+=f'-preserve_false'
    if preserve_scale is None:
        # set the format to be .3f
        preserve_scale = max(0.1, 1/len(retain_texts))
        preserve_scale = round(preserve_scale, 3)

    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text += f"-method_{technique}" 
    print_text += f"-erase_{erase_scale}"
    print_text += f"-preserve_{preserve_scale}"
    print_text += f"-lamb_{lamb}"
    print_text = print_text.lower()
    print(print_text)
    
    
    # Initialize save_path to avoid unbound warnings
    save_path = ''
    # Determine output directory based on args
    if args.intact:
        save_path = f'{args.save_path}/{concept_type}/intact/{print_text}/lambda_{args.lambda_interval}_lr_{args.lr}/seed_{seed}'
    else:
        if 'close' in emb_computing:
            if 'surrogate' in emb_computing:
                save_path = f'{args.save_path}/{concept_type}/{emb_computing}_regitem_{reg_item}/{print_text}/regular_{regular_scale}/seed_{seed}'
            else:
                save_path = f'{args.save_path}/{concept_type}/{emb_computing}/{print_text}/regular_{regular_scale}/seed_{seed}'
    os.makedirs(save_path, exist_ok=True)

    generate_images(ldm_stable, dev_df, f'{save_path}/before', ddim_steps=ddim_steps, num_samples=num_samples)
    # load UCE model
    if target_ckpt != '':
        ldm_stable.unet.load_state_dict(torch.load(target_ckpt))
        ldm_stable.to(device)
        generate_images(ldm_stable, dev_df, f'{save_path}/uce', ddim_steps=ddim_steps, num_samples=num_samples)

    start = time.time()

    if args.intact:
        print("Setting up InTAct protection for SD UNet...")
        
        # 1. Generate synthetic batches for forget data (artists to erase)
        forget_prompts = old_texts
        # 2. Generate synthetic batches for remain data (other artists to preserve)
        remain_prompts = retain_texts
        
        def generate_synthetic_batches(prompts, n_samples=50):
            """Generate (noisy_latent, text_emb) tuples using the diffusion scheduler.
            This mirrors the distribution seen at training time: random clean latent + 
            noise added at a random timestep via the same scheduler used in training."""
            batches = []
            for _ in range(n_samples):
                prompt = random.choice(prompts)
                id_prompt = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    emb = ldm_stable_copy.text_encoder(id_prompt)[0]
                clean_z = torch.randn((1, 4, 64, 64), device=device)
                noise = torch.randn_like(clean_z)
                t = torch.randint(0, 1000, (1,), device=device).long()
                z_noisy = ldm_stable.scheduler.add_noise(clean_z, noise, t)
                batches.append((z_noisy, emb))
            return batches
        
        forget_batches = generate_synthetic_batches(forget_prompts, n_samples=50)
        remain_batches = generate_synthetic_batches(remain_prompts, n_samples=50) if args.use_actual_bounds else None

        # Resolve targets from block/layer config or explicit targets
        intact_cfg = {
            'targets': args.targets,
            'target_blocks': getattr(args, 'target_blocks', None),
            'target_layers': getattr(args, 'target_layers', None),
        }
        resolved_targets = resolve_intact_targets(intact_cfg)
        log.info(f"Resolved InTAct targets: {resolved_targets[:4]}... (total {len(resolved_targets)})")

        # Create protection instance
        protection = UnlearnIntervalProtection(
            targets=resolved_targets,
            lambda_interval=args.lambda_interval,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            reduced_dim=args.reduced_dim,
            infinity_scale=args.infinity_scale,
            use_actual_bounds=args.use_actual_bounds,
            normalize_protection=args.normalize_protection,
        )

        def sd_unet_forward_fn(unet, batch, dev, **kwargs):
            z, c = batch
            z = z.to(dev)
            c = c.to(dev)
            n = z.size(0)
            t = torch.randint(0, 1000, (n,), device=dev).long()
            unet(z, t, encoder_hidden_states=c)

        protection.setup_protection(
            ldm_stable.unet,
            forget_batches,
            device,
            remain_dataloader=remain_batches,
            forward_fn=sd_unet_forward_fn,
        )

        # Freeze non-target parameters
        protection.freeze_non_target_params(ldm_stable.unet)
        trainable_params = protection.get_trainable_params(ldm_stable.unet)
        
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        criteria = torch.nn.MSELoss()
        
        ldm_stable.unet.train()

        for epoch in tqdm.tqdm(range(epochs), desc='Epoch'):
            for i in range(0, len(old_texts)):
                batch_concept = old_texts[i]
                batch_new_text = new_texts[i]
                
                # tokenize
                id_concept = tokenizer(batch_concept, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                id_new_text = tokenizer(batch_new_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                
                # get embeddings from frozen/original model
                with torch.no_grad():
                    input_embedding = ldm_stable_copy.text_encoder(id_concept)[0]
                    new_embedding = ldm_stable_copy.text_encoder(id_new_text)[0]
                
                optimizer.zero_grad()
                
                # Sample noise and add to random latent
                z = torch.randn((1, 4, 64, 64), device=device)
                noise = torch.randn_like(z)
                t = torch.randint(0, 1000, (1,), device=device).long()
                
                z_noisy = ldm_stable.scheduler.add_noise(z, noise, t)
                
                with torch.no_grad():
                    target_noise = ldm_stable_copy.unet(z_noisy, t, encoder_hidden_states=new_embedding).sample
                    
                pred_noise = ldm_stable.unet(z_noisy, t, encoder_hidden_states=input_embedding).sample
                
                base_loss = criteria(pred_noise, target_noise)
                intact_loss = protection.compute_protection_loss(ldm_stable.unet, device)
                
                total_loss = base_loss + intact_loss
                total_loss.backward()
                optimizer.step()
                
            torch.save(ldm_stable.unet.state_dict(), f'{save_path}/epoch_{epoch}.pt')
            if epoch == epochs - 1:
                generate_images(ldm_stable, dev_df, f'{save_path}/final', ddim_steps=ddim_steps, num_samples=num_samples)
            
    else:
        for epoch in tqdm.tqdm(range(epochs), desc='Epoch'):
            adv_emb_list = []
            new_emb_list = []
            for i in range(0, len(old_texts)):
                # batch size is 1
                batch_df = adv_df.iloc[i:i+1]
                batch_concept = old_texts[i]
                batch_old_target_concept = old_target_concept[i]
                batch_new_text = new_texts[i]
                
                # tokenize
                id_concept = tokenizer(batch_concept, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                id_new_text = tokenizer(batch_new_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                
                # get embeddings
                input_embedding = ldm_stable.text_encoder(id_concept)[0]
                new_embedding = ldm_stable.text_encoder(id_new_text)[0]
                new_emb_list.append(new_embedding[0])   # squeeze the batch dimension
                
                input_ids = id_concept
                if 'close' in emb_computing:
                    if 'surrogate' in emb_computing:
                        _, adv_embedding = close_form_emb(ldm_stable, ldm_stable_copy, batch_concept, with_to_k=True, save_path=save_path, old_target_concept=None, regeular_scale=regular_scale, seed=seed, save_name=f'{epoch}-{i}', reg_item=reg_item)
                    elif 'standard' in emb_computing:
                        _, adv_embedding = close_form_emb(ldm_stable, ldm_stable_copy, batch_concept, with_to_k=True, save_path=save_path, old_target_concept=batch_old_target_concept, regeular_scale=regular_scale, seed=seed, save_name=f'{epoch}-{i}')
                    elif 'regzero' in emb_computing:
                        _, adv_embedding = close_form_emb_regzero(ldm_stable, ldm_stable_copy, batch_concept, with_to_k=True, save_path=save_path, regeular_scale=regular_scale, seed=seed, save_name=f'{epoch}-{i}')
                else:
                    raise NotImplementedError
                adv_emb_list.append(adv_embedding[0])   # squeeze the batch dimension
            ldm_stable = edit_model_adversarial(ldm_stable, adv_emb_list, new_emb_list, retain_texts, technique=technique, preserve_scale=preserve_scale, erase_scale=erase_scale, lamb=lamb)
            torch.save(ldm_stable.unet.state_dict(), f'{save_path}/epoch_{epoch}.pt')
            if epoch == epochs - 1:
                generate_images(ldm_stable, dev_df, f'{save_path}/final', ddim_steps=ddim_steps, num_samples=num_samples)

    end = time.time()
    print(f'Running time: {end-start}')
    print(f'Running time per epoch: {(end-start)/epochs}')