import torch
import random
import pandas as pd
import argparse
import os
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import CLIPTokenizer
from functools import reduce
import operator
import time
import tqdm
import json
import numpy as np
from PIL import Image
from utils import list_Image_to_tensor, generate_latents
from execs import compute_nudity_rate

def embedding2img(embeddings, df, ldm_stable, save_path, save_name=None, device='cuda:0', guidance_scale=7.5, image_size=512, ddim_steps=50, num_samples=1, from_case=0, to_case=None):
    vae = ldm_stable.vae
    tokenizer = ldm_stable.tokenizer
    text_encoder = ldm_stable.text_encoder
    unet = ldm_stable.unet
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    folder_path = save_path
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f'{folder_path}/emb2imgs', exist_ok=True)
    repeated_rows = []
    for i in range(len(embeddings)):
        row = df.iloc[i]
        seed = row.evaluation_seed if hasattr(row, 'evaluation_seed') else row.sd_seed
        case_number = row.case_number if hasattr(row, 'case_number') else i
        repeated_rows.extend([row] * num_samples)
        if case_number < from_case:
            continue
        if to_case is not None and case_number >= to_case:
            break
        height = row.sd_image_height if hasattr(row, 'sd_image_height') else image_size
        width = row.sd_image_width if hasattr(row, 'sd_image_width') else image_size
        num_inference_steps = ddim_steps
        guidance_scale = row.evaluation_guidance if hasattr(row, 'evaluation_guidance') else guidance_scale
        generator = torch.cuda.manual_seed(seed)
        batch_size = num_samples
        text_embeddings = [embeddings[i]] * num_samples
        text_embeddings = torch.stack(text_embeddings)
        max_length = embeddings.shape[1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), generator=generator, device=torch_device)
        latents = latents.to(torch_device)
        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.init_noise_sigma
        from tqdm.auto import tqdm
        scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(im) for im in images]
        for num, im in enumerate(pil_images):
            if save_name is None:
                im.save(f"{folder_path}/emb2imgs/{case_number}_{num}.png")
            else:
                im.save(f"{folder_path}/emb2imgs/{save_name}_{num}.png")
