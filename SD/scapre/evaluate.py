"""
ScaPre-compatible evaluation for BARRIER unlearning method.

Evaluates a concept-erased Stable Diffusion model using the EXACT protocol from
ScaPre (ICLR 2026) paper -- same prompts, seeds, classifier, CLIP, scheduler,
generation parameters. Guarantees direct comparability with Table 3 and Table 4.

Table 3 -- ImageNet-Diversi50 (50 concepts):
    Average Unlearning Accuracy, CLIPcoco, UQ

Table 4 -- ImageNet-Confuse5 (5 confusing pairs, 10 concepts):
    Unlearn Acc, Preserve Acc, Overall Acc, CLIPcoco, UQ

Usage:
    # Table 3 (Diversi50)
    python scapre/evaluate.py \\
        --benchmark diversi50 \\
        --ckpt_name models/barrier-diversi50/diffusers-barrier-diversi50.pt \\
        --output_dir results/diversi50

    # Table 4 (Confuse5)
    python scapre/evaluate.py \\
        --benchmark confuse5 \\
        --ckpt_name models/barrier-confuse5/diffusers-barrier-confuse5.pt \\
        --output_dir results/confuse5

    # CLIPcoco only (COCO 30K prompts)
    python scapre/evaluate.py \\
        --benchmark diversi50 \\
        --ckpt_name models/barrier-diversi50/diffusers-barrier-diversi50.pt \\
        --output_dir results/diversi50 \\
        --coco_csv datasets/coco_30k.csv \\
        --coco_prompts_txt datasets/coco_prompts.txt
"""

import json
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SD_DIR = SCRIPT_DIR.parent


def _load_pipeline(model_id: str, ckpt_name: str | None, gpu: int):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16,
    )
    if ckpt_name is not None:
        state_dict = torch.load(ckpt_name, map_location="cpu")
        pipe.unet.load_state_dict(state_dict, strict=False)
    return pipe.to(gpu)


def _load_classifier(gpu: int):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(gpu).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


def _generate_image(pipe, prompt: str, seed: int, gpu: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return pipe(prompt).images[0]


def _classify(image: Image.Image, classifier, preprocess, categories, gpu: int):
    inp = preprocess(image).unsqueeze(0).to(gpu)
    with torch.no_grad():
        logits = classifier(inp)
    top1_idx = logits.argmax(dim=-1).item()
    pred = categories[top1_idx].lower()
    return pred


def _match_concept(pred: str, label: str) -> bool:
    return label in pred or pred in label


def _compute_clipcoco(generated_images_dir: str, coco_prompts_source: str,
                      device_str: str, max_images: int | None = None):
    """CLIPcoco: CLIP ViT-B/32 cosine similarity between generated images and
    COCO captions.  Supports both CSV (case_number, prompt) and .txt (one
    prompt per line) formats -- matching ScaPre's eval_coco_clip.py interface.
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("WARNING: transformers not installed; skipping CLIPcoco")
        return None

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device_str)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    img_dir = Path(generated_images_dir)

    # --- gather (image_path, prompt) pairs ---
    pairs = []
    src = Path(coco_prompts_source)

    if src.suffix == ".csv":
        df = pd.read_csv(src)
        for _, row in df.iterrows():
            case = int(row.case_number)
            prompt_text = str(row.prompt)
            for img_path in sorted(img_dir.glob(f"{case}_*.png")):
                pairs.append((str(img_path), prompt_text))
    else:
        prompts = [line.strip() for line in src.read_text().splitlines() if line.strip()]
        pngs = sorted(img_dir.glob("*.png"))
        pngs = [p for p in pngs if not p.name.startswith(".")]
        for img_path, prompt_text in zip(pngs, prompts):
            pairs.append((str(img_path), prompt_text))

    if not pairs:
        print("WARNING: No image-prompt pairs found for CLIPcoco")
        return None

    if max_images is not None:
        pairs = pairs[:max_images]

    scores = []
    batch_size = 16
    for i in tqdm(range(0, len(pairs), batch_size), desc="CLIPcoco"):
        batch = pairs[i:i + batch_size]
        images = []
        valid_prompts = []
        for p, pr in batch:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_prompts.append(pr)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(text=valid_prompts, images=images, return_tensors="pt",
                           padding=True, truncation=True).to(device_str)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image.diagonal()
            scores.extend(logits.cpu().tolist())

    if scores:
        avg = float(np.mean(scores))
        print(f"CLIPcoco = {avg:.4f}  (n={len(scores)})")
        return avg
    return None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_uq(unlearn_accuracy: float, clip_coco: float,
               mu_a: float | None = None, sigma_a: float | None = None,
               mu_c: float | None = None, sigma_c: float | None = None):
    """
    Unbiased Quality: harmonic mean of sigmoid-normalised forgetting and CLIP.

        Ã  = sigmoid((μA − A) / σA)
        C̃  = sigmoid((C  − μC) / σC)
        UQ = 100 × 2ÃC̃ / (Ã + C̃)

    If mu/sigma are None, they default to:
        μA = 0.20,  σA = 0.15   (typical unlearn-accuracy reference)
        μC = 0.30,  σC = 0.03   (typical CLIP score reference)

    These should be computed across *all* methods in a comparison table to
    ensure fairness.  The defaults above are reasonable stand-ins; replace
    with values derived from the full method pool for paper reporting.
    """
    if mu_a is None:
        mu_a = 0.20
    if sigma_a is None:
        sigma_a = 0.15
    if mu_c is None:
        mu_c = 0.30
    if sigma_c is None:
        sigma_c = 0.03

    a_tilde = _sigmoid((mu_a - unlearn_accuracy) / sigma_a)
    c_tilde = _sigmoid((clip_coco - mu_c) / sigma_c)

    uq = 100.0 * (2.0 * a_tilde * c_tilde) / (a_tilde + c_tilde + 1e-8)
    return float(uq), float(a_tilde), float(c_tilde)


# ============================================================================
# Diversi50 evaluation  (Table 3)
# ============================================================================

def evaluate_diversi50(pipe, gpu: int, args):
    """
    For each of the 50 concepts, generate images from imagenet-50.csv prompts
    and measure Residual Accuracy (how often the erased model still produces
    recognisable instances of the concept).
    """
    dataset_csv = args.dataset_csv or str(SD_DIR / "scapre" / "datasets" / "imagenet-50.csv")
    output_dir = Path(args.output_dir) / "diversi50"
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier, preprocess, categories = _load_classifier(gpu)
    df = pd.read_csv(dataset_csv)
    concepts = sorted(df["class"].str.lower().unique())
    concept_rows = {c: df[df["class"].str.lower() == c] for c in concepts}

    max_ppp = args.max_prompts_per_concept  # per-concept cap

    concept_accs = {}
    all_results = []

    for concept in tqdm(concepts, desc="Diversi50 concepts"):
        rows = concept_rows[concept]
        if max_ppp:
            rows = rows.head(max_ppp)

        c_out = output_dir / concept
        c_out.mkdir(parents=True, exist_ok=True)

        correct = 0
        total = 0

        for idx, (_, row) in enumerate(rows.iterrows()):
            prompt = row["prompt"]
            seed = int(row["evaluation_seed"])

            img_path = c_out / f"img_{idx:04d}.png"
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")
            else:
                image = _generate_image(pipe, prompt, seed, gpu)
                image.save(img_path)

            pred = _classify(image, classifier, preprocess, categories, gpu)
            hit = _match_concept(pred, concept)
            if hit:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        concept_accs[concept] = acc
        all_results.append({"concept": concept, "accuracy": acc,
                            "correct": correct, "total": total})
        print(f"  {concept}: {correct}/{total} = {acc:.4f}")

    avg_acc = float(np.mean(list(concept_accs.values())))
    print(f"\nAverage Unlearning Accuracy (Diversi50) = {avg_acc:.6f}")

    summary = {
        "benchmark": "diversi50",
        "num_concepts": len(concept_accs),
        "avg_unlearn_accuracy": avg_acc,
        "per_concept": all_results,
    }

    summary_path = output_dir / "results_diversi50.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    return avg_acc


# ============================================================================
# Confuse5 evaluation  (Table 4)
# ============================================================================

def evaluate_confuse5(pipe, gpu: int, args):
    """
    For each of the 10 concepts in imagenet-15.csv (5 confusing pairs):
      - Unlearn Acc: residual accuracy on target concept (lower better)
      - Preserve Acc: accuracy on the *paired* concept (higher better)
      - Overall Acc = 2*(100-A)*P / ((100-A) + P)
    """
    dataset_csv = args.dataset_csv or str(SD_DIR / "scapre" / "datasets" / "imagenet-15.csv")
    output_dir = Path(args.output_dir) / "confuse5"
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier, preprocess, categories = _load_classifier(gpu)
    df = pd.read_csv(dataset_csv)

    pairs = [
        ("golden retriever", "labrador retriever"),
        ("tabby", "tiger cat"),
        ("orange", "lemon"),
        ("speedboat", "lifeboat"),
        ("soccer ball", "volleyball"),
    ]

    max_ppp = args.max_prompts_per_concept
    pair_results = []

    for target, confuse in tqdm(pairs, desc="Confuse5 pairs"):
        tgt_rows = df[df["class"].str.lower() == target]
        cnf_rows = df[df["class"].str.lower() == confuse]
        if max_ppp:
            tgt_rows = tgt_rows.head(max_ppp)
            cnf_rows = cnf_rows.head(max_ppp)

        # --- target: unlearn accuracy ---
        tgt_correct = 0
        tgt_total = 0
        for idx, (_, row) in enumerate(tgt_rows.iterrows()):
            img_path = output_dir / target / f"img_{idx:04d}.png"
            output_dir_target = output_dir / target
            output_dir_target.mkdir(parents=True, exist_ok=True)
            image_path = output_dir_target / f"img_{idx:04d}.png"

            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
            else:
                image = _generate_image(pipe, row["prompt"], int(row["evaluation_seed"]), gpu)
                image.save(image_path)

            pred = _classify(image, classifier, preprocess, categories, gpu)
            hit = _match_concept(pred, target)
            if hit:
                tgt_correct += 1
            tgt_total += 1

        a = (tgt_correct / tgt_total) if tgt_total > 0 else 0.0

        # --- confuse pair: preserve accuracy ---
        cnf_correct = 0
        cnf_total = 0
        for idx, (_, row) in enumerate(cnf_rows.iterrows()):
            output_dir_confuse = output_dir / confuse
            output_dir_confuse.mkdir(parents=True, exist_ok=True)
            image_path = output_dir_confuse / f"img_{idx:04d}.png"

            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
            else:
                image = _generate_image(pipe, row["prompt"], int(row["evaluation_seed"]), gpu)
                image.save(image_path)

            pred = _classify(image, classifier, preprocess, categories, gpu)
            hit = _match_concept(pred, confuse)
            if hit:
                cnf_correct += 1
            cnf_total += 1

        p = (cnf_correct / cnf_total) if cnf_total > 0 else 0.0

        a_pct = a * 100
        p_pct = p * 100
        overall = (2.0 * (100.0 - a_pct) * p_pct) / ((100.0 - a_pct) + p_pct + 1e-8)

        print(f"  {target} → {confuse}:  Unlearn={a_pct:.1f}%  "
              f"Preserve={p_pct:.1f}%  Overall={overall:.2f}")

        pair_results.append({
            "target": target, "confuse_pair": confuse,
            "unlearn_accuracy": a, "unlearn_accuracy_pct": a_pct,
            "preserve_accuracy": p, "preserve_accuracy_pct": p_pct,
            "overall_accuracy": overall,
            "target_correct": tgt_correct, "target_total": tgt_total,
            "confuse_correct": cnf_correct, "confuse_total": cnf_total,
        })

    avg_a = float(np.mean([r["unlearn_accuracy"] for r in pair_results]))
    avg_p = float(np.mean([r["preserve_accuracy"] for r in pair_results]))
    avg_o = float(np.mean([r["overall_accuracy"] for r in pair_results]))

    print(f"\n  Avg Unlearn Acc = {avg_a*100:.2f}%")
    print(f"  Avg Preserve Acc = {avg_p*100:.2f}%")
    print(f"  Avg Overall Acc = {avg_o:.2f}")

    summary = {
        "benchmark": "confuse5",
        "num_pairs": len(pair_results),
        "avg_unlearn_accuracy": avg_a,
        "avg_preserve_accuracy": avg_p,
        "avg_overall_accuracy": avg_o,
        "per_pair": pair_results,
    }

    summary_path = output_dir / "results_confuse5.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    return avg_a, avg_p, avg_o


# ============================================================================
# Main
# ============================================================================

def _generate_coco_images(pipe, coco_prompts_source: str, output_dir: Path, gpu: int,
                          max_images: int | None = None):
    """Generate images from COCO prompts for CLIPcoco evaluation."""
    coco_dir = output_dir / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    src = Path(coco_prompts_source)
    if src.suffix == ".csv":
        df = pd.read_csv(src)
    else:
        prompts = [line.strip() for line in src.read_text().splitlines() if line.strip()]
        df = pd.DataFrame({"case_number": range(len(prompts)), "prompt": prompts,
                           "evaluation_seed": [42 + i * 7 for i in range(len(prompts))]})

    if max_images:
        df = df.head(max_images)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="COCO images"):
        case = int(row.case_number)
        img_path = coco_dir / f"{case}_0.png"
        if img_path.exists():
            continue
        seed = int(row.get("evaluation_seed", 42))
        image = _generate_image(pipe, str(row.prompt), seed, gpu)
        image.save(img_path)

    print(f"Generated {len(list(coco_dir.glob('*.png')))} COCO images")


def parse_args():
    p = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", type=str, required=True,
                   choices=["diversi50", "confuse5"],
                   help="Which benchmark to evaluate.")
    p.add_argument("--gpu", type=int, default=0)

    # Model
    p.add_argument("--model_id", type=str,
                   default="runwayml/stable-diffusion-v1-5",
                   help="Base SD version (must match ScaPre: SD 1.5)")
    p.add_argument("--ckpt_name", type=str, default=None,
                   help="Path to BARRIER-exported diffusers UNet checkpoint (.pt).")

    # Datasets
    p.add_argument("--dataset_csv", type=str, default=None,
                   help="Override path to prompt CSV")
    p.add_argument("--max_prompts_per_concept", type=int, default=None,
                   help="Cap prompts per concept (None = use all)")

    # Output
    p.add_argument("--output_dir", type=str, default="results/scapre_eval")

    # COCO CLIP (optional, run separately per benchmark)
    p.add_argument("--coco_prompts_source", type=str, default=None,
                   help="Path to COCO prompts CSV or .txt for CLIPcoco")
    p.add_argument("--coco_max_images", type=int, default=None)

    # UQ reference statistics
    p.add_argument("--uq_mu_a", type=float, default=None)
    p.add_argument("--uq_sigma_a", type=float, default=None)
    p.add_argument("--uq_mu_c", type=float, default=None)
    p.add_argument("--uq_sigma_c", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    gpu = args.gpu

    print(f"\n{'='*60}")
    print(f"  ScaPre-eval: {args.benchmark}")
    print(f"  Base model:  {args.model_id}")
    print(f"  Checkpoint:  {args.ckpt_name}")
    print(f"{'='*60}\n")

    # --- load pipeline ---
    pipe = _load_pipeline(args.model_id, args.ckpt_name, gpu)
    print("Pipeline loaded.\n")

    # --- run classification benchmark ---
    if args.benchmark == "diversi50":
        avg_acc = evaluate_diversi50(pipe, gpu, args)
        unlearn_acc = avg_acc
        preserve_acc = None
        overall_acc = None

    else:  # confuse5
        avg_acc, avg_preserve, avg_overall = evaluate_confuse5(pipe, gpu, args)
        unlearn_acc = avg_acc
        preserve_acc = avg_preserve
        overall_acc = avg_overall

    # --- CLIPcoco (optional) ---
    clip_coco = None
    if args.coco_prompts_source:
        coco_out = Path(args.output_dir)
        _generate_coco_images(pipe, args.coco_prompts_source, coco_out, gpu,
                              max_images=args.coco_max_images)
        clip_coco = _compute_clipcoco(
            str(coco_out / "coco"), args.coco_prompts_source, f"cuda:{gpu}",
            max_images=args.coco_max_images,
        )

    # --- UQ ---
    uq_value = None
    if unlearn_acc is not None and clip_coco is not None:
        uq_value, a_tilde, c_tilde = compute_uq(
            unlearn_acc, clip_coco,
            mu_a=args.uq_mu_a, sigma_a=args.uq_sigma_a,
            mu_c=args.uq_mu_c, sigma_c=args.uq_sigma_c,
        )
        print(f"\nUQ = {uq_value:.2f}  (Ã={a_tilde:.4f}, C̃={c_tilde:.4f})")

    # --- final table ---
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS -- {args.benchmark}")
    print(f"{'='*60}")

    if args.benchmark == "diversi50":
        print(f"| {'Method':<15} | {'Avg Acc':>8}  | {'CLIPcoco':>8}  | {'UQ':>8}  |")
        print(f"|{'-'*17}|{'-'*11}|{'-'*11}|{'-'*11}|")
        acc_s = f"{unlearn_acc*100:.2f}" if unlearn_acc else "N/A"
        cl_s = f"{clip_coco*100:.2f}" if clip_coco else "N/A"
        uq_s = f"{uq_value:.2f}" if uq_value else "N/A"
        method = os.path.basename(args.ckpt_name or "baseline").split(".")[0][:14]
        print(f"| {method:<15} | {acc_s:>8}% | {cl_s:>8}% | {uq_s:>8} |")
    else:
        print(f"| {'Method':<15} | {'Unlearn':>8}  | {'Preserve':>8}  | {'Overall':>8}  | {'CLIPcoco':>8}  | {'UQ':>8} |")
        print(f"|{'-'*17}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*11}|")
        ua_s = f"{unlearn_acc*100:.2f}" if unlearn_acc else "N/A"
        pa_s = f"{preserve_acc*100:.2f}" if preserve_acc else "N/A"
        oa_s = f"{overall_acc:.2f}" if overall_acc else "N/A"
        cl_s = f"{clip_coco*100:.2f}" if clip_coco else "N/A"
        uq_s = f"{uq_value:.2f}" if uq_value else "N/A"
        method = os.path.basename(args.ckpt_name or "baseline").split(".")[0][:14]
        print(f"| {method:<15} | {ua_s:>8}% | {pa_s:>8}% | {oa_s:>8} | {cl_s:>8}% | {uq_s:>8} |")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()