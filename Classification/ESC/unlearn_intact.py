import os
import sys
import time
import random

import torch
from torch import nn

import argparse
import numpy as np
import tqdm

from pathlib import Path

# InTAct (BARRIER) lives at the BARRIER repo root: <repo>/InTAct/intact.py
def _find_repo_root():
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / 'InTAct').is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]

_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import get_dataset, get_unlearn_loader, create_dir
from models import AllCNN, load_vit
from evaluation import all_eval, evaluate_KR
from mia import evaluate_mia



def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def exp_summary(args):
    print('*' * 100)
    print(' ' * 20 + 'Experiment Summary')
    print('*' * 100)
    print(f"Experiment Name: {args.exp}")
    print(f"Method: {args.method}")
    if args.method == 'ESC':
        print(f"Pruning Hyperparameter (p): {args.p}%")
    elif args.method == 'ESC_T':
        print(f"Threshold for ESC-T: {args.threshold}")
    elif args.method == 'intact':
        print(f"InTAct lambda_interval: {args.intact_lambda}")
        print(f"InTAct targets: {args.intact_targets}")
        print(f"InTAct base method: {args.intact_base_method}")
        print(f"InTAct forget weight: {args.intact_forget_weight}")
    print(f"Data Name: {args.data_name}")
    print(f"Forget Class: {args.forget_class}")
    print(f"Model Name: {args.model_name}")
    print('*' * 100)


def prepare_dataset(args):
    '''
    Prepare dataset and dataloaders for unlearning
    Notation:
    - trfl: forgetting training dataloader
    - trrl: remaining training dataloader
    - tefl: forgetting testing dataloader
    - terl: remaining testing dataloader
    - ttfl: forgetting training dataloader for testing
    - ttrl: remaining training dataloader for testing
    '''
    # Dataset
    if args.model_name == 'vit_base_patch16_224':
        input_size = 224
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        size_scale_ratio = [input_size, scale, ratio]
    else:
        size_scale_ratio = None
    trainset, testset, test_trainset = get_dataset(args.data_name, args.dataset_dir, size_scale_ratio)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    # Forget & Remain Set (number of samples to a single class)
    if args.data_name in ['cifar100', 'tiny_imagenet']:
        num_forget = 500
    else:
        num_forget = 5000

    # Unlearn Dataloader
    trfl, _, tefl, terl, ttfl, ttrl \
        = get_unlearn_loader(trainset, testset, test_trainset, args.forget_class, args.batch_size, num_forget)

    num_classes = max(trainset.targets) + 1

    return trfl, tefl, terl, ttfl, ttrl, test_loader, train_loader, num_classes


def parse_args():
    parser = argparse.ArgumentParser("ESC + InTAct(BARRIER)")
    parser.add_argument('--exp', type=str, default='ESC_cifar10', help='experiment name')
    parser.add_argument('--method', type=str, default='ESC',
                        choices=['ESC', 'ESC_T', 'intact'],
                        help='ESC unlearning method; "intact" runs the BARRIER/InTAct method '
                             'on the same ESC model/dataset/pipeline')

    ####### Data setting #######
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                        help='dataset, among [cifar10, cifar100, tiny_imagenet]')
    parser.add_argument('--dataset_dir', type=str, default='/local_datasets', help='dataset directory')
    parser.add_argument('--forget_class', nargs='+', type=int, default=[4], help='List of the forgetting classes, for reproduce using *4 index')

    ####### Model setting #######
    parser.add_argument('--model_name', type=str, default='AllCNN', choices=['AllCNN', 'resnet_18', 'vit_base_patch16_224'], help='select the model name')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='checkpoints directory')

    ####### Experimental setting #######
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch (ESC-T)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    ########Evaluation setting#######
    parser.add_argument('--evaluation', action='store_true', help='evaluate utility of unlearn model')
    parser.add_argument('--mia', action='store_true', help='evaluate mia of unlearn model')
    parser.add_argument('--use_pytorch_mia', action='store_true', help='Use PyTorch-based MIA instead of Logistic Regression')
    parser.add_argument('--mia_batch_size', type=int, default=32, help='batch size for MIA')
    parser.add_argument('--mia_lr', type=float, default=1e-4, help='learning rate for MIA')
    parser.add_argument('--kr', action='store_true', help='evaluate Knowledge Retention (KR) of unlearn model')
    parser.add_argument('--kr_lp', type=float, default=1e-3, help='learning rate for Knowledge Retention')
    parser.add_argument('--kr_epoch', type=int, default=10, help='epoch for Knowledge Retention')
    parser.add_argument('--kr_batch_size', type=int, default=64, help='batch size for Knowledge Retention')

    ####### ESC(-T) setting #######
    parser.add_argument('--p', type=float, default=1.5, help='pruning hyperparameter for ESC')
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold for ESC-T')

    ####### InTAct (BARRIER) setting #######
    parser.add_argument('--intact_targets', nargs='+', type=str, default=None,
                        help='layer name patterns to protect (InTAct). Default: "head.0" '
                             'for AllCNN, "head" for ViT (resolves to nn.Linear modules).')
    parser.add_argument('--intact_lambda', type=float, default=100.0,
                        help='InTAct protection weight (lambda_interval)')
    parser.add_argument('--intact_reduced_dim', type=int, default=32,
                        help='InTAct SVD reduced dimension')
    parser.add_argument('--intact_lower_percentile', type=float, default=0.05,
                        help='InTAct lower percentile for safe intervals')
    parser.add_argument('--intact_upper_percentile', type=float, default=0.95,
                        help='InTAct upper percentile for safe intervals')
    parser.add_argument('--intact_infinity_scale', type=float, default=20.0,
                        help='InTAct infinity scale for safe intervals')
    parser.add_argument('--intact_base_method', type=str, default='ga', choices=['ga', 'rl'],
                        help='InTAct base unlearning loss: gradient ascent (ga) or random labels (rl)')
    parser.add_argument('--intact_use_actual_bounds', action='store_true',
                        help='InTAct: use actual (min/max) bounds instead of scale-based infinity bounds')
    parser.add_argument('--unlearn_epochs', type=int, default=10,
                        help='InTAct unlearning epochs (one epoch per forget-set pass)')
    parser.add_argument('--intact_forget_weight', type=float, default=1.0,
                        help='InTAct base-loss weight (aggressiveness of forgetting; protect term stays 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (InTAct)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='SGD weight decay (InTAct)')
    parser.add_argument('--unfreeze_backbone', action='store_true',
                        help='InTAct: unfreeze the backbone before protection (default: keep ESC frozen-backbone setup)')

    ####### wandb setting #######
    parser.add_argument('--wandb', action='store_true', help='log run to wandb (sweep: merges wandb.config into args)')
    parser.add_argument('--wandb_project', type=str, default='esc-intact-tinyimagenet', help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default='oneandzero24', help='wandb entity')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name (default: auto)')

    args = parser.parse_args()

    return args


def _load_checkpoint_state(args, path, device):
    """Load a checkpoint; accepts a raw state_dict or a full nn.Module (as released by ESC)."""
    ckpt = torch.load('{}.pth'.format(path), map_location=device, weights_only=False)
    if isinstance(ckpt, nn.Module) or hasattr(ckpt, 'state_dict'):
        ckpt = ckpt.state_dict()
    return ckpt


def main(args):
    # Set seed
    seed_torch(seed=args.seed)

    # wandb: merge sweep hyperparameters (from wandb.config) into args
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=args.wandb_name)
        for k, v in wandb.config.items():
            if hasattr(args, k):
                setattr(args, k, v)
        args.exp = f"sweep_{getattr(wandb.run, 'id', 'wandb')}"
        print(f"[wandb] run {getattr(wandb.run, 'id', '?')} | params: "
              f"method={args.method} p={getattr(args, 'p', None)} "
              f"intact_lambda={getattr(args, 'intact_lambda', None)} "
              f"intact_base_method={getattr(args, 'intact_base_method', None)} "
              f"unlearn_epochs={getattr(args, 'unlearn_epochs', None)} lr={args.lr}")

    # Summary for experiment
    exp_summary(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create directories
    exp_dir = f"experiments/{args.exp}"
    ckpt_dir = f"experiments/{args.exp}/checkpoints/"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        for arg in vars(args).items():
            f.write(f"{arg}\n")

    create_dir(args.dataset_dir)
    create_dir(args.checkpoint_dir)
    path = args.checkpoint_dir + '/'

    # Dataset
    trfl, tefl, terl, ttfl, ttrl, test_loader, train_loader, num_classes = prepare_dataset(args)

    if args.model_name == 'AllCNN':
        model = AllCNN(n_channels=3, num_classes=num_classes, filters_percentage=0.5)
        state = _load_checkpoint_state(args, path + args.data_name + "_ori_allcnn", device)

    elif args.model_name == 'vit_base_patch16_224':
        model = load_vit(args.model_name, num_classes=num_classes, device=device, is_pretrained=False, is_backbone_freezed=True)
        state = _load_checkpoint_state(args, path + args.data_name + "_ori_vit", device)

    else:
        raise NotImplementedError(f"Model {args.model_name} is not implemented.")

    model.load_state_dict(state)
    model.to(device)
    del state

    if args.method == "intact" and args.unfreeze_backbone:
        for p in model.parameters():
            p.requires_grad = True

    # Start unlearning
    if args.method == "ESC":
        print('*' * 100)
        print(' ' * 20 + 'begin ESC unlearning')
        print('*' * 100)

        start = time.time()

        # save embedding features
        data_len = len(trfl.dataset)
        if args.model_name == 'AllCNN':
            feat_log = torch.zeros(data_len, int(192 * 0.5))
        elif args.model_name == 'resnet_18':
            feat_log = torch.zeros(data_len, 512)
        elif args.model_name == 'vit_base_patch16_224':
            feat_log = torch.zeros(data_len, 768)
        else:
            raise NotImplementedError(f"Model {args.model_name} is not implemented.")

        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm.tqdm(trfl)):
                x = x.to(device, non_blocking=True)
                start_ind = i * args.batch_size
                end_ind = min((i + 1) * args.batch_size, data_len)
                output = model(x, all=True)

                if args.batch_size == output['pre_logits'].shape[0]:
                    feat_log[start_ind:end_ind, :] = output['pre_logits']
                else:
                    end_ind = i * args.batch_size + output['pre_logits'].shape[0]
                    feat_log[start_ind:end_ind, :] = output['pre_logits']

        # singular value decomposition
        u, _, _ = torch.svd(feat_log.T.to(device))

        # only use bottom p% singular vectors
        if args.model_name == 'AllCNN':
            prune_k = int(192 * 0.5 * args.p / 100)
        elif args.model_name == 'resnet_18':
            prune_k = int(512 * args.p / 100)
        elif args.model_name == 'vit_base_patch16_224':
            prune_k = int(768 * args.p / 100)
        else:
            raise NotImplementedError(f"Model {args.model_name} is not implemented.")
        u_p = u[:, prune_k:]

        model.esc_set(u_p)

        end = time.time()
        print('ESC unlearning time:', end-start, 's')

        # save model (skipped under wandb sweeps to avoid per-trial dumps)
        if not args.wandb:
            torch.save(model, '{}.pth'.format(ckpt_dir + "ESC_unlearned_model"))

    elif args.method == "ESC_T":
        print('*' * 100)
        print(' ' * 20 + 'begin ESC_T unlearning')
        print('*' * 100)

        start = time.time()

        # save embedding features
        data_len = len(trfl.dataset)
        if args.model_name == 'AllCNN':
            feat_log = torch.zeros(data_len, int(192 * 0.5))
        elif args.model_name == 'resnet_18':
            feat_log = torch.zeros(data_len, 512)
        elif args.model_name == 'vit_base_patch16_224':
            feat_log = torch.zeros(data_len, 768)
        else:
            raise NotImplementedError(f"Model {args.model_name} is not implemented.")

        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm.tqdm(trfl)):
                x = x.to(device, non_blocking=True)
                start_ind = i * args.batch_size
                end_ind = min((i + 1) * args.batch_size, data_len)
                output = model(x, all=True)

                if args.batch_size == output['pre_logits'].shape[0]:
                    feat_log[start_ind:end_ind, :] = output['pre_logits']
                else:
                    end_ind = i * args.batch_size + output['pre_logits'].shape[0]
                    feat_log[start_ind:end_ind, :] = output['pre_logits']

        # singular value decomposition
        u, _, _ = torch.svd(feat_log.T.to(device))

        mask = torch.ones_like(u)

        criterion = nn.CrossEntropyLoss()

        for epo in tqdm.tqdm(range(args.epoch)):
            for x, y in trfl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                mask = mask.detach()
                mask.requires_grad_(True)

                model.esc_set(u * mask, esc_t=True)
                outputs = model(x)

                pred = outputs.argmax(dim=1)
                learned = (y == pred)

                if learned.any():
                    loss = -criterion(outputs[learned], y[learned])
                    loss.backward()

                    if mask.grad is not None:
                        with torch.no_grad():
                            mask = mask - args.lr * mask.grad
                            mask = torch.clamp(mask, min=0, max=1)
                    mask.grad = None

            model.esc_set(u * mask, esc_t=True)

            model.eval()
            with torch.no_grad():
                num_hits = 0
                for i, (x, y) in (enumerate(trfl)):
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                    outputs = model(x)
                    pred = outputs.argmax(dim=1)
                    num_hits += (y == pred).sum().item()
            if num_hits == 0:
                break

        mask = (mask > args.threshold).to(mask.dtype)

        model.esc_set(u * mask, esc_t=True)

        end = time.time()
        print('ESC-T unlearning time:', end-start, 's')

        # save model (skipped under wandb sweeps)
        if not args.wandb:
            torch.save(model, '{}.pth'.format(ckpt_dir + "ESC_T_unlearned_model"))

    elif args.method == "intact":
        from InTAct.intact import UnlearnIntervalProtection, classification_forward_fn

        print('*' * 100)
        print(' ' * 20 + 'begin InTAct (BARRIER) unlearning')
        print('*' * 100)

        # Resolve protectable Linear modules: AllCNN packs the classifier in
        # head = Sequential(Linear), the ViT head is a plain nn.Linear.
        targets = args.intact_targets
        if targets is None:
            targets = ['head.0'] if args.model_name == 'AllCNN' else ['head']
        print(f'InTAct targets: {targets}')

        protection = UnlearnIntervalProtection(
            targets=targets,
            lambda_interval=args.intact_lambda,
            lower_percentile=args.intact_lower_percentile,
            upper_percentile=args.intact_upper_percentile,
            reduced_dim=args.intact_reduced_dim,
            infinity_scale=args.intact_infinity_scale,
            use_actual_bounds=args.intact_use_actual_bounds,
            normalize_protection=True,
        )

        # InTAct setup: collect forget activations on target layers, compute SVD,
        # snapshot target params.  remain_dataloader is only used with
        # --intact_use_actual_bounds.
        t0 = time.time()
        protection.setup_protection(
            model, trfl, device,
            remain_dataloader=ttrl,
            forward_fn=classification_forward_fn,
        )
        protection.freeze_non_target_params(model)
        trainable_params = protection.get_trainable_params(model)
        t1 = time.time()
        print(f'INTACT_SETUP_SECONDS {t1 - t0:.4f}')

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            trainable_params, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        model.train()
        for epo in range(args.unlearn_epochs):
            t_start = time.time()
            for x, y in trfl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(x)

                if args.intact_base_method == 'rl':
                    rand_t = torch.randint(0, num_classes, y.shape, device=device)
                    base_loss = criterion(outputs, rand_t)
                else:
                    base_loss = -criterion(outputs, y)

                protect_loss = protection.compute_protection_loss(model, device)
                total_loss = args.intact_forget_weight * base_loss + protect_loss
                total_loss.backward()
                optimizer.step()

            t_end = time.time()
            print(f'INTACT_EPOCH_SECONDS {t_end - t_start:.4f} '
                  f'base_loss={base_loss.item():.4f} protect_loss={protect_loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                num_hits = 0
                for i, (x, y) in enumerate(trfl):
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    outputs = model(x)
                    pred = outputs.argmax(dim=1)
                    num_hits += (y == pred).sum().item()
            print(f'INTACT_EPOCH {epo} forget train hits: {num_hits}')
            model.train()

        model.eval()
        # save model (skipped under wandb sweeps)
        if not args.wandb:
            torch.save(model, '{}.pth'.format(ckpt_dir + "InTAct_unlearned_model"))

    if args.evaluation:
        with torch.no_grad():
            all_eval(model, test_loader, ttfl, ttrl, tefl, terl, device)

    if args.wandb:
        import wandb
        from evaluation import eval as eval_acc
        with torch.no_grad():
            accs = {
                'train_forget_acc': eval_acc(model, trfl, device=device),
                'train_remain_acc': eval_acc(model, ttrl, device=device),
                'forget_acc': eval_acc(model, tefl, device=device),
                'remain_acc': eval_acc(model, terl, device=device),
                'test_acc': eval_acc(model, test_loader, device=device),
            }
        # unlearning score: keep utility, lose the forget class
        score = 100.0 * (accs['remain_acc'] - accs['forget_acc'])
        wandb.log({**accs, 'score': score})
        print(f"[wandb] score={score:.2f}")
        wandb.finish()

    if args.mia:
        evaluate_mia(model, trfl, tefl, device, args)

    if args.kr:
        evaluate_KR(model, train_loader, test_loader, ttfl, ttrl, tefl, terl, num_classes, ckpt_dir=ckpt_dir, device=device, args=args)

    return model


if __name__ == '__main__':
    args = parse_args()
    main(args)