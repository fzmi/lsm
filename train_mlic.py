from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.mlic import MLICPlusPlus
from models.mlic.config.config import model_config
from models.mlic.data import MLICPatchDataset
from models.mlic.loss.rd_loss import RateDistortionLoss
from models.mlic.utils.optimizers import configure_optimizers
from utils.regions import CROSSVAL_GROUPS

EPS = 1e-9


@dataclass
class TrainConfig:
    dataset_path: Path
    train_regions: List[str]
    val_regions: List[str]
    patch_size: int
    batch_size: int
    test_batch_size: int
    epochs: int
    learning_rate: float
    aux_learning_rate: float
    lmbda: float
    num_workers: int
    clip_max_norm: float
    seed: Optional[int]
    device: torch.device
    metrics: str
    save_template: Optional[str]
    checkpoint: Optional[Path]
    wandb: bool
    checkpoint_template: Optional[str] = None
    save_path: Optional[Path] = None
    fold: Optional[int] = None
    train_sample_ratio: float = 1.0
    train_max_patches: Optional[int] = None


def _lambda_tag(value: float) -> str:
    tag = f"{value:.4g}"
    tag = tag.replace('.', 'p').replace('-', 'm')
    return tag


def _format_template(template: str, lambda_value: float, fold: Optional[int] = None, arch: Optional[str] = None) -> str:
    mapping = {
        'lambda_value': lambda_value,
        'lambda_tag': _lambda_tag(lambda_value),
    }
    if fold is not None:
        mapping['fold'] = fold
    if arch is not None:
        mapping['arch'] = arch
    return template.format(**mapping)


def _resolve_save_path(template: Optional[str], lambda_value: float, fold: Optional[int]) -> Optional[Path]:
    if template is None:
        return None
    return Path(_format_template(template, lambda_value, fold=fold, arch='mlic'))


def _resolve_checkpoint_path(cfg: TrainConfig, lambda_value: float, fold: Optional[int]) -> Optional[Path]:
    if cfg.checkpoint_template:
        return Path(_format_template(cfg.checkpoint_template, lambda_value, fold=fold, arch='mlic'))
    return cfg.checkpoint


def _select_subset_indices(total: int, ratio: float, max_patches: Optional[int], seed: Optional[int]) -> List[int]:
    if ratio >= 0.999 and (max_patches is None or max_patches >= total):
        return list(range(total))
    ratio = max(0.0, min(1.0, ratio))
    max_count = total
    if ratio < 0.999:
        max_count = min(max_count, max(1, int(math.ceil(total * ratio))))
    if max_patches is not None:
        max_count = min(max_count, max_patches)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return indices[:max_count]


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def _init_wandb(cfg: TrainConfig):
    if not cfg.wandb:
        return None
    import wandb  # lazy import

    config = {
        "train_regions": cfg.train_regions,
        "val_regions": cfg.val_regions,
        "patch_size": cfg.patch_size,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "aux_learning_rate": cfg.aux_learning_rate,
        "lambda": cfg.lmbda,
        "metrics": cfg.metrics,
    }
    if cfg.fold is not None:
        config["fold"] = cfg.fold
    return wandb.init(project="lsm-mlic", config=config)


def train_one_epoch(
    model: nn.Module,
    criterion: RateDistortionLoss,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    aux_optimizer: optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
    wandb_run,
) -> float:
    model.train()
    device = cfg.device
    psnr_meter = AverageMeter()
    prefix = f"[Fold {cfg.fold}] " if cfg.fold is not None else ""

    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Train {epoch + 1}/{cfg.epochs}",
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    for batch_idx, data in enumerate(progress, start=1):
        data = data.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        outputs = model(data)
        losses = criterion(outputs, data)

        loss_main = losses["loss"]
        if not torch.isfinite(loss_main).item():
            progress.write(f"{prefix}Non-finite main loss encountered; skipping batch")
            continue
        bpp_loss = losses.get("bpp_loss")
        mse_loss = losses.get("mse_loss")
        ms_ssim_loss = losses.get("ms_ssim_loss") if "ms_ssim_loss" in losses else None
        if bpp_loss is not None and not torch.isfinite(bpp_loss).item():
            progress.write(f"{prefix}Non-finite bpp loss encountered; skipping batch")
            continue
        if mse_loss is not None and not torch.isfinite(mse_loss).item():
            progress.write(f"{prefix}Non-finite mse loss encountered; skipping batch")
            continue
        if ms_ssim_loss is not None and not torch.isfinite(ms_ssim_loss).item():
            progress.write(f"{prefix}Non-finite ms-ssim loss encountered; skipping batch")
            continue

        loss_main.backward()
        if cfg.clip_max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        if not torch.isfinite(aux_loss).item():
            progress.write(f"{prefix}Non-finite auxiliary loss encountered; skipping batch")
            continue
        aux_loss.backward()
        aux_optimizer.step()

        with torch.no_grad():
            mse_batch = torch.mean((outputs["x_hat"].detach() - data.detach()) ** 2)
            psnr_batch = -10.0 * torch.log10(mse_batch + 1e-8)
        psnr_value = float(psnr_batch.item())
        psnr_meter.update(psnr_value, data.size(0))

        progress.set_postfix({"loss": f"{loss_main.item():.3f}", "psnr": f"{psnr_value:.2f}"})

        if wandb_run is not None:
            log_payload = {
                "train_loss": float(loss_main.detach()),
                "train_aux_loss": float(aux_loss.detach()),
                "train_psnr": psnr_value,
                "epoch": epoch,
                "step": epoch * len(dataloader) + batch_idx,
            }
            if bpp_loss is not None:
                log_payload["train_bpp"] = float(bpp_loss.detach())
            if mse_loss is not None:
                log_payload["train_mse"] = float(mse_loss.detach()) * 255.0 ** 2
            if ms_ssim_loss is not None:
                log_payload["train_ms_ssim"] = float(ms_ssim_loss.detach())
            wandb_run.log(log_payload)

    progress.close()
    return psnr_meter.avg


def evaluate(
    model: nn.Module,
    criterion: RateDistortionLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    meters = {
        "loss": AverageMeter(),
        "bpp": AverageMeter(),
        "mse": AverageMeter(),
        "aux": AverageMeter(),
    }
    if criterion.metrics == 'ms-ssim':
        meters["ms_ssim"] = AverageMeter()

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device=device, non_blocking=True)
            outputs = model(data)
            losses = criterion(outputs, data)
            aux_loss = float(model.aux_loss().detach())

            meters["loss"].update(float(losses["loss"].detach()))
            meters["bpp"].update(float(losses["bpp_loss"].detach()))
            if losses.get("mse_loss") is not None:
                meters["mse"].update(float(losses["mse_loss"].detach()))
            if losses.get("ms_ssim_loss") is not None:
                meters["ms_ssim"].update(float(losses["ms_ssim_loss"].detach()))
            meters["aux"].update(aux_loss)

    return {name: meter.avg for name, meter in meters.items()}


def save_checkpoint(path: Path, state: Dict[str, object], is_best: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.with_name(path.stem + "_best" + path.suffix)
        torch.save(state, best_path)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLIC baseline on LSM patches")
    parser.add_argument("--dataset-path", required=True, help="Path to the LSM dataset root")
    parser.add_argument("--train-regions", nargs="*", default=["brisbane", "camarillo", "cambridge"], help="Regions for training")
    parser.add_argument("--val-regions", nargs="*", default=["hawick", "kagoshima", "lamington"], help="Regions for validation")
    parser.add_argument("--patch-size", type=int, default=64, help="Square patch size (defaults to 64)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Main optimizer learning rate")
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3, help="Auxiliary optimizer learning rate")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0483, help="Rate-distortion lambda")
    parser.add_argument("--lambdas", type=float, nargs="+", help="Train separate models for each lambda value (overrides --lambda)")
    parser.add_argument("--metrics", choices=["mse", "ms-ssim"], default="mse", help="Optimization target")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--clip-max-norm", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save-path", type=str, default="outputs/mlic/lambda{lambda_tag}.pth", help="Checkpoint path or template (supports {lambda_value} / {lambda_tag} / {fold})")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path or template for resuming (supports {lambda_value} / {lambda_tag} / {fold})")
    parser.add_argument("--no-save", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--train-sample-ratio", type=float, default=1.0, help="Fraction of training patches to use (<=1)")
    parser.add_argument("--train-max-patches", type=int, help="Maximum number of training patches to use")
    parser.add_argument("--cross-validate", action="store_true", help="Train 4-fold group cross-validation models")
    return parser.parse_args(list(argv))


def build_dataloader(regions: List[str], cfg: TrainConfig, shuffle: bool, subset_training: bool = True) -> DataLoader:
    dataset = MLICPatchDataset(cfg.dataset_path, regions, (cfg.patch_size, cfg.patch_size))
    if subset_training and (cfg.train_sample_ratio < 0.999 or (cfg.train_max_patches is not None and cfg.train_max_patches > 0)):
        indices = _select_subset_indices(len(dataset), cfg.train_sample_ratio, cfg.train_max_patches, cfg.seed)
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size if shuffle else cfg.test_batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=shuffle,
    )
    return loader


def prepare_config(args: argparse.Namespace) -> TrainConfig:
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint_path = None
    checkpoint_template = None
    if args.checkpoint:
        if '{' in args.checkpoint:
            checkpoint_template = args.checkpoint
        else:
            checkpoint_path = Path(args.checkpoint)

    return TrainConfig(
        dataset_path=Path(args.dataset_path),
        train_regions=list(args.train_regions),
        val_regions=list(args.val_regions),
        patch_size=int(args.patch_size),
        batch_size=int(args.batch_size),
        test_batch_size=int(args.test_batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        aux_learning_rate=float(args.aux_learning_rate),
        lmbda=float(args.lmbda),
        num_workers=int(args.num_workers),
        clip_max_norm=float(args.clip_max_norm),
        seed=args.seed,
        device=device,
        metrics=args.metrics,
        save_template=None if args.no_save else str(args.save_path),
        checkpoint=checkpoint_path,
        wandb=bool(args.wandb),
        checkpoint_template=checkpoint_template,
        train_sample_ratio=float(args.train_sample_ratio),
        train_max_patches=args.train_max_patches,
    )


def train_for_lambda(run_cfg: TrainConfig, train_loader: DataLoader, val_loader: DataLoader, config) -> None:
    prefix = f"[Fold {run_cfg.fold}] " if run_cfg.fold is not None else ""
    print(f"{prefix}Training MLIC model with lambda={run_cfg.lmbda}")

    model = MLICPlusPlus(config=config).to(run_cfg.device)
    optimizer, aux_optimizer = configure_optimizers(model, run_cfg)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.2)
    criterion = RateDistortionLoss(lmbda=run_cfg.lmbda, metrics=run_cfg.metrics)

    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = run_cfg.checkpoint
    if checkpoint_path:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist")
        checkpoint = torch.load(checkpoint_path, map_location=run_cfg.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        scheduler.load_state_dict(checkpoint.get('lr_scheduler', scheduler.state_dict()))
        best_loss = float(checkpoint.get('best_loss', best_loss))
        start_epoch = int(checkpoint.get('epoch', 0))
        print(f"{prefix}Resumed from checkpoint '{checkpoint_path}' at epoch {start_epoch}")

    wandb_run = _init_wandb(run_cfg)

    for epoch in range(start_epoch, run_cfg.epochs):
        avg_train_psnr = train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, epoch, run_cfg, wandb_run)
        metrics = evaluate(model, criterion, val_loader, run_cfg.device)
        scheduler.step()

        print(
            f"{prefix}Epoch {epoch}: train_psnr={avg_train_psnr:.2f}, val_loss={metrics['loss']:.6f}, val_bpp={metrics['bpp']:.4f}, val_aux={metrics['aux']:.6f}"
        )

        if wandb_run is not None:
            log_data = {
                'val_loss': metrics['loss'],
                'val_bpp': metrics['bpp'],
                'val_aux': metrics['aux'],
                'train_psnr_avg': avg_train_psnr,
                'epoch': epoch,
            }
            if 'mse' in metrics:
                log_data['val_mse'] = metrics.get('mse')
            if 'ms_ssim' in metrics:
                log_data['val_ms_ssim'] = metrics.get('ms_ssim')
            if run_cfg.fold is not None:
                log_data['fold'] = run_cfg.fold
            wandb_run.log(log_data)

        current_loss = metrics['loss']
        is_best = current_loss < best_loss
        best_loss = min(best_loss, current_loss)

        if run_cfg.save_path is not None:
            run_cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'aux_optimizer': aux_optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
            }
            save_checkpoint(run_cfg.save_path, state, is_best)

    model.update(force=True)
    print(f"{prefix}Finished training MLIC lambda={run_cfg.lmbda} (best validation loss {best_loss:.6f})")

    if wandb_run is not None:
        wandb_run.finish()


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    base_cfg = prepare_config(args)
    lambda_values = [float(v) for v in (args.lambdas if args.lambdas else [args.lmbda])]
    config = model_config()

    if args.cross_validate:
        for fold_idx, group in enumerate(CROSSVAL_GROUPS, start=1):
            print(f"==== Fold {fold_idx}: training regions {group} ====")
            val_regions = list(group)
            train_loader = build_dataloader(val_regions, base_cfg, shuffle=True)
            val_loader = build_dataloader(val_regions, base_cfg, shuffle=False)
            for lambda_value in lambda_values:
                save_path = _resolve_save_path(base_cfg.save_template, lambda_value, fold_idx)
                if save_path is not None and base_cfg.save_template and '{fold}' not in base_cfg.save_template:
                    save_path = save_path.with_name(f"{save_path.stem}_fold{fold_idx}{save_path.suffix}")

                checkpoint_path = _resolve_checkpoint_path(base_cfg, lambda_value, fold_idx)
                if checkpoint_path is not None and base_cfg.checkpoint_template and '{fold}' not in base_cfg.checkpoint_template:
                    checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_fold{fold_idx}{checkpoint_path.suffix}")

                run_cfg = replace(
                    base_cfg,
                    train_regions=val_regions,
                    val_regions=val_regions,
                    lmbda=lambda_value,
                    save_path=save_path,
                    checkpoint=checkpoint_path,
                    fold=fold_idx,
                )

                train_for_lambda(run_cfg, train_loader, val_loader, config)
        return

    train_loader = build_dataloader(base_cfg.train_regions, base_cfg, shuffle=True)
    val_loader = build_dataloader(base_cfg.val_regions, base_cfg, shuffle=False)

    for lambda_value in lambda_values:
        save_path = _resolve_save_path(base_cfg.save_template, lambda_value, None)
        checkpoint_path = _resolve_checkpoint_path(base_cfg, lambda_value, None)
        run_cfg = replace(base_cfg, lmbda=lambda_value, save_path=save_path, checkpoint=checkpoint_path, fold=None)
        train_for_lambda(run_cfg, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
