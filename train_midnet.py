from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import imagecodecs
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tifffile import tifffile
from tqdm import tqdm

from utils.metrics import calc_metrics_patch


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


@dataclass
class TrainConfig:
    dataset_path: Path
    region: str
    patch_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    clip_max_norm: float
    device: torch.device
    seed: Optional[int]
    intermediate_method: str
    intermediate_scale: Optional[int]
    intermediate_quality: Optional[int]
    total_paths: int
    resblocks: int
    n_feats: int
    save_path: Optional[Path]
    checkpoint: Optional[Path]
    wandb: bool
    train_sample_ratio: float
    train_max_patches: Optional[int]
    scheduler_patience: int
    scheduler_gamma: float
    jpeg2000_root: Optional[Path]
    disable_amp: bool = False


@dataclass
class DatasetInfo:
    channel_means: torch.Tensor
    intermediate_bits: float
    pixel_count: int
    band_count: int


class MIDNetPatchDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        region: str,
        patch_size: int,
        intermediate_method: str,
        intermediate_param: Optional[int],
        total_paths: int,
        sample_ratio: float,
        max_patches: Optional[int],
        seed: Optional[int],
        jpeg2000_root: Optional[Path] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.region = region
        self.patch_size = int(patch_size)
        self.intermediate_method = intermediate_method.lower()
        self.intermediate_param = intermediate_param
        self.total_paths = max(1, int(total_paths))
        self.jpeg2000_root = Path(jpeg2000_root) if jpeg2000_root is not None else None
        self.bit_depth = 16  # LSM data is uint16

        image = self._load_region(region)
        self._band_count = image.shape[0]

        patches_gt = self._extract_patches(image)
        patches_it, bits = self._make_intermediate(image)

        # Flatten band dimension to create per-band samples
        num_patches = patches_gt.shape[0]
        bands = self._band_count
        samples_gt = patches_gt.reshape(num_patches * bands, 1, self.patch_size, self.patch_size)
        samples_it = patches_it.reshape(num_patches * bands, 1, self.patch_size, self.patch_size)
        band_indices_full = torch.arange(bands, dtype=torch.long).repeat(num_patches)

        total_samples = samples_gt.shape[0]
        indices = list(range(total_samples))
        if sample_ratio < 0.999 or (max_patches is not None and max_patches < total_samples):
            target = total_samples
            if sample_ratio < 0.999:
                target = min(target, max(1, int(round(total_samples * max(0.0, min(1.0, sample_ratio))))))
            if max_patches is not None:
                target = min(target, max_patches)
            rng = random.Random(seed)
            rng.shuffle(indices)
            indices = sorted(indices[:target])

        index_tensor = torch.tensor(indices, dtype=torch.long)
        self.patches_gt = samples_gt[index_tensor].contiguous()
        self.patches_it = samples_it[index_tensor].contiguous()
        self.band_indices = band_indices_full[index_tensor].contiguous()
        self.path_indices = self._assign_paths(self.band_indices).contiguous()

        channel_means = torch.mean(image, dim=(1, 2), keepdim=True)
        self.channel_means = channel_means.float()
        self.pixel_count = image.shape[1] * image.shape[2]
        self.intermediate_bits = float(bits)

    @property
    def band_count(self) -> int:
        return self._band_count

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            channel_means=self.channel_means.clone(),
            intermediate_bits=self.intermediate_bits,
            pixel_count=self.pixel_count,
            band_count=self.band_count,
        )

    def _load_region(self, region: str) -> torch.Tensor:
        region_dir = self.dataset_path / region
        for candidate in ("image.tiff", "image.tif"):
            path = region_dir / candidate
            if path.is_file():
                arr = tifffile.imread(path)
                tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))
                if tensor.ndim != 3:
                    raise ValueError(f"Expected 3D tensor for region '{region}'")
                return tensor.permute(2, 0, 1).contiguous()
        raise FileNotFoundError(f"No image.tif(f) found for region '{region}'")

    def _extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        bands, height, width = image.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size")
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, bands, self.patch_size, self.patch_size)
        return patches.contiguous()

    def _make_intermediate(self, image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        bands, height, width = image.shape
        if self.intermediate_method == "jpeg2000":
            quality = self.intermediate_param
            if quality is None:
                raise ValueError("JPEG2000 intermediate requires --intermediate-quality")
            recon, bits = self._load_jpeg2000_reconstruction(image.shape, quality)
            return self._extract_patches(recon), bits
        raise ValueError(f"Unsupported intermediate method '{self.intermediate_method}'")

    def _load_jpeg2000_reconstruction(self, shape: Tuple[int, int, int], quality: int) -> Tuple[torch.Tensor, float]:
        bands, height, width = shape
        recon = torch.zeros(bands, height, width, dtype=torch.float32)
        total_bits = 0.0
        for band in range(bands):
            path = self._locate_jpeg2000_file(self.region, band, quality)
            data = path.read_bytes()
            arr = imagecodecs.jpeg2k_decode(data)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.shape != (height, width):
                raise ValueError(f"JPEG2000 reconstruction shape mismatch for region '{self.region}', band {band}: expected {(height, width)}, got {arr.shape}")
            arr = arr.astype(np.float32)
            if arr.shape != (height, width):
                raise ValueError(f"JPEG2000 reconstruction shape mismatch for region '{self.region}', band {band}: expected {(height, width)}, got {arr.shape}")
            recon[band] = torch.from_numpy(arr)
            total_bits += float(path.stat().st_size) * 8.0
        return recon, total_bits

    def _locate_jpeg2000_file(self, region: str, band: int, quality: int) -> Path:
        basename = f"{region}_j2k_b{band}_q{quality}.jp2"
        band_only = f"j2k_b{band}_q{quality}.jp2"
        search_roots: List[Path] = []
        if self.jpeg2000_root is not None:
            search_roots.append(self.jpeg2000_root)
        search_roots.extend(
            [
                self.dataset_path / region,
                self.dataset_path,
                self.dataset_path.parent / "jpeg2000",
                Path.cwd() / "outputs" / "jpeg2000",
            ]
        )
        candidates = []
        for root in search_roots:
            candidates.extend(
                [
                    root / basename,
                    root / band_only,
                    root / region / basename,
                    root / region / band_only,
                ]
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not find JPEG2000 reconstruction for region '{region}', band {band}, quality {quality}. " f"Checked: {[str(p) for p in candidates]}")

    def _assign_paths(self, band_indices: torch.Tensor) -> torch.Tensor:
        if self.total_paths <= 1:
            return torch.zeros_like(band_indices)
        return band_indices % self.total_paths

    def __len__(self) -> int:
        return self.patches_gt.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input": self.patches_it[index],
            "target": self.patches_gt[index],
            "band": self.band_indices[index],
            "path": self.path_indices[index],
        }


class ContentAwareFM(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int, length: int = 1) -> None:
        super().__init__()
        self.transformers = nn.ModuleList([nn.Conv2d(in_channel, in_channel, kernel_size, padding=kernel_size // 2, groups=in_channel) for _ in range(length)])
        self.gammas = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(length)])
        self.act = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transformer, gamma in zip(self.transformers, self.gammas):
            res = transformer(x) * gamma
            res = self.act(res)
            x = res + x
        return x


class ResBlockShared(nn.Module):
    def __init__(self, conv: nn.Module, n_feats: int, kernel_size: int, total_paths: int, bias: bool = True) -> None:
        super().__init__()
        self.shared_conv1 = conv(n_feats, n_feats, kernel_size, bias=bias, padding=kernel_size // 2)
        self.shared_conv2 = conv(n_feats, n_feats, kernel_size, bias=bias, padding=kernel_size // 2)
        self.act = nn.ReLU(True)

        if total_paths == 1:
            self.split_fms1 = nn.ModuleList([ContentAwareFM(n_feats, 1, length=3)])
            self.split_fms2 = nn.ModuleList([ContentAwareFM(n_feats, 1, length=3)])
        elif total_paths == 3:
            self.split_fms1 = nn.ModuleList([ContentAwareFM(n_feats, 1, length=L) for L in (1, 3, 5)])
            self.split_fms2 = nn.ModuleList([ContentAwareFM(n_feats, 1, length=L) for L in (1, 3, 5)])
        else:
            raise ValueError("ResBlockShared supports total_paths equal to 1 or 3")

    def forward(self, x: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        if path.ndim == 0:
            path = path.unsqueeze(0).expand(x.size(0))
        if path.numel() == 1:
            branch = int(path[0].item())
            res = self.shared_conv1(x)
            res = self.split_fms1[branch](res)
            res = self.act(res)
            res = self.shared_conv2(res)
            res = self.split_fms2[branch](res)
            return (res + x).to(dtype=x.dtype)
        outputs = torch.empty_like(x)
        for branch in torch.unique(path):
            mask = path == branch
            chunk = x[mask]
            res = self.shared_conv1(chunk)
            res = self.split_fms1[branch](res)
            res = self.act(res)
            res = self.shared_conv2(res)
            res = self.split_fms2[branch](res)
            outputs[mask] = (res + chunk).to(dtype=x.dtype)
        return outputs


class BCResBlocksShared(nn.Module):
    def __init__(
        self,
        channel_means: torch.Tensor,
        total_paths: int = 3,
        n_resblocks: int = 16,
        channels: int = 1,
        n_feats: int = 64,
    ) -> None:
        super().__init__()
        if total_paths not in (1, 3):
            raise ValueError("BCResBlocksShared supports total_paths equal to 1 or 3")
        self.channel_means = channel_means
        self.n_feats = int(n_feats)
        self.head = nn.Conv2d(channels, self.n_feats, 3, padding=1)
        self.body = nn.ModuleList([
            ResBlockShared(nn.Conv2d, self.n_feats, 3, total_paths)
            for _ in range(n_resblocks)
        ])
        self.body_final = nn.Conv2d(self.n_feats, self.n_feats, 3, padding=1)
        self.tail = nn.Conv2d(self.n_feats, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, band: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        band_means = self.channel_means[band].view(-1, 1, 1, 1)
        centered = x - band_means
        res = self.head(centered)
        for block in self.body:
            res = block(res, path)
        res = self.body_final(res)
        res = self.tail(res)
        res = res + centered
        return res + band_means


def _model_weight_bits(model: nn.Module, delivery_dtype: Optional[str] = None) -> float:
    """Count delivered model bits.

        torch.save(model.state_dict(), tmp); bits = os.path.getsize(tmp) * 8

    i.e. the bits that would actually be shipped on the wire if the per-region
    weights were delivered as a `torch.save`'d state_dict (params + buffers,
    plus pickle metadata: shapes, dtypes, key strings). The earlier
    implementation summed ``numel * element_size * 8`` and under-reported by
    the ``torch.save`` serialisation overhead — typically 130 KB on a 5 MB
    checkpoint (~0.018 bpp on a 9216x6400 region), which matters for iso-bpp
    comparison against the v1 archive curves.

    ``delivery_dtype`` casts floating-point tensors to the requested precision
    BEFORE serialisation. Non-floating-point tensors (e.g., integer buffers)
    are left as-is.
    """
    import tempfile, os
    dtype_map = {
        None: None,
        "fp32": torch.float32, "float32": torch.float32, "single": torch.float32,
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    }
    key = delivery_dtype.lower() if isinstance(delivery_dtype, str) else delivery_dtype
    # Preserve the OrderedDict type returned by state_dict() — torch.save serialises
    # OrderedDict and dict with different pickle headers, leading to ~130 KB drift.
    from collections import OrderedDict
    original_sd = model.state_dict()
    if key in dtype_map:
        target = dtype_map[key]
        if target is None:
            sd = original_sd
        else:
            sd = OrderedDict((k, v.to(target) if v.is_floating_point() else v) for k, v in original_sd.items())
    elif key in ("int8", "uint8"):
        # int8 delivery — approximate by halving fp16 bytes (no clean torch.save
        # path for arbitrary quantisation). Adds the same pickle overhead.
        sd = OrderedDict((k, v.to(torch.float16).view(torch.uint8)[: v.numel()] if v.is_floating_point() else v)
                         for k, v in original_sd.items())
    else:
        raise ValueError(f"Unsupported delivery_dtype: {delivery_dtype!r}")

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as fh:
        tmp_path = fh.name
    try:
        torch.save(sd, tmp_path)
        size_bits = os.path.getsize(tmp_path) * 8
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return float(size_bits)


def _init_wandb(cfg: TrainConfig):
    if not cfg.wandb:
        return None
    import wandb

    config = {
        "region": cfg.region,
        "patch_size": cfg.patch_size,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "intermediate_method": cfg.intermediate_method,
        "intermediate_scale": cfg.intermediate_scale,
        "intermediate_quality": cfg.intermediate_quality,
        "total_paths": cfg.total_paths,
        "resblocks": cfg.resblocks,
    }
    return wandb.init(project="lsm-midnet", config=config)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    cfg: TrainConfig,
    wandb_run,
) -> Dict[str, float]:
    model.train()
    device = cfg.device
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    progress = tqdm(dataloader, total=len(dataloader), desc=f"Train {epoch + 1}/{cfg.epochs}", leave=False, disable=not sys.stderr.isatty())
    for step, batch in enumerate(progress, start=1):
        inputs = batch["input"].to(device=device, non_blocking=True)
        targets = batch["target"].to(device=device, non_blocking=True)
        band = batch["band"].to(device=device, non_blocking=True)
        path = batch["path"].to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda" and not getattr(cfg, "disable_amp", False)):
            outputs = model(inputs, band, path)
            loss = criterion(outputs, targets)
        if not torch.isfinite(loss).item():
            progress.write("Skipping batch due to non-finite loss")
            continue
        scaler.scale(loss).backward()
        if cfg.clip_max_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        metrics = calc_metrics_patch(outputs.detach(), targets.detach(), ssim=False)
        loss_meter.update(float(loss.detach()), inputs.size(0))
        psnr_meter.update(metrics.get("psnr_patch", 0.0), inputs.size(0))
        progress.set_postfix({"loss": f"{loss_meter.avg:.3f}", "psnr": f"{psnr_meter.avg:.2f}"})

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train_loss": float(loss.detach()),
                    "train_psnr": metrics.get("psnr_patch", 0.0),
                    "epoch": epoch,
                    "step": (epoch * len(dataloader)) + step,
                }
            )

    progress.close()
    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg}


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    ssim_measure = StructuralSimilarityIndexMeasure(data_range=65535.0).to(device)

    # Accumulated wall-clock time of the model forward pass only. This is the
    # network-refinement portion of the decode path; the JPEG2000 intermediate
    # decode is timed separately by the caller.
    forward_time = 0.0
    is_cuda = torch.device(device).type == "cuda"

    # Per-patch metric aggregation — matches the JP2/JPEG baselines in
    # benchmark.py (single_result_on_patch=True): compute MSE per patch, then
    # PSNR per patch, then average across all patches. The previous version
    # called calc_metrics_patch on the whole batch tensor, which gave per-batch
    # PSNR (≈ psnr_img) and was Jensen-pessimistic vs the per-patch aggregation
    # the JP2 baseline reports.
    max_value = 65535.0
    log20_max = 20.0 * math.log10(max_value)
    eps = 1e-8
    sum_per_patch_mse = 0.0
    sum_per_patch_psnr = 0.0
    n_patches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device=device, non_blocking=True)
            targets = batch["target"].to(device=device, non_blocking=True)
            band = batch["band"].to(device=device, non_blocking=True)
            path = batch["path"].to(device=device, non_blocking=True)

            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.time()
            outputs = model(inputs, band, path)
            if is_cuda:
                torch.cuda.synchronize(device)
            forward_time += time.time() - t0

            loss = criterion(outputs, targets)

            se = (outputs - targets) ** 2
            per_patch_mse = se.mean(dim=(1, 2, 3))
            per_patch_psnr = log20_max - 10.0 * torch.log10(per_patch_mse + eps)
            sum_per_patch_mse += float(per_patch_mse.sum().item())
            sum_per_patch_psnr += float(per_patch_psnr.sum().item())
            n_patches += int(inputs.size(0))

            loss_meter.update(float(loss.detach()), inputs.size(0))
            ssim_value = float(ssim_measure(outputs, targets).item())
            ssim_meter.update(ssim_value, inputs.size(0))

    mse_patch_avg = sum_per_patch_mse / max(1, n_patches)
    psnr_patch_avg = sum_per_patch_psnr / max(1, n_patches)

    return {
        "loss": loss_meter.avg,
        "mse": mse_patch_avg,
        "psnr": psnr_patch_avg,
        "ssim": ssim_meter.avg,
        "forward_time": forward_time,
    }


def save_checkpoint(path: Path, state: Dict[str, object], is_best: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.with_name(path.stem + "_best" + path.suffix)
        torch.save(state, best_path)


def build_dataloader(cfg: TrainConfig, shuffle: bool) -> Tuple[DataLoader, DatasetInfo]:
    if cfg.intermediate_method == "jpeg2000":
        intermediate_param = cfg.intermediate_quality
    else:
        raise ValueError("This benchmark release supports MIDNet with JPEG2000 intermediates only")

    dataset = MIDNetPatchDataset(
        cfg.dataset_path,
        cfg.region,
        cfg.patch_size,
        cfg.intermediate_method,
        intermediate_param,
        cfg.total_paths,
        cfg.train_sample_ratio,
        cfg.train_max_patches,
        cfg.seed,
        jpeg2000_root=cfg.jpeg2000_root,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
    )
    return loader, dataset.info()


def prepare_config(args: argparse.Namespace) -> TrainConfig:
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    save_path = Path(args.save_path).expanduser() if args.save_path else None
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else None
    jpeg_root = Path(args.jpeg2000_root).expanduser() if args.jpeg2000_root else None

    return TrainConfig(
        dataset_path=Path(args.dataset_path),
        region=str(args.region),
        patch_size=int(args.patch_size),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        clip_max_norm=float(args.clip_max_norm),
        device=device,
        seed=args.seed,
        intermediate_method=str(args.intermediate_method),
        intermediate_scale=args.intermediate_scale,
        intermediate_quality=args.intermediate_quality,
        total_paths=int(args.total_paths),
        resblocks=int(args.resblocks),
        n_feats=int(args.n_feats),
        save_path=save_path,
        checkpoint=checkpoint,
        wandb=bool(args.wandb),
        train_sample_ratio=float(args.train_sample_ratio),
        train_max_patches=args.train_max_patches,
        scheduler_patience=int(args.scheduler_patience),
        scheduler_gamma=float(args.scheduler_gamma),
        jpeg2000_root=jpeg_root,
        disable_amp=bool(args.no_amp),
    )


def train_for_region(cfg: TrainConfig) -> None:
    print(f"Training MIDNet on region '{cfg.region}' with method={cfg.intermediate_method}")

    train_loader, data_info = build_dataloader(cfg, shuffle=True)
    model = BCResBlocksShared(
        data_info.channel_means.to(device=cfg.device),
        total_paths=cfg.total_paths,
        n_resblocks=cfg.resblocks,
        channels=1,
        n_feats=cfg.n_feats,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.scheduler_patience,
        factor=cfg.scheduler_gamma,
    )
    criterion = nn.HuberLoss(reduction="mean", delta=0.1).to(cfg.device)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.device.type == "cuda" and not getattr(cfg, "disable_amp", False))

    model_bits_constant = _model_weight_bits(model)
    checkpoint_path = cfg.checkpoint
    start_epoch = 0
    best_loss = float("inf")

    if checkpoint_path:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed from checkpoint '{checkpoint_path}' at epoch {start_epoch}")

    wandb_run = _init_wandb(cfg)

    pixel_count = max(1, data_info.pixel_count)
    band_count = max(1, data_info.band_count)
    bpp_intermediate = data_info.intermediate_bits / pixel_count
    bpp_model = model_bits_constant / pixel_count
    bpp_total = bpp_intermediate + bpp_model
    bpppb_total = bpp_total / band_count

    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scaler, epoch, cfg, wandb_run)
        scheduler.step(train_metrics["loss"])

        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, train_psnr={train_metrics['psnr']:.2f}, " f"bpp_total={bpp_total:.4f}, bpp_intermediate={bpp_intermediate:.4f}, bpp_model={bpp_model:.4f}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train_loss_epoch": train_metrics["loss"],
                    "train_psnr_epoch": train_metrics["psnr"],
                    "bpp_total": bpp_total,
                    "bpp_intermediate": bpp_intermediate,
                    "bpp_model": bpp_model,
                    "bpppb_total": bpppb_total,
                    "epoch": epoch,
                }
            )

        current_loss = train_metrics["loss"]
        is_best = current_loss < best_loss
        best_loss = min(best_loss, current_loss)

        if cfg.save_path is not None:
            cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": best_loss,
                "bpp_total": bpp_total,
                "bpp_intermediate": bpp_intermediate,
                "bpp_model": bpp_model,
            }
            save_checkpoint(cfg.save_path, state, is_best)

    print(f"Finished training region '{cfg.region}' (best loss {best_loss:.6f}) with total bpp={bpp_total:.4f} " f"(intermediate={bpp_intermediate:.4f}, model={bpp_model:.4f})")

    if wandb_run is not None:
        wandb_run.finish()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit MIDNet on a single region")
    parser.add_argument("--dataset-path", required=True, help="Path to the LSM dataset root")
    parser.add_argument("--region", required=True, help="Region to overfit")
    parser.add_argument("--patch-size", type=int, default=64, help="Square patch size")
    parser.add_argument("--batch-size", type=int, default=72, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--clip-max-norm", type=float, default=0.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--intermediate-method",
        choices=["jpeg2000"],
        default="jpeg2000",
        help="Intermediate reconstruction method",
    )
    parser.add_argument("--intermediate-scale", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--intermediate-quality", type=int, help="JPEG2000 quality level for intermediate patches")
    parser.add_argument("--total-paths", type=int, default=1, choices=[1, 3], help="Number of path branches in MIDNet")
    parser.add_argument("--resblocks", type=int, default=16, help="Number of residual blocks")
    parser.add_argument("--n-feats", type=int, default=64, help="Feature channels inside the residual blocks (default 64; try 32 for a smaller model with lower bpp_model overhead)")
    parser.add_argument("--save-path", type=str, help="Optional path to store checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for resuming")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--train-sample-ratio", type=float, default=1.0, help="Fraction of patches to use (<=1)")
    parser.add_argument("--train-max-patches", type=int, help="Maximum number of patches to use")
    parser.add_argument("--scheduler-patience", type=int, default=15, help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=0.8, help="Gamma for ReduceLROnPlateau scheduler")
    parser.add_argument("--jpeg2000-root", type=str, help="Optional directory containing JPEG2000 reconstructions")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision (fp16) — useful when a region produces non-finite losses under AMP")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg = prepare_config(args)
    train_for_region(cfg)


if __name__ == "__main__":
    main()
