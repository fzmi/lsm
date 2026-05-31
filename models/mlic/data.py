from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tifffile import tifffile

PatchIndex = Tuple[int, int, int]


@dataclass(frozen=True)
class RegionInfo:
    name: str
    path: Path
    shape: Tuple[int, int, int]
    patch_counts: Tuple[int, int]
    bands: int


class MLICPatchDataset(Dataset):

    def __init__(
        self,
        dataset_root: Path | str,
        regions: Sequence[str],
        patch_size: Tuple[int, int] = (64, 64),
        max_value: float = 65535.0,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.patch_size = patch_size
        self.max_value = float(max_value)

        self._regions: List[RegionInfo] = []
        self._arrays: List[np.memmap] = []
        self._indices: List[Tuple[int, int, int]] = []  # (region_idx, band, flat_patch_idx)

        ph, pw = patch_size
        for region_name in regions:
            image_path = self._resolve_image_path(region_name)
            arr = tifffile.memmap(image_path)
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D array for {region_name}, got shape {arr.shape}")
            h, w, bands = arr.shape
            h_count = h // ph
            w_count = w // pw
            if h_count == 0 or w_count == 0:
                raise ValueError(f"Image {region_name} ({h}x{w}) is smaller than patch size {patch_size}")

            region_idx = len(self._regions)
            self._regions.append(
                RegionInfo(
                    name=region_name,
                    path=image_path,
                    shape=(h, w, bands),
                    patch_counts=(h_count, w_count),
                    bands=bands,
                )
            )
            self._arrays.append(arr)

            for band in range(bands):
                for flat_idx in range(h_count * w_count):
                    self._indices.append((region_idx, band, flat_idx))

    def _resolve_image_path(self, region_name: str) -> Path:
        base = self.dataset_root / region_name
        for candidate in ("image.tiff", "image.tif"):
            path = base / candidate
            if path.is_file():
                return path
        raise FileNotFoundError(f"Could not locate image.tif(f) for region '{region_name}'")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        region_idx, band, flat_idx = self._indices[index]
        region = self._regions[region_idx]
        arr = self._arrays[region_idx]

        ph, pw = self.patch_size
        h_count, w_count = region.patch_counts
        h_idx = flat_idx // w_count
        w_idx = flat_idx % w_count
        y0 = h_idx * ph
        y1 = y0 + ph
        x0 = w_idx * pw
        x1 = x0 + pw

        patch = np.asarray(arr[y0:y1, x0:x1, band], dtype=np.float32) / self.max_value
        tensor = torch.from_numpy(patch).unsqueeze(0)  # 1 x H x W
        return tensor

    def iter_region_band_patches(self, region_name: str) -> Iterator[torch.Tensor]:
        region_map = {info.name: idx for idx, info in enumerate(self._regions)}
        if region_name not in region_map:
            raise KeyError(f"Region '{region_name}' not part of this dataset")
        region_idx = region_map[region_name]
        region = self._regions[region_idx]
        arr = self._arrays[region_idx]
        ph, pw = self.patch_size
        h_count, w_count = region.patch_counts

        for band in range(region.bands):
            for h_idx in range(h_count):
                for w_idx in range(w_count):
                    y0 = h_idx * ph
                    y1 = y0 + ph
                    x0 = w_idx * pw
                    x1 = x0 + pw
                    patch = np.asarray(arr[y0:y1, x0:x1, band], dtype=np.float32) / self.max_value
                    tensor = torch.from_numpy(patch).unsqueeze(0)
                    yield tensor

    @property
    def region_infos(self) -> Sequence[RegionInfo]:
        return self._regions
