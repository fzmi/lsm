import os
import time
import math
from pathlib import Path

from typing import Optional
from tempfile import mkstemp

import numpy as np
import torch
import torch.nn as nn
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm
from tifffile import tifffile
from utils.metrics import filesize, calc_metrics_patch, _get_ssim_measure, bjontegaard_delta
from models.mlic import MLICPlusPlus
from models.mlic.config.config import model_config
from train_midnet import (
    BCResBlocksShared,
    MIDNetPatchDataset,
    _model_weight_bits,
    evaluate as midnet_evaluate,
)


# Module-level caches: map checkpoint-path string → loaded model.
# Avoids reloading the same weights for every region in a fold.
_stf_model_cache: dict = {}
_mlic_model_cache: dict = {}


class Encoders:
    @staticmethod
    def jpeg2000_encode(img16: np.ndarray, quality: int, path: str):
        t0 = time.time()
        ok = cv2.imwrite(path, img16, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(quality)])
        # ffmpeg.input(source_filepath).output(codec_filepath, vcodec="libopenjpeg", pix_fmt="gray16le", compression_level=quality_level).run(overwrite_output=True, cmd="ffmpeg", quiet=True)
        encode_time = time.time() - t0
        if not ok:
            raise RuntimeError("JPEG2000 write failed")

        return encode_time


class Decoders:
    @staticmethod
    def jpeg2000_decode(path: str):
        t1 = time.time()
        recon = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # uint16
        decode_time = time.time() - t1
        if recon is None:
            raise RuntimeError("JPEG2000 read failed")

        return recon, decode_time


class LSMBenchmark:
    def __init__(self, dataset_path, device="cuda"):
        self.dataset_path = dataset_path
        self.device = device
        self.regions = self._parse_dataset()
        self.region_names = sorted(self.regions.keys())
        print(f"Parsed regions: {self.region_names}")

    def _parse_dataset(self):
        datasets = {}
        for root, _, files in os.walk(self.dataset_path):
            # Only process direct subdirectories of dataset_path
            if root == self.dataset_path:
                continue

            # Get the folder name relative to dataset_path
            folder_name = os.path.relpath(root, self.dataset_path)
            # Skip nested subdirectories (only process first level)
            if os.path.sep in folder_name:
                continue

            # Check for image files in order of preference
            if "image.tiff" in files:
                datasets[folder_name] = os.path.join(root, "image.tiff")
            elif "image.tif" in files:
                datasets[folder_name] = os.path.join(root, "image.tif")

        return datasets

    def run_benchmark(
        self,
        region_names: list,
        methods: list,
        max_value=65535,
        patch_size_h=64,
        patch_size_w=64,
        stf_checkpoints=None,
        stf_lambdas=None,
        stf_checkpoint_template=None,
        stf_arch="stf",
        mlic_checkpoints=None,
        mlic_lambdas=None,
        mlic_checkpoint_template=None,
        midnet_configs=None,
        jpeg2000_qualities=None,
    ):
        # Check region names are in the parsed dataset
        if not region_names:
            raise ValueError("No regions specified for benchmarking. Please provide at least one region name.")
        for region in region_names:
            if region not in self.regions:
                raise ValueError(f"Region '{region}' not found in the dataset.")

        valid_methods = ["jpeg2000", "stf", "mlic", "midnet"]
        if not methods:
            raise ValueError("No methods specified for benchmarking. Please provide at least one method name.")
        for method in methods:
            if method not in valid_methods:
                raise ValueError(f"Method '{method}' is not a valid function. Valid methods are: {valid_methods}")

        # Read existing CSV entries up-front so each method can skip already-done work.
        os.makedirs("outputs", exist_ok=True)
        csv_file = "outputs/results.csv"
        _existing_entries: set = set()
        if os.path.exists(csv_file):
            with open(csv_file, "r") as _f:
                for _line in _f.readlines()[1:]:
                    _parts = _line.strip().split(",")
                    if len(_parts) >= 3:
                        _existing_entries.add((_parts[0], _parts[1], _parts[2]))

        all_results = {}

        for region in region_names:
            dataset_path = self.regions[region]
            print(f"Processing region: {region} with dataset path: {dataset_path}")

            # Load the dataset
            dataset = tifffile.imread(dataset_path).astype(np.uint16)

            # check the data does not contain nan or inf values
            if np.isnan(dataset).any() or np.isinf(dataset).any():
                raise ValueError(f"Dataset {dataset_path} contains NaN or Inf values.")

            all_results[region] = {}

            for method in methods:
                print(f"Running method: {method}")
                if method == "jpeg2000":
                    save_band_images = True
                    quality_levels = jpeg2000_qualities if jpeg2000_qualities is not None else [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 100]
                    results_dict = lsm_jpeg2000(quality_levels, dataset, region, patch_size_h, patch_size_w, save_band_images=save_band_images, max_value=max_value, device=self.device)

                elif method == "stf":
                    results_dict = lsm_stf(stf_checkpoints or {}, dataset, patch_size_h, patch_size_w, max_value=max_value, device=self.device, lambdas=stf_lambdas, checkpoint_template=stf_checkpoint_template, arch=stf_arch, region=region, existing_entries=_existing_entries)

                elif method == "mlic":
                    results_dict = lsm_mlic(mlic_checkpoints or {}, dataset, patch_size_h, patch_size_w, max_value=max_value, device=self.device, lambdas=mlic_lambdas, checkpoint_template=mlic_checkpoint_template, region=region, existing_entries=_existing_entries)

                elif method == "midnet":
                    results_dict = lsm_midnet(midnet_configs or {}, dataset_root=self.dataset_path, region=region, patch_size=patch_size_h, max_value=max_value, device=self.device)

                all_results[region][method] = results_dict

        print("All results:\n", all_results)
        csv_file = "outputs/results.csv"
        if not os.path.exists(csv_file):
            with open(csv_file, "w") as f:
                f.write("region,method,config,bits_total,mse_patch_avg,psnr_patch_avg,psnr_img,ssim_patch_avg,bpp,bpppb,encode_time,decode_time\n")
        with open(csv_file, "r") as f:
            lines = f.readlines()
        with open(csv_file, "w") as f:
            f.write(lines[0])  # write header
            existing_entries = set()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                existing_entries.add((parts[0], parts[1], parts[2]))
                f.write(line)
            for region in region_names:
                for method in methods:
                    results_dict = all_results[region][method]
                    for config, metrics in results_dict.items():
                        entry_key = (region, method, str(config))
                        if entry_key in existing_entries:
                            continue  # skip existing entry
                        f.write(
                            f"{region},{method},{config},{metrics.get('bits_total', '')},"
                            f"{metrics.get('mse_patch_avg', '')},{metrics.get('psnr_patch_avg', '')},"
                            f"{metrics.get('psnr_img', '')},{metrics.get('ssim_patch_avg', '')},"
                            f"{metrics.get('bpp', '')},{metrics.get('bpppb', '')},"
                            f"{metrics.get('encode_time', '')},{metrics.get('decode_time', '')}\n"
                        )

        # Compute and persist BD-PSNR / BD-SSIM vs JPEG 2000 anchor
        bd_results = compute_bd_metrics(region_names, anchor_method="jpeg2000", csv_file=csv_file)
        if bd_results:
            _write_bd_metrics(bd_results, csv_file="outputs/bd_metrics.csv")
            print("\nBD metrics (vs JPEG 2000):")
            for region in sorted(bd_results):
                print(f"  {region}:")
                for method_key in sorted(bd_results[region]):
                    m = bd_results[region][method_key]
                    bd_p = m["bd_psnr"]
                    bd_s = m["bd_ssim"]
                    p_str = f"{bd_p:+.4f} dB" if bd_p == bd_p else "nan"
                    s_str = f"{bd_s:+.6f}" if bd_s == bd_s else "nan"
                    print(f"    {method_key}: BD-PSNR={p_str}, BD-SSIM={s_str}")

        return all_results


def compute_bd_metrics(region_names, anchor_method="jpeg2000", csv_file="outputs/results.csv"):
    """Compute BD-PSNR and BD-SSIM for all methods vs anchor from results CSV.

    STF entries are split into sub-groups by architecture (prefix of config,
    e.g. 'cnn' or 'stf'), so each arch is treated as a separate RD curve.

    Returns:
        dict: {region: {method_key: {bd_psnr: float, bd_ssim: float}}}
    """
    if not os.path.exists(csv_file):
        return {}

    # column indices (0-based): region=0, method=1, config=2, psnr_patch_avg=5,
    # ssim_patch_avg=7, bpp=8
    curves = {}
    with open(csv_file, "r") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            region, method, config = parts[0], parts[1], parts[2]
            if region_names and region not in region_names:
                continue
            try:
                bpp = float(parts[8])
                psnr = float(parts[5])
                ssim = float(parts[7])
            except ValueError:
                continue
            # STF: split by arch prefix (e.g. "cnn-1" → key "stf-cnn")
            if method == "stf":
                arch = config.split("-")[0]
                method_key = f"stf-{arch}"
            else:
                method_key = method
            key = (region, method_key)
            if key not in curves:
                curves[key] = {"bpp": [], "psnr": [], "ssim": []}
            curves[key]["bpp"].append(bpp)
            curves[key]["psnr"].append(psnr)
            curves[key]["ssim"].append(ssim)

    results = {}
    target_regions = list(region_names) if region_names else sorted({r for r, _ in curves})
    for region in target_regions:
        anchor_key = (region, anchor_method)
        if anchor_key not in curves:
            continue
        anchor = curves[anchor_key]
        results[region] = {}
        for (reg, method_key), data in curves.items():
            if reg != region or method_key == anchor_method:
                continue
            bd_psnr = bjontegaard_delta(anchor["bpp"], anchor["psnr"], data["bpp"], data["psnr"])
            bd_ssim = bjontegaard_delta(anchor["bpp"], anchor["ssim"], data["bpp"], data["ssim"])
            results[region][method_key] = {"bd_psnr": bd_psnr, "bd_ssim": bd_ssim}
    return results


def _write_bd_metrics(bd_results, csv_file="outputs/bd_metrics.csv", anchor_method="jpeg2000"):
    """Merge BD-PSNR / BD-SSIM results into csv_file (upsert per region+method)."""
    import os as _os
    existing: dict = {}
    if _os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            reader = __import__("csv").DictReader(f)
            for row in reader:
                existing[(row["region"], row["method"])] = row

    for region in bd_results:
        for method_key in bd_results[region]:
            m = bd_results[region][method_key]
            bd_p = m["bd_psnr"]
            bd_s = m["bd_ssim"]
            existing[(region, method_key)] = {
                "region": region,
                "method": method_key,
                "bd_psnr": f"{bd_p:.4f}" if bd_p == bd_p else "nan",
                "bd_ssim": f"{bd_s:.6f}" if bd_s == bd_s else "nan",
                "anchor": anchor_method,
            }

    with open(csv_file, "w") as f:
        f.write("region,method,bd_psnr,bd_ssim,anchor\n")
        for (region, method_key) in sorted(existing):
            row = existing[(region, method_key)]
            f.write(f"{row['region']},{row['method']},{row['bd_psnr']},{row['bd_ssim']},{row['anchor']}\n")
    print(f"BD metrics written to {csv_file}")


def lsm_jpeg2000(qualities, dataset_u16, dataset_tag, patch_size_h, patch_size_w, save_band_images=False, max_value=65535.0, device="cuda"):
    H, W, C = dataset_u16.shape
    h_count = H // patch_size_h
    w_count = W // patch_size_w
    valid_px = H * W
    print(f"Image shape: {dataset_u16.shape}, h_count: {h_count}, w_count: {w_count}")

    # save_original_band_images(dataset, save_prefix=dataset_tag, skip_exist=True)
    # save_j2k_band_images(dataset.shape[2], quality_levels, save_prefix=dataset_tag, skip_exist=True)
    results = {}

    # todo: make dir if not exist

    for quality in qualities:
        print(f"Processing quality: {quality}")
        bits_total, encode_time_total, decode_time_total = 0, 0.0, 0.0

        total_patches = h_count * w_count * C
        with tqdm(total=total_patches) as pbar:

            for band in range(C):
                if save_band_images:
                    # check if the image already exists
                    # source_filepath = f"../data/lsm/{dataset_tag + '_' if dataset_tag else ''}j2k_b{band}.png"
                    # codec_filepath = f"../data/lsm/{dataset_tag + '_' if dataset_tag else ''}j2k_b{band}_q{quality}.jp2"
                    codec_path = f"outputs/jpeg2000/{dataset_tag + '_' if dataset_tag else ''}j2k_b{band}_q{quality}.jp2"
                    # todo: use existing functions
                else:
                    # only use temporary files
                    fd, codec_path = mkstemp(suffix=".jp2")
                    os.close(fd)

                img16 = dataset_u16[:, :, band].astype(np.uint16)

                encode_time = Encoders.jpeg2000_encode(img16, quality, codec_path)
                encode_time_total += encode_time

                size_bits = filesize(codec_path) * 8
                bits_total += size_bits

                recon, decode_time = Decoders.jpeg2000_decode(codec_path)
                decode_time_total += decode_time

                if not save_band_images:
                    os.remove(codec_path)

                recon = torch.from_numpy(recon.astype(np.float32)).to(device)
                orig = torch.from_numpy(dataset_u16.astype(np.float32)[:, :, band]).to(device)

                for h in range(h_count):
                    for w in range(w_count):
                        original_patch = orig[h * patch_size_h : (h + 1) * patch_size_h, w * patch_size_w : (w + 1) * patch_size_w]
                        recon_patch = recon[h * patch_size_h : (h + 1) * patch_size_h, w * patch_size_w : (w + 1) * patch_size_w]
                        results_patch = calc_metrics_patch(recon_patch, original_patch, ssim=True, max_value=max_value)
                        if (h, w, band) == (0, 0, 0):
                            results_accum = {k: v for k, v in results_patch.items()}
                        else:
                            for k, v in results_patch.items():
                                results_accum[k] += v
                        pbar.update(1)

        mse_img = results_accum["mse"]
        mse_patch_avg = mse_img / total_patches
        psnr_patch_avg = results_accum["psnr_patch"] / total_patches
        psnr_img = float(20.0 * math.log10(max_value) - 10.0 * math.log10(mse_img / (total_patches) + 1e-8))
        ssim_patch_avg = results_accum["ssim_patch"] / total_patches
        bpp = bits_total / valid_px
        bpppb = bpp / C

        results[quality] = {
            "bits_total": "{:.1f}".format(bits_total),
            "mse_patch_avg": "{:.4f}".format(mse_patch_avg),
            "psnr_patch_avg": "{:.4f}".format(psnr_patch_avg),
            "psnr_img": "{:.4f}".format(psnr_img),
            "ssim_patch_avg": "{:.8f}".format(ssim_patch_avg),
            "bpp": "{:.4f}".format(bpp),
            "bpppb": "{:.4f}".format(bpppb),
            "encode_time": "{:.6f}".format(encode_time_total),
            "decode_time": "{:.6f}".format(decode_time_total),
        }

        result = (
            f"Quality {quality} - Total delivery (bits): {bits_total:.1f}, "
            f"PSNR (img): {psnr_img:.4f}, PSNR (patch avg): {psnr_patch_avg:.4f}, MSE (patch avg): {mse_patch_avg:.4f}, "
            f"SSIM (patch avg): {ssim_patch_avg:.8f}, bpp: {bpp:.4f}, bpp per band: {bpppb:.4f}, "
            f"Encode time: {encode_time_total:.6f}s, Decode time: {decode_time_total:.6f}s"
        )
        print(result)

    return results


def _format_template(template: str, lambda_value: float, fold: Optional[int] = None, arch: Optional[str] = None) -> str:
    mapping = {
        "lambda_value": lambda_value,
        "lambda_tag": _format_lambda_tag(lambda_value),
    }
    if fold is not None:
        mapping["fold"] = fold
    if arch is not None:
        mapping["arch"] = arch
    return template.format(**mapping)


def lsm_stf(
    checkpoint_map,
    dataset_u16,
    patch_size_h,
    patch_size_w,
    max_value=65535.0,
    device="cuda",
    lambdas=None,
    checkpoint_template=None,
    arch="stf",
    region=None,
    existing_entries=None,
):
    lambda_values = [float(v) for v in (lambdas or [])]

    try:
        from models.stf.compressai.zoo import models as compressai_models
    except ImportError:
        try:
            from compressai.zoo import models as compressai_models
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime requirement
            raise ModuleNotFoundError("compressai is required for STF evaluation. Install via `pip install compressai`.") from exc

    if arch not in compressai_models:
        raise ValueError(f"Unknown STF architecture '{arch}'. Available: {sorted(compressai_models.keys())}")

    resolved = {}
    lambda_lookup = {}
    for key, path_str in (checkpoint_map or {}).items():
        key_str = str(key)
        resolved[key_str] = path_str
        try:
            lambda_lookup[key_str] = float(key_str)
        except ValueError:
            pass

    if checkpoint_template:
        if not lambda_values:
            raise ValueError("STF lambdas must be provided when using a checkpoint template.")
        for lam in lambda_values:
            key_str = f"{lam}"
            if key_str not in resolved:
                resolved[key_str] = _format_template(checkpoint_template, lam, arch=arch)
            lambda_lookup[key_str] = lam

    if not resolved:
        raise ValueError("No STF checkpoints resolved. Provide --stf-checkpoint entries or --stf-checkpoint-template with --stf-lambdas.")

    if lambda_values and not checkpoint_template:
        missing = []
        for lam in lambda_values:
            matched = any(math.isclose(val, lam, rel_tol=1e-6, abs_tol=1e-6) for val in lambda_lookup.values())
            if not matched:
                missing.append(lam)
        if missing:
            raise ValueError(f"Missing STF checkpoints for lambdas: {missing}")

    entries = []
    for key, path_str in resolved.items():
        lam_val = lambda_lookup.get(key)
        if lam_val is None:
            try:
                lam_val = float(key)
            except ValueError:
                lam_val = None
        label = f"{lam_val:.4g}" if lam_val is not None else str(key)
        checkpoint_path = Path(path_str)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"STF checkpoint '{checkpoint_path}' does not exist")
        entries.append((label, lam_val, checkpoint_path))

    H, W, C = dataset_u16.shape
    h_count = H // patch_size_h
    w_count = W // patch_size_w
    valid_px = H * W

    if h_count == 0 or w_count == 0:
        raise ValueError("Patch size too large for dataset dimensions.")

    # Number of patches to process per GPU call. Tune down if VRAM is tight.
    INFER_BATCH = 256

    # SSIM measure (cached, CPU)
    ssim_measure = _get_ssim_measure(torch.device("cpu"), max_value)

    results = {}

    for label, lambda_val, checkpoint_path in entries:
        result_key = f"{arch}-{label}"

        # --- Early skip: result already in CSV ---
        if existing_entries is not None and region is not None:
            if (region, "stf", result_key) in existing_entries:
                print(f"STF-{arch} {label} for region '{region}' already in CSV — skipping inference.")
                continue

        # --- Model loading with cache ---
        cache_key = str(checkpoint_path)
        if cache_key not in _stf_model_cache:
            state = torch.load(checkpoint_path, map_location=device)
            model = compressai_models[arch]().to(device)
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict)
            model.update()
            model.eval()
            _stf_model_cache[cache_key] = model
            print(f"Loaded and cached checkpoint: {checkpoint_path}")
        else:
            model = _stf_model_cache[cache_key]
            print(f"Using cached model for: {checkpoint_path}")

        bits_total = 0.0
        encode_time_total = 0.0
        decode_time_total = 0.0
        mse_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        N_band = h_count * w_count
        total_patches = N_band * C
        eps = 1e-8
        log20_max = 20.0 * math.log10(max_value)
        all_hw = [(h_idx, w_idx) for h_idx in range(h_count) for w_idx in range(w_count)]

        with torch.no_grad():
            with tqdm(total=total_patches, desc=f"STF-{arch} {label}") as pbar:
                for band in range(C):
                    for start in range(0, N_band, INFER_BATCH):
                        end = min(start + INFER_BATCH, N_band)
                        # Load only this sub-batch's patches — avoids full-band pre-allocation
                        sub_np = np.stack([
                            dataset_u16[
                                h_idx * patch_size_h:(h_idx + 1) * patch_size_h,
                                w_idx * patch_size_w:(w_idx + 1) * patch_size_w,
                                band,
                            ].astype(np.float32)
                            for h_idx, w_idx in all_hw[start:end]
                        ], axis=0)  # [B, H, W]
                        sub_orig = torch.from_numpy(sub_np)  # CPU float32

                        # Normalise and send to GPU: [B, 1, H, W]
                        batch_gpu = (sub_orig / max_value).unsqueeze(1).to(device)

                        bits_list, x_hat_cpu, enc_t, dec_t = model.compress_and_reconstruct_batch(batch_gpu)
                        encode_time_total += enc_t
                        decode_time_total += dec_t

                        bits_total += sum(bits_list)
                        recon_sub = x_hat_cpu * max_value  # [B, H, W]

                        # Per-sub-batch metrics — no full-band tensor accumulation
                        mse_sub = ((recon_sub - sub_orig) ** 2).mean(dim=(1, 2))  # [B]
                        mse_sum += mse_sub.sum().item()
                        psnr_sum += (log20_max - 10.0 * torch.log10(mse_sub + eps)).sum().item()
                        ssim_val = ssim_measure(recon_sub.unsqueeze(1), sub_orig.unsqueeze(1))
                        ssim_sum += float(ssim_val) * (end - start)

                        pbar.update(end - start)

        mse_patch_avg = mse_sum / total_patches
        psnr_patch_avg = psnr_sum / total_patches
        psnr_img = float(log20_max - 10.0 * math.log10(mse_patch_avg + eps))
        ssim_patch_avg = ssim_sum / total_patches
        bpp = bits_total / valid_px
        bpppb = bpp / C

        results[result_key] = {
            "bits_total": f"{bits_total:.1f}",
            "mse_patch_avg": f"{mse_patch_avg:.4f}",
            "psnr_patch_avg": f"{psnr_patch_avg:.4f}",
            "psnr_img": f"{psnr_img:.4f}",
            "ssim_patch_avg": f"{ssim_patch_avg:.8f}",
            "bpp": f"{bpp:.4f}",
            "bpppb": f"{bpppb:.4f}",
            "encode_time": f"{encode_time_total:.6f}",
            "decode_time": f"{decode_time_total:.6f}",
        }

        lambda_note = f" (lambda={lambda_val:.4g})" if lambda_val is not None else ""
        print(f"STF-{arch} {label}{lambda_note} - Total bits: {bits_total:.1f}, PSNR (img): {psnr_img:.4f}, "
              f"PSNR (patch avg): {psnr_patch_avg:.4f}, bpp: {bpp:.4f}, bpppb: {bpppb:.4f}")

    return results


def lsm_mlic(
    checkpoint_map,
    dataset_u16,
    patch_size_h,
    patch_size_w,
    max_value=65535.0,
    device="cuda",
    lambdas=None,
    checkpoint_template=None,
    region=None,
    existing_entries=None,
):
    lambda_values = [float(v) for v in (lambdas or [])]

    print(checkpoint_map, checkpoint_template, lambda_values)

    if checkpoint_template:
        if not lambda_values:
            raise ValueError("MLIC lambdas must be provided when using a checkpoint template.")

    resolved = {}
    lambda_lookup = {}
    for key, path_str in (checkpoint_map or {}).items():
        key_str = str(key)
        resolved[key_str] = path_str
        try:
            lambda_lookup[key_str] = float(key_str)
        except ValueError:
            pass

    if checkpoint_template:
        for lam in lambda_values:
            key_str = f"{lam}"
            if key_str not in resolved:
                resolved[key_str] = _format_template(checkpoint_template, lam, arch="mlic")
            lambda_lookup[key_str] = lam

    if not resolved:
        raise ValueError("No MLIC checkpoints resolved. Provide --mlic-checkpoint entries or --mlic-checkpoint-template with --mlic-lambdas.")

    if lambda_values and not checkpoint_template:
        missing = []
        for lam in lambda_values:
            matched = any(math.isclose(val, lam, rel_tol=1e-6, abs_tol=1e-6) for val in lambda_lookup.values())
            if not matched:
                missing.append(lam)
        if missing:
            raise ValueError(f"Missing MLIC checkpoints for lambdas: {missing}")

    config = model_config()
    H, W, C = dataset_u16.shape
    h_count = H // patch_size_h
    w_count = W // patch_size_w
    valid_px = H * W

    if h_count == 0 or w_count == 0:
        raise ValueError("Patch size too large for dataset dimensions.")

    # Number of patches per GPU call. Tune down if VRAM is tight.
    INFER_BATCH = 256

    # SSIM measure (cached, CPU)
    ssim_measure = _get_ssim_measure(torch.device("cpu"), max_value)

    results = {}

    # Build sorted entry list
    entries = []
    for key, path_str in resolved.items():
        lam_val = lambda_lookup.get(key)
        label = f"{lam_val:.4g}" if lam_val is not None else str(key)
        checkpoint_path = Path(path_str)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MLIC checkpoint '{checkpoint_path}' does not exist")
        entries.append((label, lam_val, checkpoint_path))

    N_band = h_count * w_count
    total_patches_per_lambda = N_band * C
    eps = 1e-8
    log20_max = 20.0 * math.log10(max_value)
    all_hw = [(h_idx, w_idx) for h_idx in range(h_count) for w_idx in range(w_count)]

    for label, lam_val, checkpoint_path in entries:
        result_key = f"mlic-{label}"

        # --- Early skip: result already in CSV ---
        if existing_entries is not None and region is not None:
            if (region, "mlic", result_key) in existing_entries:
                print(f"MLIC {label} for region '{region}' already in CSV — skipping inference.")
                continue

        # --- Model loading with cache ---
        cache_key = str(checkpoint_path)
        if cache_key not in _mlic_model_cache:
            state = torch.load(checkpoint_path, map_location=device)
            model = MLICPlusPlus(config=config).to(device)
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict)
            model.update(force=True)
            model.eval()
            _mlic_model_cache[cache_key] = model
            print(f"Loaded and cached MLIC checkpoint: {checkpoint_path}")
        else:
            model = _mlic_model_cache[cache_key]
            print(f"Using cached MLIC model for: {checkpoint_path}")

        bits_total = 0.0
        encode_time_total = 0.0
        decode_time_total = 0.0
        mse_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0

        with torch.no_grad():
            with tqdm(total=total_patches_per_lambda, desc=f"MLIC {label}") as pbar:
                for band in range(C):
                    for start in range(0, N_band, INFER_BATCH):
                        end = min(start + INFER_BATCH, N_band)
                        # Load only this sub-batch's patches
                        sub_np = np.stack([
                            dataset_u16[
                                h_idx * patch_size_h:(h_idx + 1) * patch_size_h,
                                w_idx * patch_size_w:(w_idx + 1) * patch_size_w,
                                band,
                            ].astype(np.float32)
                            for h_idx, w_idx in all_hw[start:end]
                        ], axis=0)  # [B, H, W]
                        sub_orig = torch.from_numpy(sub_np)  # CPU float32

                        # MLIC expects 3-channel input; replicate single band
                        batch_gpu = (sub_orig / max_value).unsqueeze(1).to(device)  # [B, 1, H, W]

                        bits_list, x_hat_cpu, enc_t, dec_t = model.compress_and_reconstruct_batch(batch_gpu)
                        encode_time_total += enc_t
                        decode_time_total += dec_t

                        bits_total += sum(bits_list)
                        recon_sub = x_hat_cpu * max_value  # [B, H, W]

                        # Per-sub-batch metrics
                        mse_sub = ((recon_sub - sub_orig) ** 2).mean(dim=(1, 2))  # [B]
                        mse_sum += mse_sub.sum().item()
                        psnr_sum += (log20_max - 10.0 * torch.log10(mse_sub + eps)).sum().item()
                        ssim_val = ssim_measure(recon_sub.unsqueeze(1), sub_orig.unsqueeze(1))
                        ssim_sum += float(ssim_val) * (end - start)

                        pbar.update(end - start)

        mse_patch_avg = mse_sum / total_patches_per_lambda
        psnr_patch_avg = psnr_sum / total_patches_per_lambda
        psnr_img = float(log20_max - 10.0 * math.log10(mse_patch_avg + eps))
        ssim_patch_avg = ssim_sum / total_patches_per_lambda
        bpp = bits_total / valid_px
        bpppb = bpp / C

        results[result_key] = {
            "bits_total": f"{bits_total:.1f}",
            "mse_patch_avg": f"{mse_patch_avg:.4f}",
            "psnr_patch_avg": f"{psnr_patch_avg:.4f}",
            "psnr_img": f"{psnr_img:.4f}",
            "ssim_patch_avg": f"{ssim_patch_avg:.8f}",
            "bpp": f"{bpp:.4f}",
            "bpppb": f"{bpppb:.4f}",
            "encode_time": f"{encode_time_total:.6f}",
            "decode_time": f"{decode_time_total:.6f}",
        }

        lambda_note = f" (lambda={lam_val:.4g})" if lam_val is not None else ""
        print(f"MLIC {label}{lambda_note} - Total bits: {bits_total:.1f}, PSNR (img): {psnr_img:.4f}, "
              f"PSNR (patch avg): {psnr_patch_avg:.4f}, bpp: {bpp:.4f}, bpppb: {bpppb:.4f}")

    return results


def _time_intermediate_codec(dataset_root, region, method: str, param):
    """Time encode+decode of the JPEG2000 intermediate consumed by MIDNet.

    Returns (encode_time_seconds, decode_time_seconds). Bands are timed
    independently and summed, matching ``lsm_jpeg2000``'s accounting. The
    reconstruction is discarded; only the timings are kept. Falls back to
    (0.0, 0.0) if the intermediate method is unsupported.
    """
    image_path = None
    region_dir = Path(dataset_root) / region
    for candidate in ("image.tiff", "image.tif"):
        p = region_dir / candidate
        if p.is_file():
            image_path = p
            break
    if image_path is None:
        return 0.0, 0.0

    arr = np.asarray(tifffile.imread(image_path))
    if arr.ndim != 3:
        return 0.0, 0.0
    H, W, C = arr.shape

    encode_total = 0.0
    decode_total = 0.0
    if method == "jpeg2000":
        for band in range(C):
            img16 = arr[:, :, band].astype(np.uint16)
            fd, codec_path = mkstemp(suffix=".jp2")
            os.close(fd)
            try:
                encode_total += Encoders.jpeg2000_encode(img16, int(param), codec_path)
                _, dec_t = Decoders.jpeg2000_decode(codec_path)
                decode_total += dec_t
            finally:
                if os.path.exists(codec_path):
                    os.remove(codec_path)
    return encode_total, decode_total


def lsm_midnet(
    configs,
    dataset_root,
    region,
    patch_size,
    max_value=65535.0,
    device="cuda",
):
    if not configs:
        raise ValueError("No MIDNet configurations provided for benchmarking.")

    device = torch.device(device)
    root_path = Path(dataset_root)
    default_jpeg_root = Path.cwd() / "outputs" / "jpeg2000"

    results = {}

    for label, cfg in configs.items():
        checkpoint_path = Path(cfg.get("checkpoint", "")).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MIDNet checkpoint '{checkpoint_path}' does not exist")

        method = cfg.get("intermediate_method", "jpeg2000")
        if method != "jpeg2000":
            raise ValueError("This benchmark release supports MIDNet with JPEG2000 intermediates only")
        quality = cfg.get("intermediate_quality")
        total_paths = int(cfg.get("total_paths", 3))
        resblocks = int(cfg.get("resblocks", 32))
        n_feats = int(cfg.get("n_feats", 64))
        delivery_dtype = cfg.get("delivery_dtype")  # e.g. "fp16" to halve bpp_model
        batch_size = int(cfg.get("batch_size", 256))
        sample_ratio = float(cfg.get("sample_ratio", 1.0))
        max_patches_raw = cfg.get("max_patches")
        if max_patches_raw is None or str(max_patches_raw).lower() == "none":
            max_patches = None
        else:
            max_patches = int(max_patches_raw)
        seed_raw = cfg.get("seed")
        if seed_raw is None or str(seed_raw).lower() == "none":
            seed = None
        else:
            seed = int(seed_raw)
        jpeg_root = cfg.get("jpeg2000_root")
        jpeg_root_path = Path(jpeg_root).expanduser() if jpeg_root else default_jpeg_root

        if quality is None:
            raise ValueError(f"MIDNet config '{label}' requires 'intermediate_quality' for JPEG2000 intermediates")
        intermediate_param = quality

        dataset = MIDNetPatchDataset(
            root_path,
            region,
            patch_size,
            method,
            intermediate_param,
            total_paths,
            sample_ratio,
            max_patches,
            seed,
            jpeg2000_root=jpeg_root_path,
        )

        info = dataset.info()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        model = BCResBlocksShared(
            info.channel_means.to(device=device),
            total_paths=total_paths,
            n_resblocks=resblocks,
            channels=1,
            n_feats=n_feats,
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        criterion = nn.HuberLoss(reduction="mean", delta=0.1).to(device)
        metrics = midnet_evaluate(model, criterion, loader, device)

        model_bits = _model_weight_bits(model, delivery_dtype=delivery_dtype)
        bits_total = float(info.intermediate_bits + model_bits)
        pixel_count = max(1, info.pixel_count)
        band_count = max(1, info.band_count)
        bpp = bits_total / pixel_count
        bpppb = bpp / band_count
        psnr_img = float(20.0 * math.log10(max_value) - 10.0 * math.log10(metrics["mse"] + 1e-8))

        # Codec timings for MIDNet (a neural *delivery* method):
        #   encode_time = JPEG2000 intermediate encode time only
        #                 (the network is not involved in encoding).
        #   decode_time = intermediate decode time + model forward (refinement)
        #                 time measured during evaluation.
        intermediate_encode_time, intermediate_decode_time = _time_intermediate_codec(
            dataset_root, region, method, intermediate_param
        )
        encode_time = intermediate_encode_time
        decode_time = intermediate_decode_time + float(metrics.get("forward_time", 0.0))

        results[label] = {
            "bits_total": f"{bits_total:.1f}",
            "mse_patch_avg": f"{metrics['mse']:.4f}",
            "psnr_patch_avg": f"{metrics['psnr']:.4f}",
            "psnr_img": f"{psnr_img:.4f}",
            "ssim_patch_avg": f"{metrics['ssim']:.8f}",
            "bpp": f"{bpp:.4f}",
            "bpppb": f"{bpppb:.4f}",
            "encode_time": f"{encode_time:.6f}",
            "decode_time": f"{decode_time:.6f}",
        }

        print(
            f"MIDNet {label} - Total bits: {bits_total:.1f}, PSNR (img): {psnr_img:.4f}, "
            f"PSNR (patch avg): {metrics['psnr']:.4f}, bpp: {bpp:.4f}, bpppb: {bpppb:.4f}, "
            f"Encode time: {encode_time:.6f}s, Decode time: {decode_time:.6f}s"
        )

    return results
