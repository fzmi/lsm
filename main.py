from __future__ import annotations
import argparse
import random
import torch
import numpy as np
import time
from datetime import datetime

from benchmark import LSMBenchmark
from utils.regions import CROSSVAL_GROUPS


def _parse_checkpoint_args(values):
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid STF checkpoint argument '{value}'. Expected tag=path format.")
        tag, path = value.split("=", 1)
        tag = tag.strip()
        path = path.strip()
        if not tag:
            raise ValueError(f"Invalid STF checkpoint argument '{value}': empty tag")
        mapping[tag] = path
    return mapping


def _lambda_tag(value: float) -> str:
    tag = f"{value:.4g}"
    tag = tag.replace(".", "p").replace("-", "m")
    return tag


def _format_template(template: str, lambda_value: float, fold: int, arch: str) -> str:
    return template.format(lambda_value=lambda_value, lambda_tag=_lambda_tag(lambda_value), fold=fold, arch=arch)


def _build_cv_checkpoint_map(template: str, lambdas: list[float], fold: int, arch: str) -> dict[str, str]:
    return {f"{lam:.4g}": _format_template(template, lam, fold, arch) for lam in lambdas}


def _print_cv_summary(results_collection):
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}

    for result in results_collection:
        if not result:
            continue
        for region, method_dict in result.items():
            for method, configs in method_dict.items():
                for config, metrics in configs.items():
                    for metric_name, value in metrics.items():
                        if value in ("", None):
                            continue
                        try:
                            value_f = float(value)
                        except ValueError:
                            continue
                        aggregated.setdefault(method, {}).setdefault(config, {}).setdefault(metric_name, 0.0)
                        counts.setdefault(method, {}).setdefault(config, {}).setdefault(metric_name, 0)
                        aggregated[method][config][metric_name] += value_f
                        counts[method][config][metric_name] += 1

    if not aggregated:
        print("No results to summarize.")
        return

    print("\nCross-validation averages:")
    for method in sorted(aggregated):
        for config in sorted(aggregated[method]):
            print(f"  {method} / {config}:")
            for metric_name in sorted(aggregated[method][config]):
                total = aggregated[method][config][metric_name]
                count = counts[method][config][metric_name]
                mean = total / count if count else float("nan")
                print(f"    {metric_name}: {mean:.4f} (n={count})")


def main():
    program_start_time = time.time()
    args = get_args()
    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device_str}")
    benchmark = LSMBenchmark(args.dataset_path, device=device_str)
    
    stf_checkpoints = _parse_checkpoint_args(args.stf_checkpoint) if args.stf_checkpoint else {}
    default_lambdas = [0.0483, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    stf_lambdas = args.stf_lambdas if args.stf_lambdas else default_lambdas
    mlic_checkpoints = _parse_checkpoint_args(args.mlic_checkpoint) if args.mlic_checkpoint else {}
    default_mlic_lambdas = [0.0483, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    mlic_lambdas = args.mlic_lambdas if args.mlic_lambdas else default_mlic_lambdas
    _midnet_label = f"midnet_q{args.midnet_intermediate_quality}"
    if args.midnet_delivery_dtype:
        _midnet_label = f"{_midnet_label}_d{args.midnet_delivery_dtype}"
    midnet_config = {
        _midnet_label: {
            "checkpoint": args.midnet_checkpoint,
            "intermediate_method": args.midnet_intermediate_method,
            "intermediate_quality": args.midnet_intermediate_quality,
            "total_paths": args.midnet_total_paths,
            "resblocks": args.midnet_resblocks,
            "n_feats": args.midnet_n_feats,
            "delivery_dtype": args.midnet_delivery_dtype,
        }
    }

    if args.cv_eval:

        if not args.methods:
            raise ValueError("Specify --methods when using --cv-eval")
        needs_stf = any(method == "stf" for method in args.methods)
        if needs_stf and not args.stf_checkpoint_template:
            raise ValueError("Cross-validation evaluation requires --stf-checkpoint-template when STF methods are selected")

        needs_mlic = any(method == "mlic" for method in args.methods)
        if needs_mlic and not args.mlic_checkpoint_template:
            raise ValueError("Cross-validation evaluation requires --mlic-checkpoint-template when mlic methods are selected")

        fold_results = []
        for fold_idx, train_group in enumerate(CROSSVAL_GROUPS, start=1):
            test_regions = [region for idx, group in enumerate(CROSSVAL_GROUPS, start=1) if idx != fold_idx for region in group]
            print(f"==== Fold {fold_idx}: training group {train_group} | testing regions {test_regions} ====")

            fold_checkpoints = None
            if needs_stf:
                fold_checkpoints = _build_cv_checkpoint_map(args.stf_checkpoint_template, stf_lambdas, fold_idx, args.stf_arch)

            mlic_fold_checkpoints = None
            if needs_mlic:
                mlic_fold_checkpoints = _build_cv_checkpoint_map(args.mlic_checkpoint_template, mlic_lambdas, fold_idx, "mlic")

            result = benchmark.run_benchmark(
                test_regions,
                args.methods,
                max_value=args.max_value,
                patch_size_h=args.patch_size_h,
                patch_size_w=args.patch_size_w,
                stf_checkpoints=fold_checkpoints,
                stf_lambdas=stf_lambdas,
                stf_checkpoint_template=None,
                stf_arch=args.stf_arch,
                mlic_checkpoints=mlic_fold_checkpoints,
                mlic_lambdas=mlic_lambdas,
                mlic_checkpoint_template=None,
                midnet_configs=midnet_config,
                jpeg2000_qualities=args.jpeg2000_qualities,
            )
            fold_results.append(result)

        _print_cv_summary(fold_results)

    else:
        if not args.methods:
            raise ValueError("Specify --methods when running evaluation")
        if not args.regions:
            raise ValueError("Specify --regions or use --cv-eval")
        benchmark.run_benchmark(
            args.regions,
            args.methods,
            max_value=args.max_value,
            patch_size_h=args.patch_size_h,
            patch_size_w=args.patch_size_w,
            stf_checkpoints=stf_checkpoints,
            stf_lambdas=stf_lambdas,
            stf_checkpoint_template=args.stf_checkpoint_template,
            stf_arch=args.stf_arch,
            mlic_checkpoints=mlic_checkpoints,
            mlic_lambdas=mlic_lambdas,
            mlic_checkpoint_template=args.mlic_checkpoint_template,
            midnet_configs=midnet_config,
            jpeg2000_qualities=args.jpeg2000_qualities,
        )

    program_end_time = time.time()
    print("Total program time: ", time.strftime("%H:%M:%S", time.gmtime(program_end_time - program_start_time)))


def get_args():
    parser = argparse.ArgumentParser(description="Run LSM benchmark for selected methods.")
    # Dataset options
    parser.add_argument("--dataset-path", "-bp", type=str, required=True, help="Path to the benchmark folder")
    parser.add_argument("--regions", "-d", nargs="+", default=[], help="Specfic region(s) to use for benchmarking ([brisbane/camarillo/cambridge/hawick/kagoshima/lamington/logan-village/melbourne/munich/sydney/vancouver])")
    parser.add_argument("--max-value", "-mv", type=float, default=65535.0, help="Maximum value of the dataset")
    parser.add_argument("--patch-size-h", "-psh", type=int, default=64, help="Patch size height")
    parser.add_argument("--patch-size-w", "-psw", type=int, default=64, help="Patch size width")

    # Method options
    parser.add_argument("--methods", "-m", nargs="+", default=[], choices=["jpeg2000", "midnet", "mlic", "stf"], help="Benchmark method(s) to run")
    parser.add_argument("--jpeg2000-qualities", type=int, nargs="+", default=None, help="JPEG2000 quality levels to evaluate (overrides default)")

    parser.add_argument("--stf-checkpoint", action="append", default=[], help="Register an STF checkpoint as tag=path")
    parser.add_argument("--stf-checkpoint-template", type=str, help="Template for STF checkpoints (supports {lambda_value} / {lambda_tag})")
    parser.add_argument("--stf-lambdas", type=float, nargs="+", help="Lambda values to evaluate for STF")
    parser.add_argument("--stf-arch", choices=["cnn"], default="cnn", help="STF architecture to evaluate")

    parser.add_argument("--mlic-checkpoint", action="append", default=[], help="Register an MLIC checkpoint as tag=path")
    parser.add_argument("--mlic-checkpoint-template", type=str, help="Template for MLIC checkpoints (supports {lambda_value} / {lambda_tag})")
    parser.add_argument("--mlic-lambdas", type=float, nargs="+", help="Lambda values to evaluate for MLIC")

    parser.add_argument("--midnet-checkpoint", type=str, help="Path to MIDNet checkpoint")
    parser.add_argument("--midnet-intermediate-method", choices=["jpeg2000"], default="jpeg2000", help="MIDNet intermediate method")
    parser.add_argument("--midnet-intermediate-quality", type=int, default=10, help="MIDNet JPEG2000 intermediate quality factor")
    parser.add_argument("--midnet-total-paths", type=int, default=1, choices=[1, 3], help="Number of path branches in MIDNet")
    parser.add_argument("--midnet-resblocks", type=int, default=16, help="MIDNet number of residual blocks")
    parser.add_argument("--midnet-n-feats", type=int, default=64, help="MIDNet feature width (default 64; matches train_midnet.py)")
    parser.add_argument("--midnet-delivery-dtype", choices=["fp32", "fp16", "bf16", "int8"], default="fp16", help="Dtype assumed when counting MIDNet model bits for bpp accounting. Default 'fp16' matches the original-paper convention for delivered per-region weights. Pass 'fp32' for the raw checkpoint accounting.")

    parser.add_argument("--cv-eval", action="store_true", help="Evaluate using predefined 4-fold cross-validation groups")

    # Program options
    parser.add_argument("--save-arguments", "-sa", action="store_true", default=False)
    parser.add_argument("--output-tag", "-ot", type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"), help="Output tag")
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    main()
