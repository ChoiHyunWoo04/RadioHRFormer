# @Description:
# RadioHRFormer FLOPs-only benchmark.
#
# This file is intentionally separate from the runtime benchmark so that
# torch.profiler / Kineto / CUPTI cannot affect paper-facing latency results.
#
# It performs one profiler forward per requested resolution and reports:
#   - GFLOPs/sample
#   - relative FLOPs vs the first resolution
#   - top FLOP-bearing operator groups
#
# No latency result from this script should be reported in the paper.

import argparse
import csv
import inspect
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List

import torch
from torch.profiler import ProfilerActivity, profile as torch_profile


# -----------------------------------------------------------------------------
# Project-local imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) in {"tools", "scripts"}:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.hrformer_regressor import HRFormerRadioMapRegressor

MODEL_SOURCE_FILE = os.path.abspath(
    inspect.getfile(HRFormerRadioMapRegressor)
)

DEFAULT_RESOLUTIONS = [256, 512, 768, 1024]


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    return state_dict


def load_model_state(
    model: torch.nn.Module,
    weight_path: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        weight_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state_dict."
        )

    model.load_state_dict(
        normalize_state_dict_keys(state_dict),
        strict=True,
    )


def get_autocast_context(
    device: torch.device,
    precision: str,
):
    if precision == "fp32":
        return nullcontext()

    if precision == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The selected CUDA device does not "
                "support bfloat16."
            )
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    raise ValueError(
        f"Unsupported precision: {precision}"
    )


def make_synthetic_input(
    batch_size: int,
    in_channels: int,
    resolution: int,
    device: torch.device,
    seed: int,
    obstacle_density: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x = torch.zeros(
        (
            batch_size,
            in_channels,
            resolution,
            resolution,
        ),
        dtype=torch.float32,
    )

    if in_channels == 1:
        x.uniform_(
            0.0,
            1.0,
            generator=generator,
        )
    else:
        masks = torch.rand(
            (
                batch_size,
                in_channels - 1,
                resolution,
                resolution,
            ),
            generator=generator,
        )
        x[:, :-1] = (
            masks < obstacle_density
        ).to(dtype=torch.float32)

        rows = torch.randint(
            0,
            resolution,
            (batch_size,),
            generator=generator,
        )
        cols = torch.randint(
            0,
            resolution,
            (batch_size,),
            generator=generator,
        )
        batches = torch.arange(batch_size)

        x[
            batches,
            in_channels - 1,
            rows,
            cols,
        ] = 1.0

    return x.to(device=device).contiguous()


def build_model(
    cfg: Dict[str, Any],
    weight_path: str,
    device: torch.device,
) -> torch.nn.Module:
    model = HRFormerRadioMapRegressor(cfg).to(device)
    load_model_state(
        model=model,
        weight_path=weight_path,
        device=device,
    )
    model.eval()
    return model


def profile_flops(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    precision: str,
    top_k: int,
) -> Dict[str, Any]:
    activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]

    torch.cuda.synchronize(device)

    with torch.inference_mode():
        with torch_profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
        ) as prof:
            with get_autocast_context(
                device,
                precision,
            ):
                output = model(x)

            torch.cuda.synchronize(device)

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "Expected Tensor output, received "
            f"{type(output).__name__}."
        )

    expected_hw = tuple(
        int(value) for value in x.shape[-2:]
    )
    output_hw = tuple(
        int(value) for value in output.shape[-2:]
    )
    if output_hw != expected_hw:
        raise RuntimeError(
            "Output resolution mismatch: "
            f"input={tuple(x.shape)}, "
            f"output={tuple(output.shape)}."
        )

    records: List[Dict[str, Any]] = []
    total_flops = 0

    for event in prof.key_averages():
        value = getattr(event, "flops", 0)
        if value is None:
            value = 0

        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0

        if value <= 0:
            continue

        total_flops += value
        records.append(
            {
                "operator": str(event.key),
                "calls": int(event.count),
                "flops": value,
                "gflops": float(value) / 1e9,
            }
        )

    records.sort(
        key=lambda item: item["flops"],
        reverse=True,
    )

    if total_flops <= 0:
        raise RuntimeError(
            "torch.profiler returned zero FLOPs."
        )

    batch_size = int(x.shape[0])
    gflops_per_batch = total_flops / 1e9
    gflops_per_sample = (
        gflops_per_batch / batch_size
    )

    result = {
        "input_shape": [
            int(value) for value in x.shape
        ],
        "output_shape": [
            int(value) for value in output.shape
        ],
        "total_flops_per_batch": int(total_flops),
        "gflops_per_batch": float(gflops_per_batch),
        "gflops_per_sample": float(gflops_per_sample),
        "counted_operator_groups": len(records),
        "top_operators": records[:top_k],
    }

    del output
    return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this FLOPs-only script."
        )

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config_path)
    model = build_model(
        cfg=cfg,
        weight_path=args.weight_path,
        device=device,
    )

    in_channels = int(
        cfg["model"]["in_channels"]
    )

    print(f"Using device: {device}")
    print(f"Imported model source: {MODEL_SOURCE_FILE}")
    print(f"Loading weights: {args.weight_path}")
    print(
        "Mode: FLOPs only. "
        "Do NOT use this process for latency reporting."
    )

    results: List[Dict[str, Any]] = []

    for resolution in args.resolutions:
        print(
            f"\nProfiling FLOPs at "
            f"{resolution} x {resolution}"
        )

        x = make_synthetic_input(
            batch_size=args.batch_size,
            in_channels=in_channels,
            resolution=int(resolution),
            device=device,
            seed=args.seed + int(resolution),
            obstacle_density=args.obstacle_density,
        )

        profile = profile_flops(
            model=model,
            x=x,
            device=device,
            precision=args.precision,
            top_k=args.top_k,
        )

        result = {
            "resolution": int(resolution),
            "status": "ok",
            **profile,
        }
        results.append(result)

        print(
            f"  input shape:  "
            f"{profile['input_shape']}"
        )
        print(
            f"  output shape: "
            f"{profile['output_shape']}"
        )
        print(
            f"  FLOPs: "
            f"{profile['gflops_per_sample']:.6f} "
            "GFLOPs/sample"
        )

        del x
        torch.cuda.empty_cache()

    if results:
        base = results[0]
        base_res = int(base["resolution"])
        base_flops = float(
            base["gflops_per_sample"]
        )

        for result in results:
            res = int(result["resolution"])
            result["pixel_ratio_vs_base"] = (
                (res * res)
                / float(base_res * base_res)
            )
            result["rel_flops_vs_base"] = (
                float(result["gflops_per_sample"])
                / max(base_flops, 1e-12)
            )

    print("\nFLOPs scaling summary")
    for result in results:
        print(
            f"  {result['resolution']:4d}: "
            f"GFLOPs={result['gflops_per_sample']:.6f} | "
            f"Rel.FLOPs={result['rel_flops_vs_base']:.4f}x | "
            f"Pixels={result['pixel_ratio_vs_base']:.4f}x"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    metadata = {
        "benchmark": "RadioHRFormer FLOPs-only",
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "model_source_file": MODEL_SOURCE_FILE,
        "config_path": os.path.abspath(
            args.config_path
        ),
        "weight_path": os.path.abspath(
            args.weight_path
        ),
        "device": str(device),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "resolutions": [
            int(value)
            for value in args.resolutions
        ],
        "method": "torch.profiler.with_flops",
        "note": (
            "PyTorch profiler FLOPs cover supported "
            "FLOP-bearing operator families, mainly "
            "matmul and 2-D convolution. Elementwise, "
            "normalization, softmax, and interpolation "
            "may be omitted. Use the same FLOPs method "
            "when comparing models."
        ),
    }

    json_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_flops.json",
    )
    csv_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_flops.csv",
    )
    log_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_flops.txt",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "results": results,
            },
            f,
            indent=2,
        )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = [
            "resolution",
            "input_shape",
            "output_shape",
            "gflops_per_sample",
            "rel_flops_vs_base",
            "pixel_ratio_vs_base",
            "total_flops_per_batch",
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: result.get(key)
                    for key in fields
                }
            )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "RadioHRFormer Resolution FLOPs Benchmark "
            "(profiler-only process)\n"
        )
        f.write("=" * 88 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

        for result in results:
            f.write(
                f"\n[{result['resolution']} x "
                f"{result['resolution']}]\n"
            )
            f.write(
                f"input_shape: "
                f"{result['input_shape']}\n"
            )
            f.write(
                f"output_shape: "
                f"{result['output_shape']}\n"
            )
            f.write(
                f"GFLOPs/sample: "
                f"{result['gflops_per_sample']:.6f}\n"
            )
            f.write(
                f"Rel. FLOPs: "
                f"{result['rel_flops_vs_base']:.4f}x\n"
            )
            f.write(
                f"Pixel ratio: "
                f"{result['pixel_ratio_vs_base']:.4f}x\n"
            )
            f.write(
                "Top FLOP-bearing operators:\n"
            )
            for item in result["top_operators"]:
                f.write(
                    f"  {item['operator']:<28s} "
                    f"{item['gflops']:12.6f} GFLOPs "
                    f"| calls={item['calls']}\n"
                )

    print("\nFLOPs benchmark complete.")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print(f"LOG : {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RadioHRFormer FLOPs-only resolution benchmark."
        )
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/carsdpm_downstream.json",
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "./runs/"
            "radiohrformer_resolution_flops"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.15,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
