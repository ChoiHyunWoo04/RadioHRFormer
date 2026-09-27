# @Description:
# RadioHRFormer paper-facing runtime benchmark.
#
# IMPORTANT:
# - This file intentionally does NOT import or invoke torch.profiler/CUPTI.
# - It measures only normal model.forward() latency and CUDA memory.
# - FLOPs must be measured with the separate FLOPs-only script.
#
# Recommended paper protocol:
#   batch size = 1
#   precision = fp32
#   resolutions = 256 512 768 1024
#   warm-up = 20
#   timed iterations = 100
#
# Run the whole script multiple times (e.g. 3 independent runs) for the final
# paper table if you want to quantify run-to-run GPU variation.

import argparse
import csv
import gc
import inspect
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import torch


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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = ["seed", "data", "model", "train"]
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise KeyError(
            f"Missing config keys: {missing}. "
            f"Loaded path: {config_path}. "
            f"Top-level keys: {list(cfg.keys())}"
        )
    return cfg


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
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "The loaded checkpoint does not contain a valid state_dict."
        )

    model.load_state_dict(
        normalize_state_dict_keys(state_dict),
        strict=True,
    )


def bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 ** 2)


def get_model_profile(
    model: torch.nn.Module,
    weight_path: Optional[str] = None,
) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )

    weight_file_size_mb = None
    if weight_path is not None and os.path.exists(weight_path):
        weight_file_size_mb = bytes_to_mb(os.path.getsize(weight_path))

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_size_mb": bytes_to_mb(param_bytes),
        "buffer_size_mb": bytes_to_mb(buffer_bytes),
        "model_size_param_buffer_mb": bytes_to_mb(
            param_bytes + buffer_bytes
        ),
        "weight_file_size_mb": weight_file_size_mb,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)


def get_autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()

    if device.type != "cuda":
        raise ValueError(
            f"{precision} benchmarking is supported only on CUDA."
        )

    if precision == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The selected CUDA device does not support bfloat16."
            )
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    raise ValueError(f"Unsupported precision: {precision}")


def percentile(values: Sequence[float], q: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    frac = position - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def summarize_times(values: Sequence[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
    }


# -----------------------------------------------------------------------------
# Synthetic input
# -----------------------------------------------------------------------------
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
        (batch_size, in_channels, resolution, resolution),
        dtype=torch.float32,
    )

    if in_channels == 1:
        x.uniform_(0.0, 1.0, generator=generator)
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


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Measurements
# -----------------------------------------------------------------------------
def measure_cuda_event_latency(
    model: torch.nn.Module,
    x: torch.Tensor,
    repeats: int,
    device: torch.device,
    precision: str,
) -> List[float]:
    times_ms: List[float] = []

    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with get_autocast_context(device, precision):
                output = model(x)
            end.record()

            end.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
            del output

    return times_ms


def measure_wall_clock_latency(
    model: torch.nn.Module,
    x: torch.Tensor,
    repeats: int,
    device: torch.device,
    precision: str,
) -> List[float]:
    times_ms: List[float] = []

    with torch.inference_mode():
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()

            with get_autocast_context(device, precision):
                output = model(x)

            synchronize(device)
            times_ms.append(
                (time.perf_counter() - start) * 1000.0
            )
            del output

    return times_ms


def benchmark_resolution(
    model: torch.nn.Module,
    resolution: int,
    in_channels: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    precision: str,
    seed: int,
    obstacle_density: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "resolution": int(resolution),
        "status": "pending",
        "error": None,
    }

    x = None
    output = None

    try:
        clear_cuda_cache(device)

        x = make_synthetic_input(
            batch_size=batch_size,
            in_channels=in_channels,
            resolution=resolution,
            device=device,
            seed=seed + resolution,
            obstacle_density=obstacle_density,
        )

        result["input_shape"] = [
            int(value) for value in x.shape
        ]

        # Shape verification before timing.
        with torch.inference_mode():
            with get_autocast_context(device, precision):
                output = model(x)
        synchronize(device)

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Expected Tensor output, received "
                f"{type(output).__name__}."
            )

        result["output_shape"] = [
            int(value) for value in output.shape
        ]

        expected_hw = (resolution, resolution)
        actual_hw = tuple(
            int(value) for value in output.shape[-2:]
        )
        if actual_hw != expected_hw:
            raise RuntimeError(
                "Output resolution mismatch: "
                f"input={tuple(x.shape)}, "
                f"output={tuple(output.shape)}, "
                f"expected={expected_hw}."
            )

        del output
        output = None

        # Warm-up.
        with torch.inference_mode():
            for _ in range(warmup):
                with get_autocast_context(device, precision):
                    output = model(x)
                del output
                output = None
        synchronize(device)

        # Memory baseline and peak are taken around the CUDA-event pass.
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)

        cuda_times = measure_cuda_event_latency(
            model=model,
            x=x,
            repeats=repeats,
            device=device,
            precision=precision,
        )

        allocated_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        # Separate synchronized wall-clock cross-check.
        wall_times = measure_wall_clock_latency(
            model=model,
            x=x,
            repeats=repeats,
            device=device,
            precision=precision,
        )

        cuda_stats = summarize_times(cuda_times)
        wall_stats = summarize_times(wall_times)

        result.update(
            {
                "status": "ok",
                "batch_size": int(batch_size),
                "precision": precision,
                "warmup_iterations": int(warmup),
                "timed_iterations": int(repeats),
                "cuda_event_ms_per_batch": cuda_stats,
                "cuda_event_ms_per_sample": {
                    key: value / batch_size
                    for key, value in cuda_stats.items()
                },
                "wall_clock_ms_per_batch": wall_stats,
                "wall_clock_ms_per_sample": {
                    key: value / batch_size
                    for key, value in wall_stats.items()
                },
                "wall_cuda_mean_ratio": (
                    wall_stats["mean"]
                    / max(cuda_stats["mean"], 1e-12)
                ),
                "allocated_before_mb": bytes_to_mb(
                    allocated_before
                ),
                "reserved_before_mb": bytes_to_mb(
                    reserved_before
                ),
                "allocated_after_mb": bytes_to_mb(
                    allocated_after
                ),
                "reserved_after_mb": bytes_to_mb(
                    reserved_after
                ),
                "peak_allocated_mb": bytes_to_mb(
                    peak_allocated
                ),
                "peak_reserved_mb": bytes_to_mb(
                    peak_reserved
                ),
                "incremental_peak_allocated_mb": bytes_to_mb(
                    max(0, peak_allocated - allocated_before)
                ),
                "incremental_peak_reserved_mb": bytes_to_mb(
                    max(0, peak_reserved - reserved_before)
                ),
            }
        )

    except torch.cuda.OutOfMemoryError as exc:
        result.update(
            {
                "status": "oom",
                "error": str(exc),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        if output is not None:
            del output
        if x is not None:
            del x
        clear_cuda_cache(device)

    return result


def add_scaling(results: List[Dict[str, Any]]) -> None:
    valid = [
        result for result in results
        if result.get("status") == "ok"
    ]
    if not valid:
        return

    base = valid[0]
    base_res = int(base["resolution"])
    base_cuda = float(
        base["cuda_event_ms_per_sample"]["mean"]
    )
    base_wall = float(
        base["wall_clock_ms_per_sample"]["mean"]
    )
    base_memory = float(
        base["incremental_peak_allocated_mb"]
    )

    for result in valid:
        res = int(result["resolution"])
        result["scaling_vs_first_resolution"] = {
            "pixel_ratio": (
                (res * res) / float(base_res * base_res)
            ),
            "cuda_latency_ratio": (
                float(
                    result[
                        "cuda_event_ms_per_sample"
                    ]["mean"]
                )
                / max(base_cuda, 1e-12)
            ),
            "wall_latency_ratio": (
                float(
                    result[
                        "wall_clock_ms_per_sample"
                    ]["mean"]
                )
                / max(base_wall, 1e-12)
            ),
            "incremental_memory_ratio": (
                float(
                    result[
                        "incremental_peak_allocated_mb"
                    ]
                )
                / max(base_memory, 1e-12)
            ),
        }


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------
def write_csv(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    fields = [
        "resolution",
        "status",
        "input_shape",
        "output_shape",
        "batch_size",
        "precision",
        "cuda_mean_ms_per_sample",
        "cuda_median_ms_per_sample",
        "cuda_p95_ms_per_sample",
        "wall_mean_ms_per_sample",
        "wall_median_ms_per_sample",
        "wall_p95_ms_per_sample",
        "peak_allocated_mb",
        "incremental_peak_allocated_mb",
        "peak_reserved_mb",
        "pixel_ratio_vs_base",
        "cuda_latency_ratio_vs_base",
        "wall_latency_ratio_vs_base",
        "memory_ratio_vs_base",
        "error",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in results:
            scaling = result.get(
                "scaling_vs_first_resolution",
                {},
            )
            cuda_stats = result.get(
                "cuda_event_ms_per_sample",
                {},
            )
            wall_stats = result.get(
                "wall_clock_ms_per_sample",
                {},
            )
            writer.writerow(
                {
                    "resolution": result["resolution"],
                    "status": result["status"],
                    "input_shape": result.get("input_shape"),
                    "output_shape": result.get("output_shape"),
                    "batch_size": result.get("batch_size"),
                    "precision": result.get("precision"),
                    "cuda_mean_ms_per_sample": cuda_stats.get(
                        "mean"
                    ),
                    "cuda_median_ms_per_sample": cuda_stats.get(
                        "median"
                    ),
                    "cuda_p95_ms_per_sample": cuda_stats.get(
                        "p95"
                    ),
                    "wall_mean_ms_per_sample": wall_stats.get(
                        "mean"
                    ),
                    "wall_median_ms_per_sample": wall_stats.get(
                        "median"
                    ),
                    "wall_p95_ms_per_sample": wall_stats.get(
                        "p95"
                    ),
                    "peak_allocated_mb": result.get(
                        "peak_allocated_mb"
                    ),
                    "incremental_peak_allocated_mb": result.get(
                        "incremental_peak_allocated_mb"
                    ),
                    "peak_reserved_mb": result.get(
                        "peak_reserved_mb"
                    ),
                    "pixel_ratio_vs_base": scaling.get(
                        "pixel_ratio"
                    ),
                    "cuda_latency_ratio_vs_base": scaling.get(
                        "cuda_latency_ratio"
                    ),
                    "wall_latency_ratio_vs_base": scaling.get(
                        "wall_latency_ratio"
                    ),
                    "memory_ratio_vs_base": scaling.get(
                        "incremental_memory_ratio"
                    ),
                    "error": result.get("error"),
                }
            )


def write_log(
    metadata: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "RadioHRFormer Resolution Runtime Benchmark "
            "(profiler-free)\n"
        )
        f.write("=" * 88 + "\n")
        for key in [
            "timestamp",
            "model_source_file",
            "config_path",
            "weight_path",
            "device",
            "gpu_name",
            "precision",
            "batch_size",
            "resolutions",
            "warmup_iterations",
            "timed_iterations",
            "cudnn_benchmark",
        ]:
            f.write(f"{key}: {metadata[key]}\n")

        f.write("\nModel Profile\n")
        f.write("-" * 88 + "\n")
        profile = metadata["model_profile"]
        f.write(
            f"total_params: {profile['total_params']:,}\n"
        )
        f.write(
            f"trainable_params: "
            f"{profile['trainable_params']:,}\n"
        )
        f.write(
            f"model_size_param_buffer_mb: "
            f"{profile['model_size_param_buffer_mb']:.2f}\n"
        )

        f.write("\nPer-resolution Results\n")
        f.write("-" * 88 + "\n")

        for result in results:
            f.write(
                f"\n[{result['resolution']} x "
                f"{result['resolution']}]\n"
            )
            f.write(f"status: {result['status']}\n")

            if result["status"] != "ok":
                f.write(f"error: {result.get('error')}\n")
                continue

            f.write(
                f"input_shape:  {result['input_shape']}\n"
            )
            f.write(
                f"output_shape: {result['output_shape']}\n"
            )

            cuda = result[
                "cuda_event_ms_per_sample"
            ]
            wall = result[
                "wall_clock_ms_per_sample"
            ]

            f.write("\nLatency\n")
            f.write(
                f"  CUDA Event mean:   "
                f"{cuda['mean']:.6f} ms/sample\n"
            )
            f.write(
                f"  CUDA Event median: "
                f"{cuda['median']:.6f} ms/sample\n"
            )
            f.write(
                f"  CUDA Event p95:    "
                f"{cuda['p95']:.6f} ms/sample\n"
            )
            f.write(
                f"  Wall clock mean:   "
                f"{wall['mean']:.6f} ms/sample\n"
            )
            f.write(
                f"  Wall clock median: "
                f"{wall['median']:.6f} ms/sample\n"
            )
            f.write(
                f"  Wall clock p95:    "
                f"{wall['p95']:.6f} ms/sample\n"
            )
            f.write(
                f"  Wall/CUDA ratio:   "
                f"{result['wall_cuda_mean_ratio']:.4f}x\n"
            )

            f.write("\nCUDA memory\n")
            f.write(
                f"  peak_allocated:             "
                f"{result['peak_allocated_mb']:.2f} MB\n"
            )
            f.write(
                f"  incremental_peak_allocated: "
                f"{result['incremental_peak_allocated_mb']:.2f} MB\n"
            )
            f.write(
                f"  peak_reserved:              "
                f"{result['peak_reserved_mb']:.2f} MB\n"
            )

            scaling = result.get(
                "scaling_vs_first_resolution",
                {},
            )
            if scaling:
                f.write("\nScaling vs first resolution\n")
                f.write(
                    f"  pixel ratio:        "
                    f"{scaling['pixel_ratio']:.4f}x\n"
                )
                f.write(
                    f"  CUDA latency ratio: "
                    f"{scaling['cuda_latency_ratio']:.4f}x\n"
                )
                f.write(
                    f"  wall latency ratio: "
                    f"{scaling['wall_latency_ratio']:.4f}x\n"
                )
                f.write(
                    f"  memory ratio:       "
                    f"{scaling['incremental_memory_ratio']:.4f}x\n"
                )

        f.write("\nProtocol\n")
        f.write("-" * 88 + "\n")
        f.write(
            "- torch.profiler is neither imported nor invoked "
            "in this process.\n"
        )
        f.write(
            "- Synthetic input construction and CPU-to-GPU "
            "transfer are excluded from latency.\n"
        )
        f.write(
            "- Warm-up iterations are excluded from latency.\n"
        )
        f.write(
            "- Paper-facing GPU latency is the CUDA Event "
            "mean/median above.\n"
        )
        f.write(
            "- Wall-clock timing is a synchronized cross-check.\n"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not args.resolutions:
        raise ValueError(
            "At least one resolution must be specified."
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "This paper-facing benchmark requires CUDA."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is unavailable."
        )

    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = (
        not args.disable_cudnn_benchmark
    )
    torch.backends.cudnn.deterministic = False

    cfg = load_config(args.config_path)

    print(f"Using device: {device}")
    print(f"Imported model source: {MODEL_SOURCE_FILE}")
    print(f"Loading weights: {args.weight_path}")
    print(
        "Profiler status: NOT imported / NOT used "
        "(runtime-only process)"
    )

    model = build_model(
        cfg=cfg,
        weight_path=args.weight_path,
        device=device,
    )
    model_profile = get_model_profile(
        model,
        args.weight_path,
    )
    in_channels = int(
        cfg["model"]["in_channels"]
    )

    results: List[Dict[str, Any]] = []

    for index, resolution in enumerate(
        args.resolutions
    ):
        if (
            index > 0
            and args.cooldown_seconds > 0
        ):
            print(
                f"\nCooling down for "
                f"{args.cooldown_seconds:.1f} s..."
            )
            synchronize(device)
            time.sleep(args.cooldown_seconds)

        print(
            f"\nBenchmarking {resolution} x {resolution} "
            f"(batch={args.batch_size}, "
            f"precision={args.precision})"
        )

        result = benchmark_resolution(
            model=model,
            resolution=int(resolution),
            in_channels=in_channels,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
            precision=args.precision,
            seed=args.seed,
            obstacle_density=args.obstacle_density,
        )
        results.append(result)

        if result["status"] == "ok":
            cuda = result[
                "cuda_event_ms_per_sample"
            ]
            wall = result[
                "wall_clock_ms_per_sample"
            ]
            print(
                f"  input shape:  "
                f"{result['input_shape']}"
            )
            print(
                f"  output shape: "
                f"{result['output_shape']}"
            )
            print(
                f"  CUDA latency: "
                f"{cuda['mean']:.6f} ms/sample"
            )
            print(
                f"  wall latency: "
                f"{wall['mean']:.6f} ms/sample"
            )
            print(
                f"  peak allocated: "
                f"{result['peak_allocated_mb']:.2f} MB"
            )
            print(
                f"  incremental peak: "
                f"{result['incremental_peak_allocated_mb']:.2f} MB"
            )
        else:
            print(
                f"  status: {result['status']} | "
                f"{result.get('error')}"
            )

    add_scaling(results)

    print("\nScaling summary vs first resolution")
    for result in results:
        if result.get("status") != "ok":
            continue
        scaling = result[
            "scaling_vs_first_resolution"
        ]
        print(
            f"  {result['resolution']:4d}: "
            f"pixels={scaling['pixel_ratio']:.2f}x | "
            f"CUDA={scaling['cuda_latency_ratio']:.3f}x | "
            f"wall={scaling['wall_latency_ratio']:.3f}x | "
            f"memory={scaling['incremental_memory_ratio']:.3f}x"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    props = torch.cuda.get_device_properties(
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )

    metadata = {
        "benchmark": "RadioHRFormer runtime-only",
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
        "gpu_name": props.name,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "resolutions": [
            int(value)
            for value in args.resolutions
        ],
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repeats,
        "cooldown_seconds": args.cooldown_seconds,
        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_profile": model_profile,
    }

    json_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_runtime.json",
    )
    csv_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_runtime.csv",
    )
    log_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_runtime.txt",
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

    write_csv(results, csv_path)
    write_log(metadata, results, log_path)

    print("\nRuntime benchmark complete.")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print(f"LOG : {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profiler-free RadioHRFormer resolution benchmark "
            "for paper-facing latency and CUDA memory."
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
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.0,
        help=(
            "Idle time between resolutions to reduce thermal/clock "
            "carry-over. Default: 2.0 s. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "./runs/"
            "radiohrformer_resolution_runtime"
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
    parser.add_argument(
        "--disable-cudnn-benchmark",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
