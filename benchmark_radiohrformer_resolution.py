# @Description:
# Resolution-scaling benchmark for RadioHRFormer.
# Measures pure model forward latency and CUDA memory usage with synthetic
# square inputs at 256, 512, 768, and 1024 resolutions.
#
# The measurement protocol intentionally matches
# benchmark_radiomamba_resolution.py:
#   - batch size 1 by default
#   - FP32 by default
#   - 20 warm-up iterations per resolution
#   - 100 timed iterations per resolution
#   - CUDA Event latency measurement
#   - synthetic input creation and host-to-device transfer excluded
#   - PyTorch CUDA peak allocated/reserved memory reported
#
# Run this file from the RadioHRFormer project so that local imports work.

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

# Resolve the RadioHRFormer project root before importing local modules.
# insert(0, ...) is intentional: append(...) can silently import another
# installed package named "models" before the project-local implementation.
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
    if config_path is None:
        raise ValueError("--config-path must be provided.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = ["seed", "data", "model", "train"]
    missing_keys = [
        key for key in required_keys
        if key not in cfg
    ]
    if missing_keys:
        raise KeyError(
            f"Missing config keys: {missing_keys}. "
            f"Loaded path: {config_path}. "
            f"Top-level keys: {list(cfg.keys())}"
        )

    return cfg


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if any(
        key.startswith("module.")
        for key in state_dict
    ):
        return {
            (
                key[7:]
                if key.startswith("module.")
                else key
            ): value
            for key, value in state_dict.items()
        }
    return state_dict


def load_model_state(
    model: torch.nn.Module,
    weight_path: str,
    device: torch.device,
) -> None:
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Weight file not found: {weight_path}"
        )

    checkpoint = torch.load(
        weight_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "The loaded checkpoint does not contain "
            "a valid state_dict."
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
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    weight_file_size_mb = None
    if (
        weight_path is not None
        and os.path.exists(weight_path)
    ):
        weight_file_size_mb = bytes_to_mb(
            os.path.getsize(weight_path)
        )

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_size_mb": bytes_to_mb(parameter_bytes),
        "buffer_size_mb": bytes_to_mb(buffer_bytes),
        "model_size_param_buffer_mb": bytes_to_mb(
            parameter_bytes + buffer_bytes
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


def get_autocast_context(
    device: torch.device,
    precision: str,
):
    if precision == "fp32":
        return nullcontext()

    if device.type != "cuda":
        raise ValueError(
            f"{precision} benchmarking is supported "
            "only on CUDA."
        )

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


def percentile(
    values: Sequence[float],
    quantile: float,
) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(
        float(value) for value in values
    )

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        len(sorted_values) - 1
    ) * quantile
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )
    fraction = position - lower_index

    return (
        sorted_values[lower_index]
        * (1.0 - fraction)
        + sorted_values[upper_index]
        * fraction
    )


def make_synthetic_input(
    batch_size: int,
    in_channels: int,
    resolution: int,
    device: torch.device,
    seed: int,
    obstacle_density: float,
) -> torch.Tensor:
    """
    Create a radio-map-like synthetic input [B, C, H, W].

    This follows the same input-generation protocol as the RadioMamba
    resolution benchmark:
      - channels 0..C-2: sparse binary environmental masks
      - last channel: one-hot transmitter-location map

    Input construction and CPU-to-GPU transfer happen before timing and
    are therefore excluded from the reported forward latency.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )
    if in_channels <= 0:
        raise ValueError(
            "in_channels must be positive."
        )
    if resolution <= 0:
        raise ValueError(
            "resolution must be positive."
        )
    if not 0.0 <= obstacle_density <= 1.0:
        raise ValueError(
            "obstacle_density must be between "
            "0 and 1."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    input_cpu = torch.zeros(
        (
            batch_size,
            in_channels,
            resolution,
            resolution,
        ),
        dtype=torch.float32,
    )

    if in_channels == 1:
        input_cpu.uniform_(
            0.0,
            1.0,
            generator=generator,
        )
    else:
        obstacle_random = torch.rand(
            (
                batch_size,
                in_channels - 1,
                resolution,
                resolution,
            ),
            generator=generator,
        )
        input_cpu[:, :-1] = (
            obstacle_random < obstacle_density
        ).to(dtype=torch.float32)

        transmitter_rows = torch.randint(
            low=0,
            high=resolution,
            size=(batch_size,),
            generator=generator,
        )
        transmitter_columns = torch.randint(
            low=0,
            high=resolution,
            size=(batch_size,),
            generator=generator,
        )
        batch_indices = torch.arange(batch_size)

        input_cpu[
            batch_indices,
            in_channels - 1,
            transmitter_rows,
            transmitter_columns,
        ] = 1.0

    return input_cpu.to(
        device=device,
        non_blocking=False,
    ).contiguous()


def measure_cuda_times(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    repeats: int,
    precision: str,
) -> List[float]:
    """
    Measure isolated forward latency.

    Each iteration is synchronized separately. This is more conservative than
    queueing every forward before one final synchronization and makes it easier
    to detect accidental asynchronous execution on another CUDA stream.
    CUDA Event elapsed time itself does not include the CPU synchronization cost.
    """
    times_ms: List[float] = []

    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with get_autocast_context(device, precision):
            output = model(inputs)
        end_event.record()

        end_event.synchronize()
        times_ms.append(
            float(start_event.elapsed_time(end_event))
        )
        del output

    return times_ms


def measure_cpu_times(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    repeats: int,
    precision: str,
) -> List[float]:
    times_ms: List[float] = []

    for _ in range(repeats):
        start_time = time.perf_counter()

        with get_autocast_context(
            device,
            precision,
        ):
            output = model(inputs)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        times_ms.append(float(elapsed_ms))
        del output

    return times_ms


def benchmark_resolution(
    model: torch.nn.Module,
    resolution: int,
    in_channels: int,
    batch_size: int,
    warmup_iterations: int,
    repeats: int,
    precision: str,
    device: torch.device,
    seed: int,
    obstacle_density: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "resolution": int(resolution),
        "height": int(resolution),
        "width": int(resolution),
        "batch_size": int(batch_size),
        "in_channels": int(in_channels),
        "input_shape": [
            int(batch_size),
            int(in_channels),
            int(resolution),
            int(resolution),
        ],
        "warmup_iterations": int(
            warmup_iterations
        ),
        "timed_iterations": int(repeats),
        "precision": precision,
        "status": "pending",
        "error": None,
    }

    inputs = None
    output = None

    try:
        clear_cuda_cache(device)

        inputs = make_synthetic_input(
            batch_size=batch_size,
            in_channels=in_channels,
            resolution=resolution,
            device=device,
            seed=seed + resolution,
            obstacle_density=obstacle_density,
        )

        result["actual_input_shape"] = [
            int(value) for value in inputs.shape
        ]
        result["actual_input_numel"] = int(inputs.numel())

        # Diagnostic forward: verify that the model really processes and returns
        # the requested spatial resolution. Without this check, a model that
        # internally resizes every input to 256x256 can produce misleadingly
        # constant latency.
        with torch.inference_mode():
            with get_autocast_context(device, precision):
                diagnostic_output = model(inputs)
        synchronize(device)

        if not isinstance(diagnostic_output, torch.Tensor):
            raise TypeError(
                "HRFormer benchmark expected a Tensor output, but received "
                f"{type(diagnostic_output).__name__}."
            )

        result["output_shape"] = [
            int(value) for value in diagnostic_output.shape
        ]

        if diagnostic_output.ndim < 4:
            raise RuntimeError(
                "Expected a 4-D dense prediction tensor, but received "
                f"shape={tuple(diagnostic_output.shape)}."
            )

        output_hw = tuple(
            int(value) for value in diagnostic_output.shape[-2:]
        )
        expected_hw = (int(resolution), int(resolution))
        if output_hw != expected_hw:
            raise RuntimeError(
                "The active model did not preserve the requested resolution: "
                f"input={tuple(inputs.shape)}, output={tuple(diagnostic_output.shape)}, "
                f"expected output spatial size={expected_hw}. "
                "Check for an internal resize/crop or an unintended imported model."
            )

        del diagnostic_output

        # Warm-up absorbs one-time CUDA kernel initialization and
        # resolution-specific cuDNN autotuning. Warm-up latency is excluded.
        with torch.inference_mode():
            for _ in range(warmup_iterations):
                with get_autocast_context(
                    device,
                    precision,
                ):
                    output = model(inputs)

                del output
                output = None

        synchronize(device)

        if device.type == "cuda":
            allocated_before = (
                torch.cuda.memory_allocated(device)
            )
            reserved_before = (
                torch.cuda.memory_reserved(device)
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )
            synchronize(device)
        else:
            allocated_before = 0
            reserved_before = 0

        with torch.inference_mode():
            if device.type == "cuda":
                times_ms = measure_cuda_times(
                    model=model,
                    inputs=inputs,
                    device=device,
                    repeats=repeats,
                    precision=precision,
                )
            else:
                times_ms = measure_cpu_times(
                    model=model,
                    inputs=inputs,
                    device=device,
                    repeats=repeats,
                    precision=precision,
                )

        synchronize(device)

        if device.type == "cuda":
            allocated_after = (
                torch.cuda.memory_allocated(device)
            )
            reserved_after = (
                torch.cuda.memory_reserved(device)
            )
            peak_allocated = (
                torch.cuda.max_memory_allocated(
                    device
                )
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved(
                    device
                )
            )
        else:
            allocated_after = 0
            reserved_after = 0
            peak_allocated = 0
            peak_reserved = 0

        mean_ms_per_batch = statistics.fmean(
            times_ms
        )
        median_ms_per_batch = statistics.median(
            times_ms
        )
        std_ms_per_batch = (
            statistics.pstdev(times_ms)
            if len(times_ms) > 1
            else 0.0
        )

        result.update(
            {
                "status": "ok",
                "mean_ms_per_batch": (
                    mean_ms_per_batch
                ),
                "median_ms_per_batch": (
                    median_ms_per_batch
                ),
                "std_ms_per_batch": (
                    std_ms_per_batch
                ),
                "min_ms_per_batch": min(
                    times_ms
                ),
                "max_ms_per_batch": max(
                    times_ms
                ),
                "p95_ms_per_batch": percentile(
                    times_ms,
                    0.95,
                ),
                "mean_ms_per_sample": (
                    mean_ms_per_batch
                    / batch_size
                ),
                "throughput_samples_per_sec": (
                    batch_size
                    * 1000.0
                    / mean_ms_per_batch
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
                    max(
                        0,
                        peak_allocated
                        - allocated_before,
                    )
                ),
                "incremental_peak_reserved_mb": bytes_to_mb(
                    max(
                        0,
                        peak_reserved
                        - reserved_before,
                    )
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

    except RuntimeError as exc:
        error_message = str(exc)

        if (
            "out of memory"
            in error_message.lower()
        ):
            result.update(
                {
                    "status": "oom",
                    "error": error_message,
                }
            )
        else:
            result.update(
                {
                    "status": "error",
                    "error": error_message,
                }
            )

    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

    finally:
        if output is not None:
            del output
        if inputs is not None:
            del inputs

        clear_cuda_cache(device)

    return result


def write_csv(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    fieldnames = [
        "resolution",
        "height",
        "width",
        "batch_size",
        "in_channels",
        "input_shape",
        "actual_input_shape",
        "actual_input_numel",
        "output_shape",
        "precision",
        "warmup_iterations",
        "timed_iterations",
        "status",
        "mean_ms_per_sample",
        "mean_ms_per_batch",
        "median_ms_per_batch",
        "std_ms_per_batch",
        "min_ms_per_batch",
        "max_ms_per_batch",
        "p95_ms_per_batch",
        "throughput_samples_per_sec",
        "allocated_before_mb",
        "reserved_before_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "incremental_peak_allocated_mb",
        "incremental_peak_reserved_mb",
        "allocated_after_mb",
        "reserved_after_mb",
        "error",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for result in results:
            writer.writerow(result)


def write_log(
    metadata: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "RadioHRFormer Resolution Benchmark\n"
        )
        f.write("=" * 80 + "\n")
        f.write(
            f"timestamp: "
            f"{metadata['timestamp']}\n"
        )
        f.write(
            f"model_source_file: "
            f"{metadata['model_source_file']}\n"
        )
        f.write(
            f"config_path: "
            f"{metadata['config_path']}\n"
        )
        f.write(
            f"weight_path: "
            f"{metadata['weight_path']}\n"
        )
        f.write(
            f"device: {metadata['device']}\n"
        )
        f.write(
            f"gpu_name: "
            f"{metadata.get('gpu_name')}\n"
        )
        f.write(
            f"precision: "
            f"{metadata['precision']}\n"
        )
        f.write(
            f"batch_size: "
            f"{metadata['batch_size']}\n"
        )
        f.write(
            f"resolutions: "
            f"{metadata['resolutions']}\n"
        )
        f.write(
            "warmup_iterations: "
            f"{metadata['warmup_iterations']}\n"
        )
        f.write(
            "timed_iterations: "
            f"{metadata['timed_iterations']}\n"
        )
        f.write(
            "cudnn_benchmark: "
            f"{metadata['cudnn_benchmark']}\n"
        )

        f.write("\nModel Profile\n")
        f.write("-" * 80 + "\n")
        model_profile = metadata[
            "model_profile"
        ]
        f.write(
            "total_params: "
            f"{model_profile['total_params']:,}\n"
        )
        f.write(
            "trainable_params: "
            f"{model_profile['trainable_params']:,}\n"
        )
        f.write(
            "param_size_mb: "
            f"{model_profile['param_size_mb']:.2f}\n"
        )
        f.write(
            "buffer_size_mb: "
            f"{model_profile['buffer_size_mb']:.2f}\n"
        )
        f.write(
            "model_size_param_buffer_mb: "
            f"{model_profile['model_size_param_buffer_mb']:.2f}\n"
        )

        if (
            model_profile[
                "weight_file_size_mb"
            ]
            is not None
        ):
            f.write(
                "weight_file_size_mb: "
                f"{model_profile['weight_file_size_mb']:.2f}\n"
            )

        f.write("\nPer-resolution Results\n")
        f.write("-" * 80 + "\n")

        for result in results:
            resolution = result["resolution"]
            f.write(
                f"\n[{resolution} x "
                f"{resolution}]\n"
            )
            f.write(
                f"status: "
                f"{result['status']}\n"
            )

            if result["status"] != "ok":
                f.write(
                    f"error: "
                    f"{result['error']}\n"
                )
                continue

            f.write(
                f"actual_input_shape: {result['actual_input_shape']}\n"
            )
            f.write(
                f"output_shape: {result['output_shape']}\n"
            )
            f.write(
                "mean_inference_time: "
                f"{result['mean_ms_per_sample']:.6f} "
                "ms/sample\n"
            )
            f.write(
                "median_inference_time: "
                f"{result['median_ms_per_batch']:.6f} "
                "ms/batch\n"
            )
            f.write(
                "std_inference_time: "
                f"{result['std_ms_per_batch']:.6f} "
                "ms/batch\n"
            )
            f.write(
                "p95_inference_time: "
                f"{result['p95_ms_per_batch']:.6f} "
                "ms/batch\n"
            )
            f.write(
                "throughput: "
                f"{result['throughput_samples_per_sec']:.6f} "
                "samples/sec\n"
            )
            f.write(
                "peak_allocated_memory: "
                f"{result['peak_allocated_mb']:.2f} "
                "MB\n"
            )
            f.write(
                "incremental_peak_allocated_memory: "
                f"{result['incremental_peak_allocated_mb']:.2f} "
                "MB\n"
            )
            f.write(
                "peak_reserved_memory: "
                f"{result['peak_reserved_mb']:.2f} "
                "MB\n"
            )

        f.write("\nMeasurement Protocol\n")
        f.write("-" * 80 + "\n")
        f.write(
            "- Input tensor shape is "
            "[batch, channels, H, W].\n"
        )
        f.write(
            "- Synthetic input creation and CPU-to-GPU "
            "transfer are excluded from latency.\n"
        )
        f.write(
            "- Warm-up iterations are excluded "
            "from latency.\n"
        )
        f.write(
            "- CUDA latency is measured with "
            "CUDA events.\n"
        )
        f.write(
            "- peak_allocated_mb includes model parameters, "
            "the input tensor, output tensor, and temporary "
            "forward allocations tracked by PyTorch's CUDA "
            "allocator.\n"
        )
        f.write(
            "- incremental_peak_allocated_mb is peak allocated "
            "memory minus allocated memory immediately before "
            "timed inference.\n"
        )
        f.write(
            "- PyTorch allocator memory is not identical to "
            "the full process memory reported by nvidia-smi.\n"
        )


def build_model(
    cfg: Dict[str, Any],
    weight_path: str,
    device: torch.device,
) -> torch.nn.Module:
    model = HRFormerRadioMapRegressor(
        cfg
    ).to(device)

    load_model_state(
        model=model,
        weight_path=weight_path,
        device=device,
    )

    model.eval()
    return model


def run_benchmark(
    args: argparse.Namespace,
) -> None:
    if args.warmup < 0:
        raise ValueError(
            "--warmup must be non-negative."
        )
    if args.repeats <= 0:
        raise ValueError(
            "--repeats must be positive."
        )
    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )
    if not args.resolutions:
        raise ValueError(
            "At least one resolution must "
            "be specified."
        )

    device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA device was requested, "
            "but CUDA is unavailable."
        )

    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            args.seed
        )

    torch.backends.cudnn.benchmark = (
        not args.disable_cudnn_benchmark
    )
    torch.backends.cudnn.deterministic = False

    cfg = load_config(args.config_path)

    print(f"Using device: {device}")
    print(f"Imported model source: {MODEL_SOURCE_FILE}")
    print(
        f"Loading weights: "
        f"{args.weight_path}"
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

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    results: List[Dict[str, Any]] = []

    for resolution in args.resolutions:
        print(
            f"\nBenchmarking "
            f"{resolution} x {resolution} "
            f"(batch={args.batch_size}, "
            f"precision={args.precision})"
        )

        result = benchmark_resolution(
            model=model,
            resolution=int(resolution),
            in_channels=in_channels,
            batch_size=args.batch_size,
            warmup_iterations=args.warmup,
            repeats=args.repeats,
            precision=args.precision,
            device=device,
            seed=args.seed,
            obstacle_density=(
                args.obstacle_density
            ),
        )
        results.append(result)

        if result["status"] == "ok":
            print(
                f"  input shape: {result['actual_input_shape']}"
            )
            print(
                f"  output shape: {result['output_shape']}"
            )
            print(
                "  latency: "
                f"{result['mean_ms_per_sample']:.6f} "
                "ms/sample"
            )
            print(
                "  peak allocated: "
                f"{result['peak_allocated_mb']:.2f} "
                "MB"
            )
            print(
                "  incremental peak: "
                f"{result['incremental_peak_allocated_mb']:.2f} "
                "MB"
            )
        else:
            print(
                f"  status: "
                f"{result['status']} | "
                f"{result['error']}"
            )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    gpu_name = None
    gpu_total_memory_mb = None

    if device.type == "cuda":
        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        properties = (
            torch.cuda.get_device_properties(
                device_index
            )
        )
        gpu_name = properties.name
        gpu_total_memory_mb = bytes_to_mb(
            properties.total_memory
        )

    metadata: Dict[str, Any] = {
        "model": "HRFormerRadioMapRegressor",
        "model_source_file": MODEL_SOURCE_FILE,
        "timestamp": timestamp,
        "config_path": os.path.abspath(
            args.config_path
        ),
        "weight_path": os.path.abspath(
            args.weight_path
        ),
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": (
            gpu_total_memory_mb
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "resolutions": [
            int(value)
            for value in args.resolutions
        ],
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repeats,
        "seed": args.seed,
        "obstacle_density": (
            args.obstacle_density
        ),
        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark
        ),
        "memory_definition": {
            "peak_allocated_mb": (
                "Maximum memory allocated by PyTorch during "
                "timed inference, including resident model, "
                "input, output, and temporary forward tensors."
            ),
            "incremental_peak_allocated_mb": (
                "peak_allocated_mb minus allocated memory "
                "immediately before timed inference."
            ),
        },
        "model_profile": model_profile,
    }

    json_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_benchmark.json",
    )
    csv_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_benchmark.csv",
    )
    log_path = os.path.join(
        args.output_dir,
        "radiohrformer_resolution_benchmark.txt",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "metadata": metadata,
                "results": results,
            },
            f,
            indent=2,
        )

    write_csv(
        results,
        csv_path,
    )
    write_log(
        metadata,
        results,
        log_path,
    )

    print("\nBenchmark complete.")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print(f"LOG : {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RadioHRFormer inference latency and CUDA "
            "memory usage at multiple square input resolutions."
        )
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/carsdpm_downstream.json",
        help=(
            "Path to the RadioHRFormer JSON "
            "configuration file."
        ),
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        required=True,
        help=(
            "Path to the trained downstream "
            "RadioHRFormer weight file."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=(
            "Single execution device, e.g. "
            "cuda:0 or cpu."
        ),
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help=(
            "Square input resolutions. Default: "
            "256 512 768 1024."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Synthetic inference batch size. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help=(
            "Warm-up forwards per resolution, "
            "excluded from timing. Default: 20."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help=(
            "Timed forwards per resolution. "
            "Default: 100."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=[
            "fp32",
            "fp16",
            "bf16",
        ],
        default="fp32",
        help=(
            "Inference precision. fp16/bf16 use "
            "CUDA autocast. Default: fp32, matching "
            "the RadioMamba resolution benchmark."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "./runs/"
            "radiohrformer_resolution_benchmark"
        ),
        help=(
            "Directory for JSON, CSV, "
            "and text results."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for synthetic inputs."
        ),
    )
    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.15,
        help=(
            "Fraction of active pixels in synthetic "
            "environmental mask channels. Default: 0.15."
        ),
    )
    parser.add_argument(
        "--disable-cudnn-benchmark",
        action="store_true",
        help=(
            "Disable cuDNN shape-specific "
            "algorithm autotuning."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
