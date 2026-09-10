import os
import sys
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchmetrics.functional import structural_similarity_index_measure

from datasets.rms_dataset import build_dataloaders
from models.hrformer_regressor import HRFormerRadioMapRegressor

from utils import (
    set_seed,
    get_amp_device_type,
    summarize_trainable_by_module,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HRFormer for RadioMapSeer radio map regression."
    )
    parser.add_argument("--config-path", type=str, default="./configs/hrt.json")
    parser.add_argument("--weight-path", type=str, required=True)
    parser.add_argument("--save-root", type=str, default="./save_eval")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "valid", "test"], default="test")

    # Output controls are run-specific rather than training configuration.
    parser.add_argument("--save-pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-npy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-gt-npy", action=argparse.BooleanOptionalAction, default=False)
    # Kept for CLI backward compatibility. Prediction/GT PNGs use the paper-style
    # RGB renderer below; scalar error maps keep their own fixed colormap.
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument(
        "--obstacle-threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert building/car input channels to binary masks for PNG overlay.",
    )
    parser.add_argument("--error-vmax", type=float, default=0.3)
    parser.add_argument("--max-save", type=int, default=100)
    return parser.parse_args()


def load_config(config_path):
    if config_path is None:
        raise ValueError("--config-path must be provided.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    required_keys = ["seed", "data", "model", "train"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(
            f"Missing config keys: {missing}. "
            f"Loaded path: {config_path}. "
            f"Top-level keys: {list(cfg.keys())}"
        )

    return cfg


def cfg_get(config, keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def prepare_eval_config(cfg, return_name=True):
    """Use the JSON configuration as the single source of data/runtime settings."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})

    if cfg["data"].get("root_dir") is None:
        raise ValueError("cfg['data']['root_dir'] must be set.")

    cfg["data"]["return_name"] = bool(return_name)
    return cfg


def resolve_options(args, cfg):
    return {
        "data_root": cfg_get(cfg, ["data", "root_dir"], None),
        "target_type": cfg_get(cfg, ["data", "target_type"], "DPM"),
        "split": "val" if args.split == "valid" else args.split,
        "num_tx": cfg_get(cfg, ["data", "num_tx"], 80),
        "thresh": cfg_get(cfg, ["data", "thresh"], 0.0),
        "batch_size": cfg_get(cfg, ["data", "batch_size"], 32),
        "num_workers": cfg_get(cfg, ["data", "num_workers"], 4),
        "seed": cfg_get(cfg, ["seed"], 42),
    }


def resolve_runtime(cfg):
    """Resolve physical CUDA indices declared in cfg['runtime']['gpus']."""
    requested = cfg_get(cfg, ["runtime", "gpus"], [0])
    if requested is None:
        requested = []
    if not isinstance(requested, (list, tuple)):
        raise TypeError("cfg['runtime']['gpus'] must be a list, e.g. [1, 2].")

    gpu_ids = [int(gpu_id) for gpu_id in requested]
    if len(gpu_ids) != len(set(gpu_ids)) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"Invalid GPU list: {gpu_ids}")

    if not gpu_ids or not torch.cuda.is_available():
        if gpu_ids and not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu"), [], False

    visible_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= visible_count]
    if invalid:
        raise ValueError(
            f"runtime.gpus={gpu_ids}, but CUDA exposes device indices 0..{visible_count - 1}. "
            "Do not set CUDA_VISIBLE_DEVICES in the script; either unset it in the shell or "
            "use indices relative to the visible devices."
        )

    primary_gpu = gpu_ids[0]
    torch.cuda.set_device(primary_gpu)
    device = torch.device(f"cuda:{primary_gpu}")
    use_amp = bool(cfg_get(cfg, ["runtime", "amp"], True))
    return device, gpu_ids, use_amp


def maybe_wrap_data_parallel(model, gpu_ids):
    if len(gpu_ids) <= 1:
        return model
    print(f"Using torch.nn.DataParallel on GPUs: {gpu_ids} (primary: cuda:{gpu_ids[0]}).")
    return torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def normalize_state_dict_keys(state_dict):
    if any(key.startswith("module.") for key in state_dict):
        return {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    return state_dict


def get_split_dict(cfg):
    try:
        return build_dataloaders(cfg, return_datasets=True)
    except TypeError as exc:
        raise TypeError(
            "rms_dataset.build_dataloaders must accept return_datasets=True. "
            "Update rms_dataset.py with the provided version."
        ) from exc


def select_split(split_dict, split_name):
    split_name = "val" if split_name == "valid" else split_name
    if split_name not in ["train", "val", "test"]:
        raise ValueError(f"Unsupported split: {split_name}")
    return split_dict[f"{split_name}_dataset"], split_dict[f"{split_name}_loader"]


def unpack_batch(batch, device):
    if len(batch) == 3:
        x, y, names = batch
    elif len(batch) == 2:
        x, y = batch
        names = None
    else:
        raise ValueError(f"Unexpected batch format with length {len(batch)}.")

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y, names


def load_model_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    unwrap_model(model).load_state_dict(normalize_state_dict_keys(state_dict), strict=True)


def tensor_to_image(x):
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    elif x.ndim == 3:
        x = x.permute(1, 2, 0)
    return x.numpy()


def _single_channel_numpy(img):
    """Convert [1,H,W] / [H,W] tensor-like input to a float32 [H,W] array."""
    arr = tensor_to_image(img).astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel map, got shape={arr.shape}.")
    return arr


def build_paper_rgb_map(
    field,
    input_map,
    target_type="DPM",
    obstacle_threshold=0.5,
):
    """Create the RadioMapSeer-style qualitative RGB visualization.

    Color convention follows the qualitative figures used by RadioUNet/RadioDiff:
      - radio/path-gain field: black -> yellow, with normalized field intensity
      - static buildings: blue
      - vehicles/cars: red

    The model output is *not* per-image min-max normalized. Values are only clipped
    to [0,1], preserving the common RadioMapSeer gray-level scale across samples.

    Expected input channel order in this project:
      cars_input=False: [building, building, Tx]
      cars_input=True : [building, cars, Tx]
    """
    field_arr = _single_channel_numpy(field)
    field_arr = np.nan_to_num(field_arr, nan=0.0, posinf=1.0, neginf=0.0)
    field_arr = np.clip(field_arr, 0.0, 1.0)

    input_arr = input_map.detach().float().cpu()
    if input_arr.ndim != 3:
        raise ValueError(f"Expected input_map=[C,H,W], got shape={tuple(input_arr.shape)}.")
    if input_arr.shape[0] < 1:
        raise ValueError("input_map must contain at least the building channel.")

    building = input_arr[0].numpy() > float(obstacle_threshold)
    cars = np.zeros_like(building, dtype=bool)
    if str(target_type).lower() == "carsdpm":
        if input_arr.shape[0] < 2:
            raise ValueError(
                "target_type=carsdpm but input_map does not contain channel 1 for cars."
            )
        cars = input_arr[1].numpy() > float(obstacle_threshold)

    if building.shape != field_arr.shape:
        raise ValueError(
            f"Geometry/radio-map shape mismatch: building={building.shape}, "
            f"field={field_arr.shape}."
        )

    # Black-to-yellow radio field: [v, v, 0].
    rgb = np.zeros((*field_arr.shape, 3), dtype=np.float32)
    rgb[..., 0] = field_arr
    rgb[..., 1] = field_arr

    # Overlay geometry with categorical colors. Cars are written last so that
    # a rare overlapping pixel remains visually identifiable as a dynamic obstacle.
    rgb[building] = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # blue
    rgb[cars] = np.array([1.0, 0.0, 0.0], dtype=np.float32)      # red
    return rgb


def save_paper_rgb_map(
    field,
    input_map,
    save_path,
    target_type="DPM",
    obstacle_threshold=0.5,
):
    """Save a borderless RGB PNG at the native map resolution."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rgb = build_paper_rgb_map(
        field=field,
        input_map=input_map,
        target_type=target_type,
        obstacle_threshold=obstacle_threshold,
    )
    # imsave writes the HxW RGB array directly, so a 256x256 map remains 256x256.
    plt.imsave(save_path, rgb, vmin=0.0, vmax=1.0)


def save_map_image(img, save_path, cmap="jet", vmin=None, vmax=None, title=None):
    """Generic scalar-map saver retained for error-map visualization."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    arr = tensor_to_image(img)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_npy_map(img, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    arr = tensor_to_image(img).astype(np.float32)
    np.save(save_path, arr)


def save_error_image(pred, gt, save_path, error_vmax):
    error = torch.abs(pred - gt)
    save_map_image(img=error, save_path=save_path, cmap="hot", vmin=0.0, vmax=error_vmax)


def bytes_to_mb(num_bytes):
    return float(num_bytes) / (1024.0 ** 2)


def get_model_profile(model, weight_path=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_bytes = param_bytes + buffer_bytes

    weight_file_mb = None
    if weight_path is not None and os.path.exists(weight_path):
        weight_file_mb = bytes_to_mb(os.path.getsize(weight_path))

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_size_mb": bytes_to_mb(param_bytes),
        "buffer_size_mb": bytes_to_mb(buffer_bytes),
        "model_size_mb": bytes_to_mb(model_bytes),
        "weight_file_size_mb": weight_file_mb,
    }


def reset_cuda_peak_memory(device, gpu_ids):
    if torch.device(device).type != "cuda":
        return
    for gpu_id in gpu_ids:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gpu_id)
        torch.cuda.synchronize(gpu_id)


def get_memory_profile(device, gpu_ids):
    if torch.device(device).type != "cuda":
        return {
            "device_type": torch.device(device).type,
            "per_gpu": {},
        }

    per_gpu = {}
    for gpu_id in gpu_ids:
        torch.cuda.synchronize(gpu_id)
        per_gpu[gpu_id] = {
            "allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(gpu_id)),
            "reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(gpu_id)),
            "peak_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(gpu_id)),
            "peak_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(gpu_id)),
        }
    return {
        "device_type": "cuda",
        "per_gpu": per_gpu,
    }


def sync_if_cuda(device, gpu_ids):
    if torch.device(device).type == "cuda":
        for gpu_id in gpu_ids:
            torch.cuda.synchronize(gpu_id)


def compute_regression_metrics(pred, target, data_range=1.0, eps=1e-12):
    """Compute radio-map regression and image-quality metrics.

    Metrics:
        MAE  = mean absolute error
        MSE  = mean squared error
        RMSE = sqrt(MSE)
        NMSE = ||pred-target||_2^2 / ||target||_2^2
        PSNR = 10 * log10(data_range^2 / MSE)
        SSIM = structural similarity index
    """
    pred = pred.detach().float()
    target = target.detach().float()

    if pred.ndim == 2:
        pred = pred[None, None, ...]
    elif pred.ndim == 3:
        pred = pred[None, ...]

    if target.ndim == 2:
        target = target[None, None, ...]
    elif target.ndim == 3:
        target = target[None, ...]

    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match: "
            f"{tuple(pred.shape)} vs {tuple(target.shape)}"
        )

    error = pred - target

    mae = error.abs().mean()
    mse = error.square().mean()
    rmse = torch.sqrt(mse)

    target_energy = target.square().mean()
    nmse = mse / torch.clamp(target_energy, min=eps)

    if mse <= eps:
        psnr = torch.tensor(
            float("inf"),
            device=pred.device,
            dtype=pred.dtype,
        )
    else:
        psnr = 10.0 * torch.log10(
            torch.tensor(
                data_range ** 2,
                device=pred.device,
                dtype=pred.dtype,
            ) / mse
        )

    ssim = structural_similarity_index_measure(
        pred,
        target,
        data_range=data_range,
    )

    return {
        "MAE": mae.item(),
        "MSE": mse.item(),
        "RMSE": rmse.item(),
        "NMSE": nmse.item(),
        "PSNR": psnr.item(),
        "SSIM": ssim.item(),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    gpu_ids,
    args,
    save_folder,
    use_amp=False,
    target_type="DPM",
):
    model.eval()

    pred_dir = os.path.join(save_folder, "pred_png")
    pred_npy_dir = os.path.join(save_folder, "pred_npy")
    gt_dir = os.path.join(save_folder, "gt_png")
    gt_npy_dir = os.path.join(save_folder, "gt_npy")
    err_dir = os.path.join(save_folder, "error_png")

    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)
    if args.save_npy:
        os.makedirs(pred_npy_dir, exist_ok=True)
    if args.save_gt:
        os.makedirs(gt_dir, exist_ok=True)
    if args.save_gt_npy:
        os.makedirs(gt_npy_dir, exist_ok=True)
    if args.save_error:
        os.makedirs(err_dir, exist_ok=True)

    global_idx = 0
    amp_device_type = get_amp_device_type(device)
    total_forward_time_sec = 0.0
    total_infer_samples = 0
    metric_sums = defaultdict(float)
    metric_count = 0

    reset_cuda_peak_memory(device, gpu_ids)

    for batch in tqdm(loader):
        x, y, names = unpack_batch(batch, device)

        # Measure only model forward time. This excludes data loading, CPU transfer,
        # metric computation, and PNG/NPY saving overhead.
        sync_if_cuda(device, gpu_ids)
        start_time = time.perf_counter()
        with autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(x)
        sync_if_cuda(device, gpu_ids)
        elapsed = time.perf_counter() - start_time

        batch_size = x.size(0)
        total_forward_time_sec += elapsed
        total_infer_samples += batch_size

        for b in range(pred.size(0)):
            sample_metrics = compute_regression_metrics(
                pred[b],
                y[b],
                data_range=1.0,
            )
            for k, v in sample_metrics.items():
                metric_sums[k] += float(v)
            metric_count += 1

        for b in range(x.size(0)):
            if names is not None:
                name = os.path.splitext(str(names[b]))[0]
            else:
                name = f"{global_idx:06d}"

            input_b = x[b].detach().cpu()
            pred_b = pred[b].detach().cpu()
            gt_b = y[b].detach().cpu()
            sample_mae = torch.mean(torch.abs(pred_b - gt_b)).item()

            should_save_png = args.max_save < 0 or global_idx < args.max_save
            if args.save_npy:
                save_npy_map(pred_b, os.path.join(pred_npy_dir, f"{name}.npy"))
            if args.save_gt_npy:
                save_npy_map(gt_b, os.path.join(gt_npy_dir, f"{name}.npy"))

            if should_save_png:
                if args.save_pred:
                    save_paper_rgb_map(
                        field=pred_b,
                        input_map=input_b,
                        save_path=os.path.join(
                            pred_dir, f"{name}_mae_{sample_mae:.4f}.png"
                        ),
                        target_type=target_type,
                        obstacle_threshold=args.obstacle_threshold,
                    )
                if args.save_gt:
                    save_paper_rgb_map(
                        field=gt_b,
                        input_map=input_b,
                        save_path=os.path.join(gt_dir, f"{name}_gt.png"),
                        target_type=target_type,
                        obstacle_threshold=args.obstacle_threshold,
                    )
                if args.save_error:
                    save_error_image(
                        pred=pred_b,
                        gt=gt_b,
                        save_path=os.path.join(err_dir, f"{name}_error.png"),
                        error_vmax=args.error_vmax,
                    )

            global_idx += 1

    avg_metrics = {
        k: float(v / max(1, metric_count))
        for k, v in metric_sums.items()
    }
    avg_inference_time_sec = (
        total_forward_time_sec / total_infer_samples
        if total_infer_samples > 0
        else float("nan")
    )
    memory_profile = get_memory_profile(device, gpu_ids)

    return {
        "MAE": avg_metrics.get("MAE", float("nan")),
        "MSE": avg_metrics.get("MSE", float("nan")),
        "RMSE": avg_metrics.get("RMSE", float("nan")),
        "NMSE": avg_metrics.get("NMSE", float("nan")),
        "PSNR": avg_metrics.get("PSNR", float("nan")),
        "SSIM": avg_metrics.get("SSIM", float("nan")),
        "metric_protocol": "per_sample_float_prediction",
        "total_forward_time_sec": float(total_forward_time_sec),
        "avg_inference_time_sec_per_sample": float(avg_inference_time_sec),
        "avg_inference_time_ms_per_sample": float(avg_inference_time_sec * 1000.0),
        "num_inference_samples": int(total_infer_samples),
        **memory_profile,
    }


def main():
    args = parse_args()
    cfg = prepare_eval_config(load_config(args.config_path), return_name=True)
    opts = resolve_options(args, cfg)

    device, gpu_ids, use_amp = resolve_runtime(cfg)
    runtime_gpus = gpu_ids if gpu_ids else ["cpu"]
    set_seed(opts["seed"])

    split_dict = get_split_dict(cfg)
    dataset, loader = select_split(split_dict, opts["split"])

    print(f"EVAL_SIZE : {len(dataset)}")
    print(f"Split     : {opts['split']}")
    print(f"Target    : {opts['target_type']}")

    base_model = HRFormerRadioMapRegressor(cfg).to(device)
    load_model_state(base_model, args.weight_path, device)
    summarize_trainable_by_module(base_model)
    model_profile = get_model_profile(base_model, args.weight_path)
    model = maybe_wrap_data_parallel(base_model, gpu_ids)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.save_root, run_name)
    os.makedirs(save_folder, exist_ok=True)

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        gpu_ids=gpu_ids,
        args=args,
        save_folder=save_folder,
        use_amp=use_amp,
        target_type=opts["target_type"],
    )

    metrics_json_path = os.path.join(save_folder, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f_json:
        json.dump(metrics, f_json, indent=2)

    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "a") as f:
        f.write("model: HRFormerRadioMapRegressor\n")
        f.write("metric_protocol: per-sample evaluation on floating-point predictions\n")
        f.write(f"config_path: {args.config_path}\n")
        f.write(f"weight_path: {args.weight_path}\n")
        f.write(f"gpus: {runtime_gpus}\n")
        f.write(f"amp: {use_amp}\n")
        f.write(f"data_root: {opts['data_root']}\n")
        f.write(f"split: {opts['split']}\n")
        f.write(f"target_type: {opts['target_type']}\n")
        f.write(
            "png_style: black-to-yellow radio field + blue buildings + red cars\n"
        )
        f.write(f"obstacle_threshold: {args.obstacle_threshold}\n")
        f.write(f"num_tx: {opts['num_tx']}\n")
        f.write("\nModel Profile\n")
        f.write(f"total_params: {model_profile['total_params']:,}\n")
        f.write(f"trainable_params: {model_profile['trainable_params']:,}\n")
        f.write(f"param_size: {model_profile['param_size_mb']:.2f} MB\n")
        f.write(f"buffer_size: {model_profile['buffer_size_mb']:.2f} MB\n")
        f.write(f"model_size_param_buffer: {model_profile['model_size_mb']:.2f} MB\n")
        if model_profile["weight_file_size_mb"] is not None:
            f.write(f"weight_file_size: {model_profile['weight_file_size_mb']:.2f} MB\n")
        f.write("\nRuntime Profile\n")
        f.write(f"num_inference_samples: {metrics['num_inference_samples']}\n")
        f.write(f"total_forward_time: {metrics['total_forward_time_sec']:.6f} sec\n")
        f.write(
            f"avg_inference_time_per_sample: "
            f"{metrics['avg_inference_time_ms_per_sample']:.6f} ms/sample\n"
        )
        f.write(f"device_type: {metrics['device_type']}\n")
        for gpu_id, memory in metrics["per_gpu"].items():
            f.write(
                f"cuda:{gpu_id} allocated_after_eval: {memory['allocated_mb']:.2f} MB | "
                f"reserved_after_eval: {memory['reserved_mb']:.2f} MB | "
                f"peak_allocated: {memory['peak_allocated_mb']:.2f} MB | "
                f"peak_reserved: {memory['peak_reserved_mb']:.2f} MB\n"
            )
        f.write("\nResults\n")
        f.write(f"MAE : {metrics['MAE']:.6f}\n")
        f.write(f"MSE : {metrics['MSE']:.6f}\n")
        f.write(f"RMSE : {metrics['RMSE']:.6f}\n")
        f.write(f"NMSE : {metrics['NMSE']:.6f}\n")
        f.write(f"PSNR : {metrics['PSNR']:.4f} dB\n")
        f.write(f"SSIM : {metrics['SSIM']:.6f}\n")

    print("\nResults")
    print(f"MAE : {metrics['MAE']:.6f}")
    print(f"MSE : {metrics['MSE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"NMSE : {metrics['NMSE']:.6f}")
    print(f"PSNR : {metrics['PSNR']:.4f} dB")
    print(f"SSIM : {metrics['SSIM']:.6f}")
    print("\nModel Profile")
    print(f"Total params: {model_profile['total_params']:,}")
    print(f"Model size(param+buffer): {model_profile['model_size_mb']:.2f} MB")
    print("\nRuntime Profile")
    print(f"Avg inference time: {metrics['avg_inference_time_ms_per_sample']:.6f} ms/sample")
    for gpu_id, memory in metrics["per_gpu"].items():
        print(
            f"CUDA:{gpu_id} peak memory: "
            f"{memory['peak_allocated_mb']:.2f} MB allocated, "
            f"{memory['peak_reserved_mb']:.2f} MB reserved"
        )
    print(f"Saved to: {save_folder}")
    print(f"Saved metrics JSON to: {metrics_json_path}")


if __name__ == "__main__":
    main()
