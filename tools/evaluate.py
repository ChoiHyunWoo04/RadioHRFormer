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

from datasets.rms_dataset import build_dataloaders, build_irt4_dataloaders, normalize_target_type
from models.hrformer_regressor import HRFormerRadioMapRegressor
from metrics import (
    compute_ssim,
    get_obstacle_mask,
    update_spatial_metrics,
    summarize_spatial_metrics,
)

from utils import (
    set_seed,
    get_amp_device_type,
    summarize_trainable_by_module,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HRFormer for RadioMapSeer radio map regression."
    )
    parser.add_argument("--config-path", type=str, required=True, help="Path to the experiment JSON configuration.")
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

    # Spatial evaluation (Boundary / LoS / NLoS / Long-range NLoS).
    # Values in cfg['spatial_eval'] are used by default; these two arguments are
    # convenient run-time overrides for the precomputed geometry root and d0.
    parser.add_argument(
        "--spatial-geo-root",
        type=str,
        default=None,
        help=(
            "Root containing precomputed obstacle-transmittance .pt files. "
            "Overrides cfg['spatial_eval']['geo_precompute_root']."
        ),
    )
    parser.add_argument(
        "--spatial-d0",
        type=float,
        default=None,
        help=(
            "Optional fixed Long-range NLoS distance threshold in pixels. "
            "If omitted, d0 is derived once from the validation split using "
            "the configured outdoor-distance quantile (default: 0.75)."
        ),
    )
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
    target_type = normalize_target_type(cfg_get(cfg, ["data", "target_type"], "DPM"))
    return {
        "data_root": cfg_get(cfg, ["data", "root_dir"], None),
        "target_type": target_type,
        "split": "val" if args.split == "valid" else args.split,
        "num_tx": cfg_get(cfg, ["data", "num_tx"], 80),
        "thresh": cfg_get(cfg, ["data", "thresh"], 0.0),
        "batch_size": cfg_get(cfg, ["data", "batch_size"], 32),
        "num_workers": cfg_get(cfg, ["data", "num_workers"], 4),
        "seed": cfg_get(cfg, ["seed"], 42),
    }



def to_environment_target_type(target_type):
    """Map fidelity-specific targets to the geometry mode expected by metrics.py."""
    target_type = normalize_target_type(target_type)
    if target_type in ["carsDPM", "carsIRT4"]:
        return "carsDPM"
    return "DPM"


def infer_spatial_geo_mode_name(target_type):
    environment_type = to_environment_target_type(target_type)
    if environment_type == "carsDPM":
        return "cars_carsDPM"
    return "building_DPM"


def resolve_spatial_options(args, cfg, target_type):
    """Resolve paper spatial-evaluation settings.

    Recommended config block:
        "spatial_eval": {
          "geo_precompute_root": "./data/precomputed_obstacle",
          "geo_mode_name": null,
          "geo_key": "obstacle_saturating_a007",
          "boundary_radius": 3,
          "long_range_quantile": 0.75,
          "d0": null
        }

    d0=None means: derive it from the validation split only, then keep it fixed
    for the requested evaluation split.
    """
    spatial_cfg = cfg_get(cfg, ["spatial_eval"], {}) or {}

    geo_root = (
        args.spatial_geo_root
        or spatial_cfg.get("geo_precompute_root")
        or cfg_get(cfg, ["physics", "geo_precompute_root"], None)
        or cfg_get(cfg, ["precompute_obstacle", "save_root"], None)
        or "./data/precomputed_obstacle"
    )
    geo_mode_name = (
        spatial_cfg.get("geo_mode_name")
        or infer_spatial_geo_mode_name(target_type)
    )
    geo_key = str(spatial_cfg.get("geo_key", "obstacle_saturating_a007"))
    boundary_radius = int(spatial_cfg.get("boundary_radius", 3))
    long_range_quantile = float(spatial_cfg.get("long_range_quantile", 0.75))

    configured_d0 = spatial_cfg.get("d0", None)
    d0 = args.spatial_d0 if args.spatial_d0 is not None else configured_d0
    if d0 is not None:
        d0 = float(d0)

    if boundary_radius < 0:
        raise ValueError("spatial_eval.boundary_radius must be >= 0.")
    if not (0.0 < long_range_quantile < 1.0):
        raise ValueError("spatial_eval.long_range_quantile must be in (0, 1).")

    return {
        "geo_precompute_root": os.path.abspath(os.path.expanduser(str(geo_root))),
        "geo_mode_name": str(geo_mode_name),
        "geo_key": geo_key,
        "boundary_radius": boundary_radius,
        "long_range_quantile": long_range_quantile,
        "d0": d0,
    }


def _safe_sample_stem(name):
    stem = os.path.basename(str(name))
    if stem.endswith(".png"):
        stem = stem[:-4]
    return stem.replace(os.sep, "_").replace(" ", "_")


def load_precomputed_transmittance(
    names,
    device,
    geo_precompute_root,
    geo_mode_name,
    split_name,
    geo_key="obstacle_saturating_a007",
):
    """Load z_tr=exp(-alpha*A(p)) for the current batch.

    Expected layout:
        <root>/<mode>/<split>/<sample>.pt

    The saved .pt must contain `geo_key`, e.g. obstacle_saturating_a007.
    """
    if names is None:
        raise ValueError(
            "Spatial metrics require sample names so precomputed geometry targets "
            "can be loaded. prepare_eval_config(..., return_name=True) must be used."
        )

    split_name = "val" if split_name == "valid" else str(split_name)
    tensors = []

    for name in names:
        stem = _safe_sample_stem(name)
        path = os.path.join(
            geo_precompute_root,
            geo_mode_name,
            split_name,
            f"{stem}.pt",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing precomputed spatial target: {path}\n"
                "Precompute this evaluation split first (include 'test' when "
                "evaluating the test set)."
            )

        data = torch.load(path, map_location="cpu")
        if geo_key not in data:
            raise KeyError(
                f"'{geo_key}' was not found in {path}. "
                f"Available keys: {list(data.keys())}"
            )

        z_tr = data[geo_key].float()
        if z_tr.ndim == 2:
            z_tr = z_tr.unsqueeze(0)
        if z_tr.ndim != 3 or z_tr.shape[0] != 1:
            raise ValueError(
                f"Expected {geo_key} to have shape [1,H,W] or [H,W], "
                f"got {tuple(z_tr.shape)} from {path}."
            )
        tensors.append(z_tr)

    return torch.stack(tensors, dim=0).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


@torch.no_grad()
def estimate_long_range_d0_from_validation(
    val_loader,
    device,
    target_type,
    quantile=0.75,
):
    """Compute the exact pixel-distance quantile over validation outdoor pixels.

    Distance squared is integer-valued on the pixel grid, so we accumulate a
    histogram of squared Tx-to-pixel distances instead of storing all distances.
    This gives an exact quantile with small, fixed memory usage.
    """
    hist = None
    max_dist2 = None

    for batch in tqdm(val_loader, desc="derive spatial d0 (val)", leave=False):
        x = batch[0].to(device, non_blocking=True)
        x = x.detach().float()

        obstacle = get_obstacle_mask(x, to_environment_target_type(target_type))
        outdoor = ~obstacle

        batch_size, _, height, width = x.shape
        current_max_dist2 = (height - 1) ** 2 + (width - 1) ** 2
        if hist is None:
            max_dist2 = current_max_dist2
            hist = torch.zeros(max_dist2 + 1, dtype=torch.int64)
        elif current_max_dist2 != max_dist2:
            raise ValueError("All validation samples must share the same spatial size.")

        tx_map = x[:, -1:, :, :]
        flat_idx = tx_map.reshape(batch_size, -1).argmax(dim=1)
        tx_y = (flat_idx // width).view(batch_size, 1, 1, 1).long()
        tx_x = (flat_idx % width).view(batch_size, 1, 1, 1).long()

        yy = torch.arange(height, device=device, dtype=torch.long).view(1, 1, height, 1)
        xx = torch.arange(width, device=device, dtype=torch.long).view(1, 1, 1, width)
        dist2 = (yy - tx_y).square() + (xx - tx_x).square()

        valid_dist2 = dist2[outdoor].to(torch.int64)
        if valid_dist2.numel() == 0:
            continue

        hist += torch.bincount(
            valid_dist2.detach().cpu(),
            minlength=max_dist2 + 1,
        )

    if hist is None or int(hist.sum().item()) == 0:
        raise RuntimeError("No outdoor validation pixels were available to derive d0.")

    total = int(hist.sum().item())
    rank = max(1, int(math.ceil(float(quantile) * total)))
    cumulative = torch.cumsum(hist, dim=0)
    dist2_threshold = int(torch.searchsorted(cumulative, torch.tensor(rank)).item())
    return math.sqrt(float(dist2_threshold))


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
    """Select DPM/carsDPM or IRT4/carsIRT4 loaders from target_type only."""
    target_type = normalize_target_type(
        cfg_get(cfg, ["data", "target_type"], "DPM")
    )

    if target_type in ["DPM", "carsDPM"]:
        builder = build_dataloaders
    elif target_type in ["IRT4", "carsIRT4"]:
        builder = build_irt4_dataloaders
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    try:
        return builder(cfg, return_datasets=True)
    except TypeError as exc:
        raise TypeError(
            f"{builder.__name__} must accept return_datasets=True. "
            "Update datasets/rms_dataset.py with the unified target_type version."
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
    if normalize_target_type(target_type) in ["carsDPM", "carsIRT4"]:
        if input_arr.shape[0] < 2:
            raise ValueError(
                "Car-aware target selected but input_map does not contain channel 1 for cars."
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

    ssim = compute_ssim(
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
        "SSIM": float(ssim),
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
    split_name="test",
    spatial_options=None,
    spatial_d0=None,
    spatial_d0_source="validation_outdoor_distance_quantile",
):
    model.eval()

    if spatial_options is None:
        raise ValueError("spatial_options must be provided for spatial evaluation.")
    if spatial_d0 is None:
        raise ValueError("spatial_d0 must be resolved before evaluate().")

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
    spatial_metric_store = defaultdict(list)

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

        # Spatial-region metrics are intentionally computed outside the timed
        # model-forward section so they do not contaminate inference latency.
        z_tr = load_precomputed_transmittance(
            names=names,
            device=device,
            geo_precompute_root=spatial_options["geo_precompute_root"],
            geo_mode_name=spatial_options["geo_mode_name"],
            split_name=split_name,
            geo_key=spatial_options["geo_key"],
        )
        update_spatial_metrics(
            metric_store=spatial_metric_store,
            pred=pred.detach().float(),
            target=y.detach().float(),
            x=x.detach().float(),
            target_type=to_environment_target_type(target_type),
            d0=float(spatial_d0),
            z_tr=z_tr,
            boundary_radius=spatial_options["boundary_radius"],
        )

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
    spatial_summary = summarize_spatial_metrics(spatial_metric_store)

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
        "spatial_metric_protocol": "per_sample_masked_rmse_then_mean",
        "spatial_metrics": spatial_summary,
        "spatial_eval": {
            "boundary_radius_px": int(spatial_options["boundary_radius"]),
            "long_range_quantile": float(spatial_options["long_range_quantile"]),
            "d0_px": float(spatial_d0),
            "d0_source": str(spatial_d0_source),
            "geo_precompute_root": spatial_options["geo_precompute_root"],
            "geo_mode_name": spatial_options["geo_mode_name"],
            "geo_key": spatial_options["geo_key"],
        },
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

    spatial_options = resolve_spatial_options(args, cfg, opts["target_type"])
    if spatial_options["d0"] is None:
        spatial_d0 = estimate_long_range_d0_from_validation(
            val_loader=split_dict["val_loader"],
            device=device,
            target_type=opts["target_type"],
            quantile=spatial_options["long_range_quantile"],
        )
        spatial_d0_source = "validation_outdoor_distance_quantile"
    else:
        spatial_d0 = float(spatial_options["d0"])
        spatial_d0_source = "configured_or_cli"

    spatial_split_dir = os.path.join(
        spatial_options["geo_precompute_root"],
        spatial_options["geo_mode_name"],
        opts["split"],
    )
    if not os.path.isdir(spatial_split_dir):
        raise FileNotFoundError(
            f"Spatial precompute directory not found: {spatial_split_dir}\n"
            "Run tools/precompute_obstacle_targets.py with this split included."
        )

    print(f"EVAL_SIZE : {len(dataset)}")
    print(f"Split     : {opts['split']}")
    print(f"Target    : {opts['target_type']}")
    print(f"Spatial d0: {spatial_d0:.4f} px ({spatial_d0_source})")
    print(f"Geo target: {spatial_options['geo_key']} @ {spatial_split_dir}")

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
        split_name=opts["split"],
        spatial_options=spatial_options,
        spatial_d0=spatial_d0,
        spatial_d0_source=spatial_d0_source,
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
        f.write("\nSpatial Evaluation\n")
        f.write(f"boundary_radius_px: {metrics['spatial_eval']['boundary_radius_px']}\n")
        f.write(f"long_range_quantile: {metrics['spatial_eval']['long_range_quantile']:.4f}\n")
        f.write(f"d0_px: {metrics['spatial_eval']['d0_px']:.6f}\n")
        f.write(f"d0_source: {metrics['spatial_eval']['d0_source']}\n")
        f.write(f"geo_mode_name: {metrics['spatial_eval']['geo_mode_name']}\n")
        f.write(f"geo_key: {metrics['spatial_eval']['geo_key']}\n")
        f.write(f"geo_precompute_root: {metrics['spatial_eval']['geo_precompute_root']}\n")
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
        f.write("\nSpatial RMSE\n")
        for display_name, key in [
            ("Boundary", "boundary"),
            ("LoS", "los"),
            ("NLoS", "nlos"),
            ("Long-range NLoS", "long_nlos"),
        ]:
            result = metrics["spatial_metrics"].get(key, {})
            f.write(
                f"{display_name} RMSE : {result.get('rmse', float('nan')):.6f} "
                f"| std: {result.get('std', float('nan')):.6f} "
                f"| n: {result.get('num_samples', 0)}\n"
            )

    print("\nResults")
    print(f"MAE : {metrics['MAE']:.6f}")
    print(f"MSE : {metrics['MSE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"NMSE : {metrics['NMSE']:.6f}")
    print(f"PSNR : {metrics['PSNR']:.4f} dB")
    print(f"SSIM : {metrics['SSIM']:.6f}")
    print("\nSpatial RMSE")
    for display_name, key in [
        ("Boundary", "boundary"),
        ("LoS", "los"),
        ("NLoS", "nlos"),
        ("Long-range NLoS", "long_nlos"),
    ]:
        result = metrics["spatial_metrics"].get(key, {})
        print(
            f"{display_name:<16}: {result.get('rmse', float('nan')):.6f} "
            f"(std={result.get('std', float('nan')):.6f}, "
            f"n={result.get('num_samples', 0)})"
        )
    print(f"d0: {metrics['spatial_eval']['d0_px']:.4f} px")
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
