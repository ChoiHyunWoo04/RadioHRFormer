import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
from torch.amp import autocast

from datasets.rms_dataset import build_dataloaders
from models.hrformer_regressor import HRFormerRadioMapRegressor

from utils import (
    set_seed,
    show_current_cuda_memory,
    prepare_device,
    is_cuda_device,
    get_amp_device_type,
    use_amp_on_device,
    summarize_trainable_by_module,
)
from losses import (
    MAE,
)
from metrics import (
    compute_rmse,
    compute_nmse,
    compute_psnr,
    compute_ssim,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HRFormer for RadioMapSeer radio map regression."
    )

    # Paths
    parser.add_argument("--config-path", type=str, default="./configs/hrformer_radiomapseer.json")
    parser.add_argument("--weight-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--save-root", type=str, default="./save_eval")
    parser.add_argument("--run-name", type=str, default=None)

    # RadioMapSeer setting
    parser.add_argument(
        "--input-mode",
        choices=["building", "cars"],
        default="building",
        help="building: [building, building, Tx], cars: [building, cars, Tx].",
    )
    parser.add_argument(
        "--target-type",
        choices=["DPM", "carsDPM"],
        default=None,
        help="If omitted: building -> DPM, cars -> carsDPM.",
    )
    parser.add_argument("--split", choices=["train", "val", "valid", "test"], default="test")
    parser.add_argument("--num-tx", type=int, default=None)
    parser.add_argument("--thresh", type=float, default=None)

    # Runtime
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    # Visualization
    parser.add_argument("--save-pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cmap", type=str, default="jet")
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


def arg_or_cfg(arg_value, config, keys, default):
    if arg_value is not None:
        return arg_value
    return cfg_get(config, keys, default)


def apply_radiomapseer_overrides(cfg, args, return_name=True):
    cfg.setdefault("data", {})

    if args.data_root is not None:
        cfg["data"]["root_dir"] = args.data_root
    if args.num_tx is not None:
        cfg["data"]["num_tx"] = args.num_tx
    if args.thresh is not None:
        cfg["data"]["thresh"] = args.thresh
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    cfg["data"]["pin_memory"] = args.pin_memory
    cfg["data"]["persistent_workers"] = args.persistent_workers
    cfg["data"]["return_name"] = return_name

    if args.input_mode == "building":
        cfg["data"]["cars_input"] = False
        cfg["data"]["target_type"] = args.target_type if args.target_type is not None else "DPM"
    elif args.input_mode == "cars":
        cfg["data"]["cars_input"] = True
        cfg["data"]["target_type"] = args.target_type if args.target_type is not None else "carsDPM"
    else:
        raise ValueError(f"Unsupported input_mode: {args.input_mode}")

    cfg["data"].setdefault("root_dir", None)
    cfg["data"].setdefault("num_tx", 80)
    cfg["data"].setdefault("thresh", 0.0)
    cfg["data"].setdefault("batch_size", 32)
    cfg["data"].setdefault("num_workers", 4)

    if cfg["data"]["root_dir"] is None:
        raise ValueError("RadioMapSeer path is required. Use --data-root or cfg['data']['root_dir'].")

    return cfg


def resolve_options(args, cfg):
    return {
        "data_root": cfg_get(cfg, ["data", "root_dir"], None),
        "input_mode": args.input_mode,
        "cars_input": cfg_get(cfg, ["data", "cars_input"], False),
        "target_type": cfg_get(cfg, ["data", "target_type"], "DPM"),
        "split": "val" if args.split == "valid" else args.split,
        "num_tx": cfg_get(cfg, ["data", "num_tx"], 80),
        "thresh": cfg_get(cfg, ["data", "thresh"], 0.0),
        "batch_size": cfg_get(cfg, ["data", "batch_size"], 32),
        "num_workers": cfg_get(cfg, ["data", "num_workers"], 4),
        "seed": arg_or_cfg(args.seed, cfg, ["seed"], 42),
    }


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
    model.load_state_dict(state_dict, strict=True)


def tensor_to_image(x):
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    elif x.ndim == 3:
        x = x.permute(1, 2, 0)
    return x.numpy()


def save_map_image(img, save_path, cmap="jet", vmin=None, vmax=None, title=None):
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


def reset_cuda_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def get_memory_profile(device):
    if torch.device(device).type != "cuda":
        return {
            "device_type": torch.device(device).type,
            "cuda_memory_allocated_mb": None,
            "cuda_memory_reserved_mb": None,
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }

    torch.cuda.synchronize(device)
    return {
        "device_type": "cuda",
        "cuda_memory_allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(device)),
        "cuda_max_memory_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(device)),
    }


def sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, loader, device, args, save_folder, use_amp=False):
    model.eval()
    pred_all = []
    gt_all = []

    pred_dir = os.path.join(save_folder, "pred_png")
    gt_dir = os.path.join(save_folder, "gt_png")
    err_dir = os.path.join(save_folder, "error_png")

    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)
    if args.save_gt:
        os.makedirs(gt_dir, exist_ok=True)
    if args.save_error:
        os.makedirs(err_dir, exist_ok=True)

    global_idx = 0
    amp_device_type = get_amp_device_type(device)
    total_forward_time_sec = 0.0
    total_infer_samples = 0

    reset_cuda_peak_memory(device)

    for batch in tqdm(loader):
        x, y, names = unpack_batch(batch, device)

        # Measure only model forward time. This excludes data loading, CPU transfer,
        # metric computation, and PNG saving overhead.
        sync_if_cuda(device)
        start_time = time.perf_counter()
        with autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(x)
        sync_if_cuda(device)
        elapsed = time.perf_counter() - start_time

        batch_size = x.size(0)
        total_forward_time_sec += elapsed
        total_infer_samples += batch_size

        pred_all.append(pred.detach().cpu())
        gt_all.append(y.detach().cpu())

        for b in range(x.size(0)):
            should_save = args.max_save < 0 or global_idx < args.max_save
            if not should_save:
                global_idx += 1
                continue

            if names is not None:
                name = os.path.splitext(str(names[b]))[0]
            else:
                name = f"{global_idx:06d}"

            pred_b = pred[b].detach().cpu()
            gt_b = y[b].detach().cpu()
            sample_mae = torch.mean(torch.abs(pred_b - gt_b)).item()

            if args.save_pred:
                save_map_image(
                    img=pred_b,
                    save_path=os.path.join(pred_dir, f"{name}_mae_{sample_mae:.4f}.png"),
                    cmap=args.cmap,
                )
            if args.save_gt:
                save_map_image(
                    img=gt_b,
                    save_path=os.path.join(gt_dir, f"{name}_gt.png"),
                    cmap=args.cmap,
                )
            if args.save_error:
                save_error_image(
                    pred=pred_b,
                    gt=gt_b,
                    save_path=os.path.join(err_dir, f"{name}_error.png"),
                    error_vmax=args.error_vmax,
                )

            global_idx += 1

    pred_all = torch.cat(pred_all, dim=0).float()
    gt_all = torch.cat(gt_all, dim=0).float()
    avg_inference_time_sec = (
        total_forward_time_sec / total_infer_samples
        if total_infer_samples > 0
        else float("nan")
    )
    memory_profile = get_memory_profile(device)

    return {
        "MAE": float(MAE(gt_all, pred_all)),
        "RMSE": float(compute_rmse(pred_all, gt_all)),
        "NMSE": float(compute_nmse(pred_all, gt_all)),
        "PSNR": float(compute_psnr(pred_all, gt_all)),
        "SSIM": float(compute_ssim(pred_all, gt_all)),
        "total_forward_time_sec": float(total_forward_time_sec),
        "avg_inference_time_sec_per_sample": float(avg_inference_time_sec),
        "avg_inference_time_ms_per_sample": float(avg_inference_time_sec * 1000.0),
        "num_inference_samples": int(total_infer_samples),
        **memory_profile,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    cfg = apply_radiomapseer_overrides(cfg, args, return_name=True)
    opts = resolve_options(args, cfg)

    set_seed(opts["seed"])
    device = prepare_device(args.cuda)
    use_amp = use_amp_on_device(device, args.amp)

    split_dict = get_split_dict(cfg)
    dataset, loader = select_split(split_dict, opts["split"])

    print(f"EVAL_SIZE : {len(dataset)}")
    print(f"Split     : {opts['split']}")
    print(f"Input mode: {opts['input_mode']}")
    print(f"Target    : {opts['target_type']}")

    model = HRFormerRadioMapRegressor(cfg).to(device)
    load_model_state(model, args.weight_path, device)
    summarize_trainable_by_module(model)
    model_profile = get_model_profile(model, args.weight_path)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.save_root, run_name)
    os.makedirs(save_folder, exist_ok=True)

    metrics = evaluate(model=model, loader=loader, device=device, args=args, save_folder=save_folder, use_amp=use_amp)

    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "a") as f:
        f.write("model: HRFormerRadioMapRegressor\n")
        f.write(f"config_path: {args.config_path}\n")
        f.write(f"weight_path: {args.weight_path}\n")
        f.write(f"data_root: {opts['data_root']}\n")
        f.write(f"split: {opts['split']}\n")
        f.write(f"input_mode: {opts['input_mode']}\n")
        f.write(f"cars_input: {opts['cars_input']}\n")
        f.write(f"target_type: {opts['target_type']}\n")
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
        f.write(f"avg_inference_time_per_sample: {metrics['avg_inference_time_ms_per_sample']:.6f} ms/sample\n")
        f.write(f"device_type: {metrics['device_type']}\n")
        if metrics["cuda_max_memory_allocated_mb"] is not None:
            f.write(f"cuda_memory_allocated_after_eval: {metrics['cuda_memory_allocated_mb']:.2f} MB\n")
            f.write(f"cuda_memory_reserved_after_eval: {metrics['cuda_memory_reserved_mb']:.2f} MB\n")
            f.write(f"cuda_peak_memory_allocated: {metrics['cuda_max_memory_allocated_mb']:.2f} MB\n")
            f.write(f"cuda_peak_memory_reserved: {metrics['cuda_max_memory_reserved_mb']:.2f} MB\n")
        f.write("\nResults\n")
        f.write(f"MAE : {metrics['MAE']:.4f}\n")
        f.write(f"RMSE : {metrics['RMSE']:.4f}\n")
        f.write(f"NMSE : {metrics['NMSE']:.4f}\n")
        f.write(f"PSNR : {metrics['PSNR']:.2f} dB\n")
        f.write(f"SSIM : {metrics['SSIM']:.4f}\n")

    print("\nResults")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"NMSE : {metrics['NMSE']:.4f}")
    print(f"PSNR : {metrics['PSNR']:.2f} dB")
    print(f"SSIM : {metrics['SSIM']:.4f}")
    print("\nModel Profile")
    print(f"Total params: {model_profile['total_params']:,}")
    print(f"Model size(param+buffer): {model_profile['model_size_mb']:.2f} MB")
    print("\nRuntime Profile")
    print(f"Avg inference time: {metrics['avg_inference_time_ms_per_sample']:.6f} ms/sample")
    if metrics["cuda_max_memory_allocated_mb"] is not None:
        print(f"CUDA peak memory allocated: {metrics['cuda_max_memory_allocated_mb']:.2f} MB")
        print(f"CUDA peak memory reserved: {metrics['cuda_max_memory_reserved_mb']:.2f} MB")
    print(f"Saved to: {save_folder}")


if __name__ == "__main__":
    main()
