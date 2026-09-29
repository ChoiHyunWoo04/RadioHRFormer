import os
import sys
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler

from datasets.rms_dataset import build_dataloaders, build_irt4_dataloaders, normalize_target_type
from models.hrformer_regressor import HRFormerRadioMapRegressor, load_physics_pretrained_for_downstream

from utils import (
    set_seed,
    get_amp_device_type,
    summarize_trainable_by_module,
    get_default_run_name
)
from losses import (
    MAE,
    JointLoss,
)
from metrics import (
    compute_rmse,
    compute_nmse,
    compute_psnr,
    compute_ssim,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HRFormer for RadioMapSeer radio map regression."
    )
    parser.add_argument("--config-path", type=str, required=True, help="Path to the experiment JSON configuration.")
    parser.add_argument("--save-root", type=str, default="./save")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--physics-pretrained", type=str, default=None)
    parser.add_argument(
        "--carsdpm-pretrained",
        type=str,
        default=None,
        help=(
            "Path to a fully trained carsDPM downstream checkpoint used to initialize "
            "carsIRT4 fine-tuning. The complete model, including the regression head, "
            "is loaded with strict=True; optimizer state is not restored."
        ),
    )
    parser.add_argument("--eval-split", choices=["val", "valid", "test"], default="val")
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


def prepare_train_config(cfg, return_name=False):
    """Use the JSON configuration as the single source of data/training settings."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})

    if cfg["data"].get("root_dir") is None:
        raise ValueError("cfg['data']['root_dir'] must be set.")

    cfg["data"]["return_name"] = bool(return_name)
    return cfg


def resolve_options(cfg):
    target_type = normalize_target_type(cfg_get(cfg, ["data", "target_type"], "DPM"))
    return {
        "data_root": cfg_get(cfg, ["data", "root_dir"], None),
        "target_type": target_type,
        "input_mode": "cars" if target_type in ["carsDPM", "carsIRT4"] else "building",
        "num_tx": cfg_get(cfg, ["data", "num_tx"], 80),
        "thresh": cfg_get(cfg, ["data", "thresh"], 0.0),
        "batch_size": cfg_get(cfg, ["data", "batch_size"], 32),
        "num_workers": cfg_get(cfg, ["data", "num_workers"], 4),
        "epochs": cfg_get(cfg, ["train", "epochs"], 200),
        "lr": cfg_get(cfg, ["train", "lr"], 1e-4),
        "weight_decay": cfg_get(cfg, ["train", "weight_decay"], 1e-4),
        "warmup_epochs": cfg_get(cfg, ["train", "warmup_epochs"], 5),
        "min_lr": cfg_get(cfg, ["train", "min_lr"], 1e-6),
        "loss": cfg_get(cfg, ["train", "loss"], "joint"),
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
    """Select the loader builder from data.target_type only."""
    target_type = normalize_target_type(
        cfg_get(cfg, ["data", "target_type"], "DPM")
    )

    if target_type in ["DPM", "carsDPM"]:
        builder = build_dataloaders
    elif target_type in ["IRT4", "carsIRT4"]:
        builder = build_irt4_dataloaders
    else:  # normalize_target_type already guards this path.
        raise ValueError(f"Unsupported target_type: {target_type}")

    try:
        return builder(cfg, return_datasets=True)
    except TypeError as exc:
        raise TypeError(
            f"{builder.__name__} must accept return_datasets=True. "
            "Update datasets/rms_dataset.py with the unified target_type version."
        ) from exc


def get_eval_loader(split_dict, split_name):
    split_name = "val" if split_name == "valid" else split_name
    if split_name not in ["val", "test"]:
        raise ValueError(f"Unsupported eval split: {split_name}")
    return split_dict[f"{split_name}_loader"], split_dict[f"{split_name}_dataset"]


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


def build_loss(loss_name):
    loss_name = loss_name.lower()
    if loss_name == "l1":
        return torch.nn.L1Loss()
    if loss_name == "mse":
        return torch.nn.MSELoss()
    if loss_name == "joint":
        return JointLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def adjust_learning_rate(optimizer, epoch, opts):
    base_lr = opts["lr"]
    min_lr = opts["min_lr"]
    warmup_epochs = opts["warmup_epochs"]
    total_epochs = opts["epochs"]

    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    else:
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler, use_amp, epoch=None):
    model.train()

    total_loss = 0.0
    total_seen = 0

    # Streaming MAE
    total_abs_error = 0.0
    total_elements = 0

    desc = "Train" if epoch is None else f"Train Epoch {epoch}"
    pbar = tqdm(loader, desc=desc, leave=False)
    amp_device_type = get_amp_device_type(device)

    for batch in pbar:
        x, y, _ = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(x)
            loss = loss_fn(pred, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_seen += bs

        # Accumulate MAE without storing all predictions.
        # Use FP32 for the reported metric even when AMP is enabled.
        with torch.no_grad():
            abs_error = torch.abs(
                pred.detach().float() - y.detach().float()
            )
            total_abs_error += abs_error.sum().item()
            total_elements += abs_error.numel()

        pbar.set_postfix({
            "loss": f"{total_loss / total_seen:.5f}",
        })

    avg_loss = total_loss / total_seen
    avg_mae = total_abs_error / total_elements

    return avg_loss, avg_mae


@torch.no_grad()
def evaluate_one_epoch(model, loader, loss_fn, device, use_amp=False, epoch=None, split_name="Valid"):
    model.eval()
    total_loss = 0.0
    total_seen = 0

    # JointLoss 항목별 누적용
    component_sums = {}

    pred_all = []
    gt_all = []

    desc = split_name if epoch is None else f"{split_name} Epoch {epoch}"
    pbar = tqdm(loader, desc=desc, leave=False)
    amp_device_type = get_amp_device_type(device)

    for batch in pbar:
        x, y, _ = unpack_batch(batch, device)

        with autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(x)

            # JointLoss처럼 return_dict를 지원하는 loss만 항목별 반환
            try:
                loss, loss_dict = loss_fn(pred, y, return_dict=True)
            except TypeError:
                loss = loss_fn(pred, y)
                loss_dict = None

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_seen += bs

        if loss_dict is not None:
            for key, value in loss_dict.items():
                if key not in component_sums:
                    component_sums[key] = 0.0
                component_sums[key] += float(value.item()) * bs

        pred_all.append(pred.detach().cpu())
        gt_all.append(y.detach().cpu())

        postfix = {
            "loss": f"{total_loss / total_seen:.5f}",
        }

        if component_sums:
            postfix.update({
                "mae_l": f"{component_sums['loss_mae'] / total_seen:.5f}",
                "mse_l": f"{component_sums['loss_mse'] / total_seen:.6f}",
                "grad_l": f"{component_sums['loss_grad'] / total_seen:.5f}",
                "ssim_l": f"{component_sums['loss_ssim'] / total_seen:.5f}",
            })

        pbar.set_postfix(postfix)

    pred_all = torch.cat(pred_all, dim=0).float()
    gt_all = torch.cat(gt_all, dim=0).float()

    metrics = {
        "loss": total_loss / total_seen,
        "MAE": float(MAE(gt_all, pred_all)),
        "RMSE": float(compute_rmse(pred_all, gt_all)),
        "NMSE": float(compute_nmse(pred_all, gt_all)),
        "PSNR": float(compute_psnr(pred_all, gt_all)),
        "SSIM": float(compute_ssim(pred_all, gt_all)),
    }

    # 항목별 validation loss 평균 추가
    for key, value in component_sums.items():
        metrics[key] = value / total_seen

    return metrics


def save_loss_curve(train_losses, val_losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(x, train_losses, label="train")
    plt.plot(x, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_checkpoint(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        unwrap_model(model).state_dict(),
        save_path,
    )


def load_model_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    unwrap_model(model).load_state_dict(normalize_state_dict_keys(state_dict), strict=True)


def main():
    args = parse_args()
    cfg = prepare_train_config(load_config(args.config_path), return_name=False)
    opts = resolve_options(cfg)

    set_seed(opts["seed"])
    device, gpu_ids, use_amp = resolve_runtime(cfg)
    opts["gpus"] = gpu_ids if gpu_ids else ["cpu"]
    opts["amp"] = use_amp

    split_dict = get_split_dict(cfg)
    train_loader = split_dict["train_loader"]
    train_dataset = split_dict["train_dataset"]
    val_loader, val_dataset = get_eval_loader(split_dict, args.eval_split)

    run_name = args.run_name or get_default_run_name(args.config_path)
    save_folder = os.path.join(args.save_root, run_name)
    weight_dir = os.path.join(save_folder, "weight")
    os.makedirs(weight_dir, exist_ok=True)
    log_path = os.path.join(save_folder, "log.txt")

    print(f"TRAIN_SIZE: {len(train_dataset)}")
    print(f"VAL_SIZE : {len(val_dataset)}")
    print(f"Input mode: {opts['input_mode']}")
    print(f"Target    : {opts['target_type']}")

    base_model = HRFormerRadioMapRegressor(cfg).to(device)

    # Exactly one initialization mode may be used.
    init_args = {
        "physics_pretrained": args.physics_pretrained,
        "carsdpm_pretrained": args.carsdpm_pretrained,
    }
    active_init = [name for name, path in init_args.items() if path is not None]
    if len(active_init) > 1:
        raise ValueError(
            "Use only one of --physics-pretrained, or --carsdpm-pretrained. "
            f"Received: {active_init}"
        )

    if args.physics_pretrained is not None:
        load_physics_pretrained_for_downstream(
            model=base_model,
            ckpt_path=args.physics_pretrained,
            device=device,
            verbose=True,
        )
        print(f"Loaded physics-pretrained initialization: {args.physics_pretrained}")

    if args.carsdpm_pretrained is not None:
        if opts["target_type"] != "carsIRT4":
            raise ValueError(
                "--carsdpm-pretrained is intended for carsDPM -> carsIRT4 fine-tuning. "
                "Set cfg['data']['target_type'] = 'carsIRT4'."
            )
        if args.eval_split == "test":
            raise ValueError(
                "Do not use --eval-split test during carsIRT4 fine-tuning. "
                "Use val/valid for checkpoint selection and evaluate test only afterward."
            )

        # Load the complete carsDPM downstream predictor (backbone + decoder +
        # regression head). Unlike --physics-pretrained, this is a full strict load.
        # A fresh optimizer is created below, so the carsDPM optimizer state and LR
        # schedule are intentionally not restored.
        load_model_state(base_model, args.carsdpm_pretrained, device)
        print(
            "Loaded full carsDPM downstream initialization for carsIRT4 fine-tuning: "
            f"{args.carsdpm_pretrained}"
        )

    summarize_trainable_by_module(base_model)
    model = maybe_wrap_data_parallel(base_model, gpu_ids)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opts["lr"],
        weight_decay=opts["weight_decay"],
        betas=(0.9, 0.999),
    )
    loss_fn = build_loss(opts["loss"]).to(device)
    scaler = GradScaler(device.type, enabled=use_amp)

    train_losses = []
    val_losses = []
    best_rmse = float("inf")
    last_metrics = None

    with open(log_path, "w") as f:
        f.write("model: HRFormerRadioMapRegressor\n")
        f.write(f"config_path: {args.config_path}\n")
        f.write(f"gpus: {opts['gpus']}\n")
        f.write(f"amp: {opts['amp']}\n")
        f.write(f"data_root: {opts['data_root']}\n")
        f.write(f"input_mode: {opts['input_mode']}\n")
        f.write(f"target_type: {opts['target_type']}\n")
        f.write(f"num_tx: {opts['num_tx']}\n")
        f.write(f"batch_size_global: {opts['batch_size']}\n")
        f.write(f"epochs: {opts['epochs']}\n")
        f.write(f"lr: {opts['lr']}\n")
        f.write(f"weight_decay: {opts['weight_decay']}\n")
        f.write(f"loss: {opts['loss']}\n")
        f.write(f"physics_pretrained: {args.physics_pretrained}\n")
        f.write(f"carsdpm_pretrained: {args.carsdpm_pretrained}\n\n")

    with open(os.path.join(save_folder, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for epoch in range(opts["epochs"]):
        lr = adjust_learning_rate(optimizer, epoch, opts)
        train_loss, train_mae = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch + 1,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            epoch=epoch + 1,
            split_name="Valid",
        )
        last_metrics = val_metrics
        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        msg = (
            f"Epoch [{epoch + 1}/{opts['epochs']}] "
            f"lr={lr:.6e} "
            f"train_loss={train_loss:.5f} train_MAE={train_mae:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_MAE={val_metrics['MAE']:.5f} "
            f"val_RMSE={val_metrics['RMSE']:.5f} val_NMSE={val_metrics['NMSE']:.5f} "
            f"val_PSNR={val_metrics['PSNR']:.2f} val_SSIM={val_metrics['SSIM']:.5f}"
        )

        if "loss_mae" in val_metrics:
            msg += (
                f" | raw: "
                f"mae={val_metrics['loss_mae']:.5f} "
                f"mse={val_metrics['loss_mse']:.7f} "
                f"grad={val_metrics['loss_grad']:.5f} "
                f"ssim={val_metrics['loss_ssim']:.5f}"
            )

        if "w_loss_mae" in val_metrics:
            msg += (
                f" | weighted: "
                f"mae={val_metrics['w_loss_mae']:.5f} "
                f"mse={val_metrics['w_loss_mse']:.7f} "
                f"grad={val_metrics['w_loss_grad']:.5f} "
                f"ssim={val_metrics['w_loss_ssim']:.5f}"
            )
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        if val_metrics["RMSE"] < best_rmse:
            best_rmse = val_metrics["RMSE"]
            save_checkpoint(
                model,
                os.path.join(weight_dir, "best.pth"),
            )

    save_checkpoint(
        model,
        os.path.join(weight_dir, "last.pth"),
    )
    save_loss_curve(train_losses, val_losses, os.path.join(save_folder, "loss.png"))

    print("\nFinished training")
    print(f"Best RMSE checkpoint: {os.path.join(weight_dir, 'best.pth')}")
    print(f"Last checkpoint    : {os.path.join(weight_dir, 'last.pth')}")
    print(f"Log                : {log_path}")


if __name__ == "__main__":
    main()
