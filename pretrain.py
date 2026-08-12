import os
import sys
import copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler

from datasets.rms_dataset import build_dataloaders
from datasets.physics_targets import PhysicsTargetBuilder, PhysicsPretrainLoss
from models.hrformer_regressor import HRFormerPhysicsPretrainer
from utils import (
    set_seed,
    get_amp_device_type,
    summarize_trainable_by_module,
)


def parse_args():
    p = argparse.ArgumentParser(description="Physics-map pretraining for HRFormer on RadioMapSeer.")
    p.add_argument("--config-path", type=str, default="./configs/hrt.json")
    p.add_argument("--save-root", type=str, default="./save_pretrain")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval-split", choices=["val", "valid", "test"], default="val")
    p.add_argument("--save-every", type=int, default=0)

    # Optional overrides for offline-precomputed geometry targets.
    # When omitted, values in cfg["physics"] are used unchanged.
    p.add_argument(
        "--geo-precompute-root",
        type=str,
        default=None,
        help=(
            "Optional root of precomputed geometry .pt files. Overrides "
            "physics.geo_precompute_root in the JSON config. Expected layout: "
            "<root>/<geo_mode_name>/<split>/<sample_name>.pt"
        ),
    )
    p.add_argument(
        "--geo-mode-name",
        type=str,
        default=None,
        help=(
            "Optional folder name under --geo-precompute-root. Overrides "
            "physics.geo_mode_name in the JSON config. If omitted, the script "
            "uses the configured value or '<input_mode>_<target_type>'."
        ),
    )
    return p.parse_args()


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def cfg_get(cfg, keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def apply_precompute_overrides(cfg, args):
    """Apply only explicit CLI overrides for the offline geometry-target directory."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("physics", {})

    if args.geo_precompute_root is not None:
        cfg["physics"]["geo_precompute_root"] = args.geo_precompute_root
    if args.geo_mode_name is not None:
        cfg["physics"]["geo_mode_name"] = args.geo_mode_name

    return cfg


def prepare_pretrain_config(cfg):
    """Validate the resolved config and enable names when .pt targets are used."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})
    cfg.setdefault("physics", {})
    cfg.setdefault("pretrain", {})

    if cfg["data"].get("root_dir") is None:
        raise ValueError("cfg['data']['root_dir'] must be set.")

    # Precomputed geometry targets are loaded by sample name. Force the dataset
    # to return names only when a precompute root is resolved.
    if cfg["physics"].get("geo_precompute_root"):
        cfg["data"]["return_name"] = True

    return cfg


def resolve_options(cfg):
    return {
        "epochs": cfg_get(cfg, ["pretrain", "epochs"], cfg_get(cfg, ["train", "epochs"], 200)),
        "lr": cfg_get(cfg, ["pretrain", "lr"], cfg_get(cfg, ["train", "lr"], 1e-4)),
        "weight_decay": cfg_get(cfg, ["pretrain", "weight_decay"], 1e-4),
        "warmup_epochs": cfg_get(cfg, ["pretrain", "warmup_epochs"], 5),
        "min_lr": cfg_get(cfg, ["pretrain", "min_lr"], 1e-6),
        "seed": cfg_get(cfg, ["seed"], 42),
        "batch_size": cfg_get(cfg, ["data", "batch_size"], 32),
        "target_type": cfg_get(cfg, ["data", "target_type"], "DPM"),
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


def load_model_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    unwrap_model(model).load_state_dict(normalize_state_dict_keys(state_dict), strict=True)


def adjust_learning_rate(optimizer, epoch, opts):
    base_lr, min_lr = opts["lr"], opts["min_lr"]
    warmup, total = opts["warmup_epochs"], opts["epochs"]
    if epoch < warmup:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup))
    else:
        progress = float(epoch - warmup) / float(max(1, total - warmup))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def unpack_batch(batch, device):
    if len(batch) == 3:
        x, y, names = batch
    elif len(batch) == 2:
        x, y = batch
        names = None
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), names


def format_logs(logs):
    return " ".join([f"{k}={v:.5f}" for k, v in logs.items()])


def train_one_epoch(model, loader, target_builder, loss_fn, optimizer, device, scaler, use_amp, epoch):
    model.train()
    total_loss, total_seen = 0.0, 0
    sum_logs = {}
    amp_device_type = get_amp_device_type(device)
    pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch}", leave=False)
    for batch in pbar:
        x, y, names = unpack_batch(batch, device)
        with torch.no_grad():
            targets = target_builder(x, names=names)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=amp_device_type, enabled=use_amp):
            preds = model(x)
            loss, logs = loss_fn(preds, targets)
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
        for k, v in logs.items():
            sum_logs[k] = sum_logs.get(k, 0.0) + v * bs
        pbar.set_postfix({"loss": f"{total_loss / total_seen:.5f}"})
    avg_logs = {k: v / total_seen for k, v in sum_logs.items()}
    avg_logs["loss"] = total_loss / total_seen
    return avg_logs


@torch.no_grad()
def evaluate_one_epoch(model, loader, target_builder, loss_fn, device, use_amp, epoch, split_name="Valid"):
    model.eval()
    total_seen = 0
    sum_logs = {}
    amp_device_type = get_amp_device_type(device)
    pbar = tqdm(loader, desc=f"{split_name} Epoch {epoch}", leave=False)
    for batch in pbar:
        x, y, names = unpack_batch(batch, device)
        targets = target_builder(x, names=names)
        with autocast(device_type=amp_device_type, enabled=use_amp):
            preds = model(x)
            _, logs = loss_fn(preds, targets)
        bs = x.size(0)
        total_seen += bs
        for k, v in logs.items():
            sum_logs[k] = sum_logs.get(k, 0.0) + v * bs
        pbar.set_postfix({"loss": f"{sum_logs['loss'] / total_seen:.5f}"})
    return {k: v / total_seen for k, v in sum_logs.items()}


def save_checkpoint(model, optimizer, epoch, metrics, opts, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "options": opts,
        },
        save_path,
    )


def save_loss_curve(train_losses, val_losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(x, train_losses, label="train")
    plt.plot(x, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("physics pretrain loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def make_target_builder(cfg, target_names, geo_split, device):
    physics_cfg = cfg_get(cfg, ["physics"], {})
    target_type = cfg_get(cfg, ["data", "target_type"], "DPM")
    input_mode = "cars" if target_type == "carsDPM" else "building"
    geo_mode_name = physics_cfg.get("geo_mode_name") or f"{input_mode}_{target_type}"

    return PhysicsTargetBuilder(
        target_names=target_names,
        tx_channel=int(physics_cfg.get("tx_channel", -1)),
        geo_precompute_root=physics_cfg.get("geo_precompute_root"),
        geo_mode_name=geo_mode_name,
        geo_split=geo_split,
    ).to(device)


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    cfg = apply_precompute_overrides(cfg, args)
    cfg = prepare_pretrain_config(cfg)
    opts = resolve_options(cfg)

    physics_cfg = cfg_get(cfg, ["physics"], {})
    raw_target_names = tuple(physics_cfg.get("targets", ["radial_gain","obstacle_saturating_a007"]))
    target_names = PhysicsTargetBuilder.normalize_target_names(raw_target_names)
    head_specs = PhysicsTargetBuilder.head_specs(target_names)

    input_mode = "cars" if opts["target_type"] == "carsDPM" else "building"
    geo_mode_name = physics_cfg.get("geo_mode_name") or f"{input_mode}_{opts['target_type']}"
    opts.update(
        {
            "physics_targets": target_names,
            "geo_precompute_root": physics_cfg.get("geo_precompute_root"),
            "geo_mode_name": geo_mode_name,
        }
    )

    if opts["geo_precompute_root"]:
        print(
            "[INFO] Loading precomputed geometry targets from: "
            f"{opts['geo_precompute_root']} / {opts['geo_mode_name']}"
        )
    else:
        print("[INFO] No geo_precompute_root is set; geometry targets are computed online.")

    set_seed(opts["seed"])
    device, gpu_ids, use_amp = resolve_runtime(cfg)
    opts["gpus"] = gpu_ids if gpu_ids else ["cpu"]
    opts["amp"] = use_amp

    try:
        split_dict = build_dataloaders(cfg, return_datasets=True)
    except TypeError as exc:
        raise TypeError("rms_dataset.build_dataloaders must support return_datasets=True.") from exc
    train_loader = split_dict["train_loader"]
    val_key = "val" if args.eval_split == "valid" else args.eval_split
    val_loader = split_dict[f"{val_key}_loader"]

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.save_root, run_name)
    weight_dir = os.path.join(save_folder, "weight")
    os.makedirs(weight_dir, exist_ok=True)
    log_path = os.path.join(save_folder, "log.txt")

    base_model = HRFormerPhysicsPretrainer(cfg, head_specs=head_specs).to(device)
    if args.resume is not None:
        load_model_state(base_model, args.resume, device)
        print(f"Loaded checkpoint: {args.resume}")
    summarize_trainable_by_module(base_model)
    model = maybe_wrap_data_parallel(base_model, gpu_ids)

    train_target_builder = make_target_builder(cfg, target_names, "train", device)
    val_target_builder = make_target_builder(cfg, target_names, val_key, device)

    loss_fn = PhysicsPretrainLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts["lr"],
        weight_decay=opts["weight_decay"],
        betas=(0.9, 0.999),
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    with open(log_path, "a") as f:
        f.write("model: HRFormerPhysicsPretrainer\n")
        f.write(json.dumps(opts, indent=2, default=str) + "\n")
        f.write(f"raw_target_names: {raw_target_names}\n")
        f.write(f"target_names: {target_names}\n")
        f.write(f"head_specs: {head_specs}\n\n")

    train_losses, val_losses = [], []
    best_loss = float("inf")
    last_metrics = None
    for epoch in range(opts["epochs"]):
        lr = adjust_learning_rate(optimizer, epoch, opts)
        train_logs = train_one_epoch(
            model,
            train_loader,
            train_target_builder,
            loss_fn,
            optimizer,
            device,
            scaler,
            use_amp,
            epoch + 1,
        )
        val_logs = evaluate_one_epoch(
            model,
            val_loader,
            val_target_builder,
            loss_fn,
            device,
            use_amp,
            epoch + 1,
        )
        last_metrics = val_logs
        train_losses.append(train_logs["loss"])
        val_losses.append(val_logs["loss"])
        msg = (
            f"Epoch [{epoch + 1}/{opts['epochs']}] lr={lr:.6e} "
            f"train_{format_logs(train_logs)} val_{format_logs(val_logs)}"
        )
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        if val_logs["loss"] < best_loss:
            best_loss = val_logs["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_logs,
                opts,
                os.path.join(weight_dir, "best.pth"),
            )
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_logs,
                opts,
                os.path.join(weight_dir, f"epoch_{epoch + 1:03d}.pth"),
            )

    save_checkpoint(
        model,
        optimizer,
        opts["epochs"],
        last_metrics,
        opts,
        os.path.join(weight_dir, "last.pth"),
    )
    save_loss_curve(train_losses, val_losses, os.path.join(save_folder, "loss.png"))
    print("\nFinished physics pretraining")
    print(f"Best checkpoint: {os.path.join(weight_dir, 'best.pth')}")
    print(f"Last checkpoint: {os.path.join(weight_dir, 'last.pth')}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
