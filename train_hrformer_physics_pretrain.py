import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    prepare_device,
    get_amp_device_type,
    use_amp_on_device,
    summarize_trainable_by_module,
)


def parse_args():
    p = argparse.ArgumentParser(description="Physics-map pretraining for HRFormer on RadioMapSeer.")
    p.add_argument("--config-path", type=str, default="./configs/hrformer_radiomapseer.json")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--save-root", type=str, default="./save_pretrain")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--input-mode", choices=["building", "cars"], default="building")
    p.add_argument("--target-type", choices=["DPM", "carsDPM"], default=None)
    p.add_argument("--num-tx", type=int, default=None)
    p.add_argument("--thresh", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cuda", type=str, default="0")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval-split", choices=["val", "valid", "test"], default="val")
    p.add_argument("--save-every", type=int, default=0)

    # Physics target options.
    p.add_argument(
        "--physics-targets",
        type=str,
        default="grad,lap,singularity,los,obstacle",
        help=(
            "Comma-separated targets. Supported: grad,lap,singularity,los,obstacle. "
            "Aliases such as radiodiff_k2_inv are internally mapped to singularity."
        ),
    )
    p.add_argument("--field-mode", choices=["normalized_power", "db_power", "pathloss_db"], default="normalized_power")
    p.add_argument("--gaussian-sigma", type=float, default=2.0)
    p.add_argument("--tx-channel", type=int, default=-1)
    p.add_argument("--building-threshold", type=float, default=0.5)
    p.add_argument("--obstacle-channels", type=str, default="0,1")
    p.add_argument("--ray-stride", type=int, default=1, help="Fallback online geometry only. Ignored when --geo-precompute-root is used.")
    p.add_argument("--radiodiff-pathloss-trunc", type=float, default=-147.0)
    p.add_argument("--radiodiff-pathloss-max", type=float, default=-47.0)
    p.add_argument("--radiodiff-source-power-dbm", type=float, default=23.0)
    p.add_argument("--radiodiff-h", type=float, default=1.0)
    p.add_argument("--radiodiff-border-value", type=float, default=1.0)
    p.add_argument("--radiodiff-eps", type=float, default=1e-30)
    p.add_argument("--radiodiff-smooth-sigma", type=float, default=0.9)

    # Precomputed geometry options.
    p.add_argument(
        "--geo-precompute-root",
        type=str,
        default=None,
        help=(
            "Root directory of precomputed los/obstacle .pt files. "
            "Expected: <root>/<geo-mode-name>/<split>/<sample_name>.pt. "
            "Example: ./data/precomputed_geo"
        ),
    )
    p.add_argument(
        "--geo-mode-name",
        type=str,
        default=None,
        help="Folder name under geo-precompute-root. If omitted, uses '<input_mode>_<target_type>'.",
    )
    return p.parse_args()


def parse_int_list(s):
    return tuple(int(v.strip()) for v in str(s).split(",") if v.strip())


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def cfg_get(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def arg_or_cfg(arg_value, cfg, keys, default):
    return arg_value if arg_value is not None else cfg_get(cfg, keys, default)


def apply_overrides(cfg, args):
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
    
    # Required for loading precomputed geo targets by sample name.
    cfg["data"]["return_name"] = True

    if args.input_mode == "building":
        cfg["data"]["cars_input"] = False
        cfg["data"]["target_type"] = args.target_type or "DPM"
    else:
        cfg["data"]["cars_input"] = True
        cfg["data"]["target_type"] = args.target_type or "carsDPM"
    if cfg["data"].get("root_dir") is None:
        raise ValueError("Use --data-root or set cfg['data']['root_dir'].")
    return cfg


def resolve_options(args, cfg):
    return {
        "epochs": arg_or_cfg(args.epochs, cfg, ["pretrain", "epochs"], arg_or_cfg(None, cfg, ["train", "epochs"], 200)),
        "lr": arg_or_cfg(args.lr, cfg, ["pretrain", "lr"], arg_or_cfg(None, cfg, ["train", "lr"], 1e-4)),
        "weight_decay": arg_or_cfg(args.weight_decay, cfg, ["pretrain", "weight_decay"], 1e-4),
        "warmup_epochs": arg_or_cfg(args.warmup_epochs, cfg, ["pretrain", "warmup_epochs"], 5),
        "min_lr": arg_or_cfg(args.min_lr, cfg, ["pretrain", "min_lr"], 1e-6),
        "seed": arg_or_cfg(args.seed, cfg, ["seed"], 42),
        "batch_size": cfg_get(cfg, ["data", "batch_size"], 32),
        "target_type": cfg_get(cfg, ["data", "target_type"], "DPM"),
        "cars_input": cfg_get(cfg, ["data", "cars_input"], False),
    }


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
            targets = target_builder(x, y, names=names)
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
        targets = target_builder(x, y, names=names)
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
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "metrics": metrics, "options": opts}, save_path)


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


def make_target_builder(args, cfg, target_names, geo_split, device):
    target_type = cfg_get(cfg, ["data", "target_type"], "DPM")
    geo_mode_name = args.geo_mode_name or f"{args.input_mode}_{target_type}"

    return PhysicsTargetBuilder(
        target_names=target_names,
        field_mode=args.field_mode,
        gaussian_sigma=args.gaussian_sigma,
        tx_channel=args.tx_channel,
        building_threshold=args.building_threshold,
        obstacle_channels=parse_int_list(args.obstacle_channels),
        ray_stride=args.ray_stride,
        radiodiff_pathloss_trunc=args.radiodiff_pathloss_trunc,
        radiodiff_pathloss_max=args.radiodiff_pathloss_max,
        radiodiff_source_power_dbm=args.radiodiff_source_power_dbm,
        radiodiff_h=args.radiodiff_h,
        radiodiff_border_value=args.radiodiff_border_value,
        radiodiff_eps=args.radiodiff_eps,
        radiodiff_smooth_sigma=args.radiodiff_smooth_sigma,
        geo_precompute_root=args.geo_precompute_root,
        geo_mode_name=geo_mode_name,
        geo_split=geo_split,
    ).to(device)


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config_path), args)
    opts = resolve_options(args, cfg)

    raw_target_names = tuple([s.strip() for s in args.physics_targets.split(",") if s.strip()])
    target_names = PhysicsTargetBuilder.normalize_target_names(raw_target_names)
    head_specs = PhysicsTargetBuilder.head_specs(target_names)

    opts["physics_targets"] = target_names
    opts["field_mode"] = args.field_mode
    opts["radiodiff_pathloss_trunc"] = args.radiodiff_pathloss_trunc
    opts["radiodiff_pathloss_max"] = args.radiodiff_pathloss_max
    opts["radiodiff_source_power_dbm"] = args.radiodiff_source_power_dbm
    opts["radiodiff_h"] = args.radiodiff_h
    opts["radiodiff_border_value"] = args.radiodiff_border_value
    opts["radiodiff_eps"] = args.radiodiff_eps
    opts["radiodiff_smooth_sigma"] = args.radiodiff_smooth_sigma
    opts["geo_precompute_root"] = args.geo_precompute_root
    opts["geo_mode_name"] = args.geo_mode_name or f"{args.input_mode}_{cfg['data']['target_type']}"

    set_seed(opts["seed"])
    device = prepare_device(args.cuda)
    use_amp = use_amp_on_device(device, args.amp)

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

    model = HRFormerPhysicsPretrainer(cfg, head_specs=head_specs).to(device)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
        print(f"Loaded checkpoint: {args.resume}")
    summarize_trainable_by_module(model)

    train_target_builder = make_target_builder(args, cfg, target_names, "train", device)
    val_target_builder = make_target_builder(args, cfg, target_names, val_key, device)

    loss_fn = PhysicsPretrainLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["lr"], weight_decay=opts["weight_decay"], betas=(0.9, 0.999))
    scaler = GradScaler("cuda", enabled=use_amp)

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
        train_logs = train_one_epoch(model, train_loader, train_target_builder, loss_fn, optimizer, device, scaler, use_amp, epoch + 1)
        val_logs = evaluate_one_epoch(model, val_loader, val_target_builder, loss_fn, device, use_amp, epoch + 1)
        last_metrics = val_logs
        train_losses.append(train_logs["loss"])
        val_losses.append(val_logs["loss"])
        msg = f"Epoch [{epoch + 1}/{opts['epochs']}] lr={lr:.6e} train_{format_logs(train_logs)} val_{format_logs(val_logs)}"
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        if val_logs["loss"] < best_loss:
            best_loss = val_logs["loss"]
            save_checkpoint(model, optimizer, epoch + 1, val_logs, opts, os.path.join(weight_dir, "best.pth"))
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_logs, opts, os.path.join(weight_dir, f"epoch_{epoch + 1:03d}.pth"))

    save_checkpoint(model, optimizer, opts["epochs"], last_metrics, opts, os.path.join(weight_dir, "last.pth"))
    save_loss_curve(train_losses, val_losses, os.path.join(save_folder, "loss.png"))
    print("\nFinished physics pretraining")
    print(f"Best checkpoint: {os.path.join(weight_dir, 'best.pth')}")
    print(f"Last checkpoint: {os.path.join(weight_dir, 'last.pth')}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
