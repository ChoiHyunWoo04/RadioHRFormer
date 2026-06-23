import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import math
from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler

from datasets.rms_dataset import build_dataloaders
from models.hrformer_regressor import HRFormerRadioMapRegressor, load_physics_pretrained_for_downstream

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

    # Paths
    parser.add_argument("--config-path", type=str, default="./configs/hrformer_radiomapseer.json")
    parser.add_argument("--data-root", type=str, default=None, help="RadioMapSeer root directory.")
    parser.add_argument("--save-root", type=str, default="./save")
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
    parser.add_argument("--num-tx", type=int, default=None)
    parser.add_argument("--thresh", type=float, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--loss", choices=["l1", "mse", "radiomamba"], default=None)

    # Runtime
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    # Checkpoint / evaluation
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--physics-pretrained", type=str, default=None)
    parser.add_argument("--eval-split", choices=["val", "valid", "test"], default="val")
    parser.add_argument("--save-every", type=int, default=10)

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


def apply_radiomapseer_overrides(cfg, args, return_name=False):
    """Apply CLI overrides to cfg['data'] so rms_dataset.build_dataloaders(cfg) is the only loader path."""
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

    # Safe defaults for a minimal config.
    cfg["data"].setdefault("root_dir", None)
    cfg["data"].setdefault("num_tx", 80)
    cfg["data"].setdefault("thresh", 0.0)
    cfg["data"].setdefault("batch_size", 32)
    cfg["data"].setdefault("num_workers", 4)

    if cfg["data"]["root_dir"] is None:
        raise ValueError("RadioMapSeer path is required. Use --data-root or cfg['data']['root_dir'].")

    return cfg


def resolve_options(args, cfg):
    opts = {
        "data_root": cfg_get(cfg, ["data", "root_dir"], None),
        "input_mode": args.input_mode,
        "cars_input": cfg_get(cfg, ["data", "cars_input"], False),
        "target_type": cfg_get(cfg, ["data", "target_type"], "DPM"),
        "num_tx": cfg_get(cfg, ["data", "num_tx"], 80),
        "thresh": cfg_get(cfg, ["data", "thresh"], 0.0),
        "batch_size": cfg_get(cfg, ["data", "batch_size"], 32),
        "num_workers": cfg_get(cfg, ["data", "num_workers"], 4),
        "epochs": arg_or_cfg(args.epochs, cfg, ["train", "epochs"], 200),
        "lr": arg_or_cfg(args.lr, cfg, ["train", "lr"], 1e-4),
        "weight_decay": arg_or_cfg(args.weight_decay, cfg, ["train", "weight_decay"], 1e-4),
        "warmup_epochs": arg_or_cfg(args.warmup_epochs, cfg, ["train", "warmup_epochs"], 5),
        "min_lr": arg_or_cfg(args.min_lr, cfg, ["train", "min_lr"], 1e-6),
        "loss": arg_or_cfg(args.loss, cfg, ["train", "loss"], "radiomamba"),
        "seed": arg_or_cfg(args.seed, cfg, ["seed"], 42),
    }
    return opts


def get_split_dict(cfg):
    """Requires the updated rms_dataset.build_dataloaders(cfg, return_datasets=True)."""
    try:
        return build_dataloaders(cfg, return_datasets=True)
    except TypeError as exc:
        raise TypeError(
            "rms_dataset.build_dataloaders must accept return_datasets=True. "
            "Update rms_dataset.py with the provided version."
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
    pred_all = []
    gt_all = []

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

        pred_all.append(pred.detach().cpu())
        gt_all.append(y.detach().cpu())

        pbar.set_postfix({
            "loss": f"{total_loss / total_seen:.5f}",
        })

    pred_all = torch.cat(pred_all, dim=0).float()
    gt_all = torch.cat(gt_all, dim=0).float()

    avg_loss = total_loss / total_seen
    avg_mae = float(MAE(gt_all, pred_all))

    return avg_loss, avg_mae


@torch.no_grad()
def evaluate_one_epoch(model, loader, loss_fn, device, use_amp=False, epoch=None, split_name="Valid"):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    pred_all = []
    gt_all = []

    desc = split_name if epoch is None else f"{split_name} Epoch {epoch}"
    pbar = tqdm(loader, desc=desc, leave=False)
    amp_device_type = get_amp_device_type(device)
    
    for batch in pbar:
        x, y, _ = unpack_batch(batch, device)

        with autocast(device_type=amp_device_type, enabled=use_amp):
            pred = model(x)
            loss = loss_fn(pred, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_seen += bs

        pred_all.append(pred.detach().cpu())
        gt_all.append(y.detach().cpu())

        pbar.set_postfix({
            "loss": f"{total_loss / total_seen:.5f}",
        })

    pred_all = torch.cat(pred_all, dim=0).float()
    gt_all = torch.cat(gt_all, dim=0).float()

    return {
        "loss": total_loss / total_seen,
        "MAE": float(MAE(gt_all, pred_all)),
        "RMSE": float(compute_rmse(pred_all, gt_all)),
        "NMSE": float(compute_nmse(pred_all, gt_all)),
        "PSNR": float(compute_psnr(pred_all, gt_all)),
        "SSIM": float(compute_ssim(pred_all, gt_all)),
    }


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


def save_checkpoint(model, optimizer, epoch, metrics, opts, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "options": opts,
        },
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
    model.load_state_dict(state_dict, strict=True)


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    cfg = apply_radiomapseer_overrides(cfg, args, return_name=False)
    opts = resolve_options(args, cfg)

    set_seed(opts["seed"])
    device = prepare_device(args.cuda)
    use_amp = use_amp_on_device(device, args.amp)

    split_dict = get_split_dict(cfg)
    train_loader = split_dict["train_loader"]
    train_dataset = split_dict["train_dataset"]
    val_loader, val_dataset = get_eval_loader(split_dict, args.eval_split)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(args.save_root, run_name)
    weight_dir = os.path.join(save_folder, "weight")
    os.makedirs(weight_dir, exist_ok=True)
    log_path = os.path.join(save_folder, "log.txt")

    print(f"TRAIN_SIZE: {len(train_dataset)}")
    print(f"VAL_SIZE : {len(val_dataset)}")
    print(f"Input mode: {opts['input_mode']}")
    print(f"Target    : {opts['target_type']}")

    model = HRFormerRadioMapRegressor(cfg).to(device)
    
    if args.resume is not None and args.physics_pretrained is not None:
        raise ValueError(
            "Use either --resume or --physics-pretrained, not both. "
            "--resume is for continuing a downstream run, while --physics-pretrained "
            "is for initializing a new downstream fine-tuning run."
        )

    if args.physics_pretrained is not None:
        load_physics_pretrained_for_downstream(
            model=model,
            ckpt_path=args.physics_pretrained,
            device=device,
            verbose=True,
        )
        print(f"Loaded physics-pretrained initialization: {args.physics_pretrained}")

    if args.resume is not None:
        load_model_state(model, args.resume, device)
        print(f"Loaded checkpoint: {args.resume}")

    summarize_trainable_by_module(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opts["lr"],
        weight_decay=opts["weight_decay"],
        betas=(0.9, 0.999),
    )
    loss_fn = build_loss(opts["loss"]).to(device)
    scaler = GradScaler("cuda", enabled=use_amp)

    train_losses = []
    val_losses = []
    best_mae = float("inf")
    last_metrics = None

    with open(log_path, "a") as f:
        f.write("model: HRFormerRadioMapRegressor\n")
        f.write(f"config_path: {args.config_path}\n")
        f.write(f"data_root: {opts['data_root']}\n")
        f.write(f"input_mode: {opts['input_mode']}\n")
        f.write(f"cars_input: {opts['cars_input']}\n")
        f.write(f"target_type: {opts['target_type']}\n")
        f.write(f"num_tx: {opts['num_tx']}\n")
        f.write(f"batch_size: {opts['batch_size']}\n")
        f.write(f"epochs: {opts['epochs']}\n")
        f.write(f"lr: {opts['lr']}\n")
        f.write(f"weight_decay: {opts['weight_decay']}\n")
        f.write(f"loss: {opts['loss']}\n\n")

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
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        if val_metrics["MAE"] < best_mae:
            best_mae = val_metrics["MAE"]
            save_checkpoint(model, optimizer, epoch + 1, val_metrics, opts, os.path.join(weight_dir, "best.pth"))

        #if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
        #    save_checkpoint(model, optimizer, epoch + 1, val_metrics, opts, os.path.join(weight_dir, f"epoch_{epoch + 1:03d}.pth"))

    save_checkpoint(model, optimizer, opts["epochs"], last_metrics, opts, os.path.join(weight_dir, "last.pth"))
    save_loss_curve(train_losses, val_losses, os.path.join(save_folder, "loss.png"))

    print("\nFinished training")
    print(f"Best MAE checkpoint: {os.path.join(weight_dir, 'best.pth')}")
    print(f"Last checkpoint    : {os.path.join(weight_dir, 'last.pth')}")
    print(f"Log                : {log_path}")


if __name__ == "__main__":
    main()
