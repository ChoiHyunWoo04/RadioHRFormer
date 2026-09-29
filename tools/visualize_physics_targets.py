import os
import sys
import copy
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.rms_dataset import build_dataloaders
from datasets.physics_targets import PhysicsTargetBuilder
from utils import set_seed


TARGET_NAMES = (
    "radial_gain",
    "los",
    "obstacle_saturating_a007",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a paper-ready 2x2 visualization of the downstream label and "
            "three propagation-aware pretraining targets."
        )
    )
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional override for cfg['visualize']['save_dir'].",
    )
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def cfg_get(cfg, keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def safe_stem(name):
    name = os.path.basename(str(name))
    if name.endswith(".png"):
        name = name[:-4]
    return name.replace(os.sep, "_").replace(" ", "_")


def prepare_visualizer_config(cfg, args):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})
    cfg.setdefault("physics", {})
    cfg.setdefault("visualize", {})

    if cfg["data"].get("root_dir") is None:
        raise ValueError("cfg['data']['root_dir'] must be set.")

    vis_cfg = cfg["visualize"]
    cfg["data"]["batch_size"] = int(vis_cfg.get("batch_size", 1))
    cfg["data"]["num_workers"] = int(
        vis_cfg.get("num_workers", cfg["data"].get("num_workers", 0))
    )
    cfg["data"]["pin_memory"] = bool(
        vis_cfg.get("pin_memory", cfg["data"].get("pin_memory", True))
    )
    cfg["data"]["persistent_workers"] = bool(
        vis_cfg.get(
            "persistent_workers",
            cfg["data"].get("persistent_workers", False),
        )
    ) and cfg["data"]["num_workers"] > 0
    cfg["data"]["return_name"] = True

    if args.save_dir is not None:
        vis_cfg["save_dir"] = args.save_dir

    vis_cfg.setdefault("save_dir", "./save/pretrain_target_visualization")
    vis_cfg.setdefault("split", "test")
    vis_cfg.setdefault("num_samples", 1)
    vis_cfg.setdefault("dpi", 300)
    return cfg


def resolve_device(cfg):
    requested = cfg_get(cfg, ["runtime", "gpus"], [0])
    if requested is None:
        requested = []
    if not isinstance(requested, (list, tuple)):
        raise TypeError("cfg['runtime']['gpus'] must be a list, e.g. [0].")
    if not requested or not torch.cuda.is_available():
        return torch.device("cpu")

    gpu_id = int(requested[0])
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def resolve_options(cfg):
    data_cfg = cfg["data"]
    physics_cfg = cfg["physics"]
    vis_cfg = cfg["visualize"]

    target_type = str(data_cfg.get("target_type", "DPM"))
    split = str(vis_cfg.get("split", "test"))
    split = "val" if split == "valid" else split

    geo_mode_name = physics_cfg.get("geo_mode_name")
    if geo_mode_name is None:
        geo_mode_name = (
            "cars_carsDPM"
            if target_type.lower() == "carsdpm"
            else "building_DPM"
        )

    return {
        "seed": int(cfg.get("seed", 42)),
        "split": split,
        "num_samples": int(vis_cfg.get("num_samples", 1)),
        "save_dir": str(
            vis_cfg.get("save_dir", "./save/pretrain_target_visualization")
        ),
        "dpi": int(vis_cfg.get("dpi", 300)),
        "target_type": target_type,
        "tx_channel": int(physics_cfg.get("tx_channel", -1)),
        "geo_precompute_root": physics_cfg.get("geo_precompute_root"),
        "geo_mode_name": str(geo_mode_name),
        "transmittance_alpha": float(
            physics_cfg.get("transmittance_alpha", 0.07)
        ),
    }


def get_split_loader(cfg, split):
    split_dict = build_dataloaders(cfg, return_datasets=True)
    key = f"{split}_loader"
    if key not in split_dict:
        raise KeyError(f"Missing {key}. Available keys: {list(split_dict.keys())}")
    return split_dict[key]


def unpack_batch(batch):
    if len(batch) == 3:
        return batch
    if len(batch) == 2:
        x, y = batch
        return x, y, None
    raise ValueError(f"Unexpected batch format with length {len(batch)}.")


def to_2d_numpy(tensor):
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected [H,W] or [1,H,W], got {tuple(tensor.shape)}."
        )

    arr = tensor.numpy().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def yellow_rgb(tensor):
    arr = to_2d_numpy(tensor)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = arr
    rgb[..., 1] = arr
    return rgb


def save_individual_map(tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, yellow_rgb(tensor), vmin=0.0, vmax=1.0)


def save_2x2_grid(label, radial_gain, los, obstacle, save_path, dpi=300):
    panels = (label, radial_gain, los, obstacle)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0), squeeze=False)

    for ax, panel in zip(axes.flat, panels):
        ax.imshow(yellow_rgb(panel), interpolation="nearest")
        ax.axis("off")

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        wspace=0.015,
        hspace=0.015,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = prepare_visualizer_config(load_config(args.config_path), args)
    opts = resolve_options(cfg)

    if opts["num_samples"] <= 0:
        raise ValueError("cfg['visualize']['num_samples'] must be positive.")

    if opts["geo_precompute_root"] is None:
        raise ValueError(
            "cfg['physics']['geo_precompute_root'] must be set because "
            "LoS and obstacle_saturating_a007 are loaded from precomputed .pt files."
        )

    device = resolve_device(cfg)
    set_seed(opts["seed"])

    builder = PhysicsTargetBuilder(
        target_names=TARGET_NAMES,
        tx_channel=opts["tx_channel"],
        geo_precompute_root=opts["geo_precompute_root"],
        geo_mode_name=opts["geo_mode_name"],
        geo_split=opts["split"],
        transmittance_alpha=opts["transmittance_alpha"],
    ).to(device)
    builder.eval()

    loader = get_split_loader(cfg, opts["split"])
    os.makedirs(opts["save_dir"], exist_ok=True)

    print(f"[INFO] Device        : {device}")
    print(f"[INFO] Split         : {opts['split']}")
    print(f"[INFO] Target type   : {opts['target_type']}")
    print(f"[INFO] Geo mode      : {opts['geo_mode_name']}")
    print(f"[INFO] Targets       : {TARGET_NAMES}")

    saved = 0

    with torch.no_grad():
        for batch in loader:
            x, y, names = unpack_batch(batch)

            if names is None:
                raise ValueError(
                    "Sample names are required to load precomputed LoS/transmittance maps."
                )

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            targets = builder(x, names=names)

            for i in range(x.size(0)):
                if saved >= opts["num_samples"]:
                    break

                sample_name = safe_stem(names[i])
                sample_dir = os.path.join(
                    opts["save_dir"],
                    f"sample_{saved:03d}_{sample_name}",
                )
                os.makedirs(sample_dir, exist_ok=True)

                label_i = y[i].detach().cpu()
                radial_i = targets["radial_gain"][i].detach().cpu()
                los_i = targets["los"][i].detach().cpu()
                obstacle_i = (
                    targets["obstacle_saturating_a007"][i].detach().cpu()
                )

                save_2x2_grid(
                    label=label_i,
                    radial_gain=radial_i,
                    los=los_i,
                    obstacle=obstacle_i,
                    save_path=os.path.join(
                        sample_dir,
                        "pretrain_targets_2x2.png",
                    ),
                    dpi=opts["dpi"],
                )

                save_individual_map(
                    label_i,
                    os.path.join(sample_dir, "downstream_label.png"),
                )
                save_individual_map(
                    radial_i,
                    os.path.join(sample_dir, "radial_gain.png"),
                )
                save_individual_map(
                    los_i,
                    os.path.join(sample_dir, "los.png"),
                )
                save_individual_map(
                    obstacle_i,
                    os.path.join(
                        sample_dir,
                        "obstacle_saturating_a007.png",
                    ),
                )

                print(f"[SAVED] {sample_dir}")
                saved += 1

            if saved >= opts["num_samples"]:
                break

    print(
        f"[DONE] Saved {saved} visualization sample(s) "
        f"to {opts['save_dir']}."
    )


if __name__ == "__main__":
    main()
