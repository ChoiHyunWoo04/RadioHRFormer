import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from tqdm import tqdm
import torch

from datasets.rms_dataset import build_dataloaders
from utils import set_seed, prepare_device


# Saved target keys in every .pt file:
#   obstacle_sum
#   obstacle_saturating_a003   # alpha = 0.03
#   obstacle_saturating_a005   # alpha = 0.05
#
# No LoS, radial_gain, or corner_diffraction maps are computed or saved.


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline precompute RadioMapSeer ray-obstruction targets: "
            "obstacle_sum, obstacle_saturating_a003, and "
            "obstacle_saturating_a005."
        )
    )

    # Paths
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/hrformer_radiomapseer.json",
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--save-root",
        type=str,
        default="./data/precomputed_obstacle",
    )

    # RadioMapSeer mode
    parser.add_argument("--input-mode", choices=["building", "cars"], default="cars")
    parser.add_argument("--target-type", choices=["DPM", "carsDPM"], default=None)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--num-tx", type=int, default=None)
    parser.add_argument("--thresh", type=float, default=None)

    # Runtime
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")

    # Geometry input interpretation
    parser.add_argument("--tx-channel", type=int, default=-1)
    parser.add_argument(
        "--obstacle-channels",
        type=str,
        default="0,1",
        help=(
            "Comma-separated channels used as obstacles. "
            "For [building, cars, Tx], use 0,1."
        ),
    )
    parser.add_argument("--building-threshold", type=float, default=0.5)

    # Fixed saturation variants retained for ablation.
    parser.add_argument(
        "--obstacle-alphas",
        type=str,
        default="0.03,0.05",
        help=(
            "Must contain exactly 0.03 and 0.05. These are saved as "
            "obstacle_saturating_a003 and obstacle_saturating_a005."
        ),
    )

    return parser.parse_args()


def load_config(path):
    with open(path, "r") as file:
        return json.load(file)


def apply_overrides(cfg, args):
    cfg.setdefault("data", {})

    cfg["data"]["root_dir"] = args.data_root
    cfg["data"]["batch_size"] = args.batch_size
    cfg["data"]["num_workers"] = args.num_workers
    cfg["data"]["pin_memory"] = False
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["return_name"] = True

    if args.num_tx is not None:
        cfg["data"]["num_tx"] = args.num_tx
    if args.thresh is not None:
        cfg["data"]["thresh"] = args.thresh

    if args.input_mode == "building":
        cfg["data"]["cars_input"] = False
        cfg["data"]["target_type"] = args.target_type or "DPM"
    else:
        cfg["data"]["cars_input"] = True
        cfg["data"]["target_type"] = args.target_type or "carsDPM"

    return cfg


def parse_int_list(text):
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("--obstacle-channels must contain at least one channel.")
    return tuple(values)


def parse_fixed_alphas(text):
    values = {round(float(value.strip()), 8) for value in text.split(",") if value.strip()}
    required = {0.03, 0.05}
    if values != required:
        raise ValueError(
            "--obstacle-alphas must contain exactly '0.03,0.05' because this "
            "script saves the fixed ablation keys "
            "'obstacle_saturating_a003' and 'obstacle_saturating_a005'."
        )
    return (0.03, 0.05)


def safe_stem(name):
    name = os.path.basename(str(name))
    if name.endswith(".png"):
        name = name[:-4]
    return name.replace(os.sep, "_").replace(" ", "_")


def select_channel(x, channel_idx):
    """Select one [1,H,W] channel from x=[C,H,W], supporting negative indices."""
    channels = x.size(0)
    if channel_idx < 0:
        channel_idx += channels

    if channel_idx < 0 or channel_idx >= channels:
        raise IndexError(
            f"Invalid channel index {channel_idx} for input with {channels} channels."
        )

    return x[channel_idx:channel_idx + 1]


def build_obstacle_map(x_i, obstacle_channels, threshold):
    """Merge selected input channels into a binary obstacle map [1,H,W]."""
    selected = [select_channel(x_i, channel).float() for channel in obstacle_channels]
    obstacle = torch.stack(selected, dim=0).amax(dim=0)
    return (obstacle > threshold).float()


def tx_center(tx_map):
    """Return the Tx pixel center as integer (y, x) from tx_map=[1,H,W]."""
    _, _, width = tx_map.shape
    flat_index = tx_map.flatten().argmax()
    return int(flat_index // width), int(flat_index % width)


def minmax(z, eps=1e-8):
    z_min = z.amin()
    z_max = z.amax()
    return (z - z_min) / (z_max - z_min + eps)


@torch.no_grad()
def compute_obstacle_targets(
    x_i,
    tx_channel=-1,
    obstacle_channels=(0, 1),
    building_threshold=0.5,
):
    """Compute all retained ray-obstruction targets for one sample.

    Returns:
        obstacle_sum:
            Ray obstacle-intersection length, then sample-wise min-max normalized.

        obstacle_saturating_a003:
            1 - exp(-0.03 * ray obstacle-intersection length).

        obstacle_saturating_a005:
            1 - exp(-0.05 * ray obstacle-intersection length).

    The two saturating targets retain their natural [0,1] scale. They are not
    min-max normalized, so their alpha-dependent difference is preserved.
    """
    x_i = x_i.float()
    obstacle_mask = build_obstacle_map(
        x_i,
        obstacle_channels=obstacle_channels,
        threshold=building_threshold,
    )
    tx_map = select_channel(x_i, tx_channel).float()

    _, height, width = obstacle_mask.shape
    device = x_i.device
    tx_y, tx_x = tx_center(tx_map)

    obstacle_sum_raw = torch.zeros((1, height, width), device=device, dtype=torch.float32)
    obstacle_saturating_a003 = torch.zeros_like(obstacle_sum_raw)
    obstacle_saturating_a005 = torch.zeros_like(obstacle_sum_raw)

    # One ray traversal produces every requested target.
    for y1 in range(height):
        dy = y1 - tx_y

        for x1 in range(width):
            dx = x1 - tx_x
            num_points = max(abs(dx), abs(dy), 1) + 1

            rows = (
                torch.linspace(tx_y, y1, num_points, device=device)
                .round()
                .long()
                .clamp(0, height - 1)
            )
            cols = (
                torch.linspace(tx_x, x1, num_points, device=device)
                .round()
                .long()
                .clamp(0, width - 1)
            )

            hit_length = obstacle_mask[0, rows, cols].sum()

            obstacle_sum_raw[0, y1, x1] = hit_length
            obstacle_saturating_a003[0, y1, x1] = 1.0 - torch.exp(-0.03 * hit_length)
            obstacle_saturating_a005[0, y1, x1] = 1.0 - torch.exp(-0.05 * hit_length)

    return {
        "obstacle_sum": minmax(obstacle_sum_raw).cpu(),
        "obstacle_saturating_a003": obstacle_saturating_a003.clamp(0.0, 1.0).cpu(),
        "obstacle_saturating_a005": obstacle_saturating_a005.clamp(0.0, 1.0).cpu(),
    }


def get_split_loader(split_dict, split):
    split = "val" if split == "valid" else split
    key = f"{split}_loader"

    if key not in split_dict:
        raise KeyError(f"Missing {key}. Available keys: {list(split_dict.keys())}")

    return split_dict[key]


def unpack_batch(batch):
    if len(batch) == 3:
        x, _, names = batch
    elif len(batch) == 2:
        x, _ = batch
        names = None
    else:
        raise ValueError(f"Unexpected batch format: len={len(batch)}")

    if names is None:
        raise ValueError(
            "Dataset must return sample names. Set cfg['data']['return_name']=True."
        )

    return x, names


def save_manifest(save_base, args, cfg):
    manifest = {
        "target_keys": [
            "obstacle_sum",
            "obstacle_saturating_a003",
            "obstacle_saturating_a005",
        ],
        "saturation_alpha_by_key": {
            "obstacle_saturating_a003": 0.03,
            "obstacle_saturating_a005": 0.05,
        },
        "dtype": args.dtype,
        "input_mode": args.input_mode,
        "target_type": cfg["data"]["target_type"],
        "tx_channel": args.tx_channel,
        "obstacle_channels": list(parse_int_list(args.obstacle_channels)),
        "building_threshold": args.building_threshold,
    }

    with open(os.path.join(save_base, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = apply_overrides(load_config(args.config_path), args)
    device = prepare_device(args.cuda)

    obstacle_channels = parse_int_list(args.obstacle_channels)
    parse_fixed_alphas(args.obstacle_alphas)

    mode_name = f"{args.input_mode}_{cfg['data']['target_type']}"
    save_base = os.path.join(args.save_root, mode_name)
    os.makedirs(save_base, exist_ok=True)

    save_manifest(save_base, args, cfg)
    split_dict = build_dataloaders(cfg, return_datasets=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    print(f"[INFO] save_base         : {save_base}")
    print(f"[INFO] splits            : {splits}")
    print("[INFO] target keys       : obstacle_sum, obstacle_saturating_a003, obstacle_saturating_a005")
    print(f"[INFO] obstacle_channels : {obstacle_channels}")
    print(f"[INFO] tx_channel        : {args.tx_channel}")
    print(f"[INFO] dtype             : {args.dtype}")

    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    for split in splits:
        loader = get_split_loader(split_dict, split)
        split_key = "val" if split == "valid" else split
        save_dir = os.path.join(save_base, split_key)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[INFO] Precomputing split={split_key}, save_dir={save_dir}")

        saved = 0
        skipped = 0
        progress = tqdm(loader, desc=f"precompute {split_key}")

        for batch in progress:
            x, names = unpack_batch(batch)

            for index in range(x.size(0)):
                stem = safe_stem(names[index])
                save_path = os.path.join(save_dir, f"{stem}.pt")

                if os.path.exists(save_path) and not args.overwrite:
                    skipped += 1
                    continue

                targets = compute_obstacle_targets(
                    x_i=x[index].to(device, non_blocking=True),
                    tx_channel=args.tx_channel,
                    obstacle_channels=obstacle_channels,
                    building_threshold=args.building_threshold,
                )
                targets = {
                    key: value.to(dtype=save_dtype)
                    for key, value in targets.items()
                }

                torch.save(targets, save_path)
                saved += 1

            progress.set_postfix({"saved": saved, "skipped": skipped})

        print(f"[DONE] split={split_key} saved={saved}, skipped={skipped}")

    print("\nFinished precomputing ray-obstruction targets.")


if __name__ == "__main__":
    main()
