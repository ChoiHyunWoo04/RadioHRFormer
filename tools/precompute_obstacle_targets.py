import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import torch

from datasets.rms_dataset import build_dataloaders
from utils import set_seed, prepare_device


# Every saved .pt contains only:
#   obstacle_saturating_a007
#
# All settings are read from the shared JSON config.
# Required CLI argument: --config-path ./configs/hrt.json


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Precompute RadioMapSeer obstacle_sum and "
            "obstacle_saturating_a005 using the shared JSON config."
        )
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/hrt.json",
        help="Shared HRFormer JSON config.",
    )
    return parser.parse_args()


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as file:
        return json.load(file)


def infer_cars_input(target_type):
    """Infer the RadioMapSeer input mode directly from target_type."""
    target_type = str(target_type)
    if target_type == "DPM":
        return False
    if target_type == "carsDPM":
        return True
    raise ValueError(
        f"Unsupported target_type='{target_type}'. Expected 'DPM' or 'carsDPM'."
    )


def infer_mode_name(target_type):
    prefix = "cars" if infer_cars_input(target_type) else "building"
    return f"{prefix}_{target_type}"


def resolve_device(cfg):
    runtime_cfg = cfg.get("runtime", {})
    gpu_ids = runtime_cfg.get("gpus", [0])
    if isinstance(gpu_ids, (int, str)):
        gpu_ids = [gpu_ids]

    if torch.cuda.is_available() and gpu_ids:
        return prepare_device(str(int(gpu_ids[0])))
    return torch.device("cpu")


def prepare_precompute_config(cfg):
    """Prepare the dataloader and resolve all precompute settings from cfg.

    Optional config block:
        "precompute_obstacle": {
          "save_root": "...",          # defaults to physics.geo_precompute_root
          "splits": ["train", "val", "test"],
          "batch_size": 1,
          "num_workers": 0,
          "pin_memory": false,
          "persistent_workers": false,
          "dtype": "float16",
          "overwrite": false,
          "mode_name": null              # defaults to inferred cars_carsDPM / building_DPM
        }
    """
    cfg.setdefault("data", {})
    cfg.setdefault("physics", {})
    precompute_cfg = cfg.setdefault("precompute_obstacle", {})

    data_cfg = cfg["data"]
    physics_cfg = cfg["physics"]

    if not data_cfg.get("root_dir"):
        raise ValueError("cfg['data']['root_dir'] must be set.")

    target_type = str(data_cfg.get("target_type", "DPM"))
    # input-mode is intentionally not an argument: cars_input is inferred from
    # the RadioMapSeer target type.
    data_cfg["cars_input"] = infer_cars_input(target_type)
    data_cfg["return_name"] = True

    # Ray traversal is per sample, so a tiny loader is appropriate.
    data_cfg["batch_size"] = int(precompute_cfg.get("batch_size", 1))
    data_cfg["num_workers"] = int(precompute_cfg.get("num_workers", 0))
    data_cfg["pin_memory"] = bool(precompute_cfg.get("pin_memory", False))
    data_cfg["persistent_workers"] = (
        bool(precompute_cfg.get("persistent_workers", False))
        and data_cfg["num_workers"] > 0
    )

    save_root = precompute_cfg.get(
        "save_root",
        physics_cfg.get("geo_precompute_root"),
    )
    if not save_root:
        raise ValueError(
            "Set cfg['physics']['geo_precompute_root'] or "
            "cfg['precompute_obstacle']['save_root']."
        )

    mode_name = (
        precompute_cfg.get("mode_name")
        or physics_cfg.get("geo_mode_name")
        or infer_mode_name(target_type)
    )

    splits = precompute_cfg.get("splits", ["train", "val", "test"])
    if isinstance(splits, str):
        splits = [item.strip() for item in splits.split(",") if item.strip()]
    splits = ["val" if str(split) == "valid" else str(split) for split in splits]

    invalid_splits = [split for split in splits if split not in {"train", "val", "test"}]
    if invalid_splits:
        raise ValueError(
            f"Invalid splits: {invalid_splits}. Valid: ['train', 'val', 'test']."
        )

    dtype_name = str(precompute_cfg.get("dtype", "float16"))
    if dtype_name not in {"float16", "float32"}:
        raise ValueError("precompute_obstacle.dtype must be 'float16' or 'float32'.")

    obstacle_channels = physics_cfg.get("obstacle_channels", [0, 1])
    if isinstance(obstacle_channels, str):
        obstacle_channels = [
            int(value.strip())
            for value in obstacle_channels.split(",")
            if value.strip()
        ]
    obstacle_channels = tuple(int(channel) for channel in obstacle_channels)
    if not obstacle_channels:
        raise ValueError("physics.obstacle_channels cannot be empty.")

    return {
        "save_root": str(save_root),
        "mode_name": str(mode_name),
        "splits": splits,
        "dtype_name": dtype_name,
        "overwrite": bool(precompute_cfg.get("overwrite", False)),
        "tx_channel": int(physics_cfg.get("tx_channel", -1)),
        "obstacle_channels": obstacle_channels,
        "building_threshold": float(physics_cfg.get("building_threshold", 0.5)),
    }


def safe_stem(name):
    name = os.path.basename(str(name))
    if name.endswith(".png"):
        name = name[:-4]
    return name.replace(os.sep, "_").replace(" ", "_")


def select_channel(x, channel_idx):
    """Select [1,H,W] from x=[C,H,W], supporting negative indices."""
    channels = x.size(0)
    if channel_idx < 0:
        channel_idx += channels
    if channel_idx < 0 or channel_idx >= channels:
        raise IndexError(
            f"Invalid channel index {channel_idx} for input with {channels} channels."
        )
    return x[channel_idx:channel_idx + 1]


def build_obstacle_map(x_i, obstacle_channels, threshold):
    """Merge the configured building/car channels into a binary obstacle map."""
    selected = [
        select_channel(x_i, channel).float()
        for channel in obstacle_channels
    ]
    obstacle = torch.stack(selected, dim=0).amax(dim=0)
    return (obstacle > threshold).float()


def tx_center(tx_map):
    """Return Tx center as integer (y, x) from tx_map=[1,H,W]."""
    _, _, width = tx_map.shape
    flat_index = tx_map.flatten().argmax()
    return int(flat_index // width), int(flat_index % width)


def minmax(z, eps=1e-8):
    z_min = z.amin()
    z_max = z.amax()
    return (z - z_min) / (z_max - z_min + eps)


def alpha_to_key(alpha: float) -> str:
    return f"obstacle_saturating_a{int(round(alpha * 100)):03d}"


#OBSTACLE_ALPHAS = (0.05, 0.06, 0.07, 0.08 0.09)
OBSTACLE_ALPHAS = (0.07,)


@torch.no_grad()
def compute_obstacle_targets(
    x_i,
    tx_channel=-1,
    obstacle_channels=(0, 1),
    building_threshold=0.5,
):
    """Compute the only two retained ray-obstruction maps for one sample.

    obstacle_sum:
        Per-sample min-max-normalized obstruction length.

    obstacle_saturating_a005:
        1 - exp(-0.05 * obstruction length), already in [0, 1].
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

    obstacle_sum_raw = torch.zeros(
        (1, height, width), device=device, dtype=torch.float32
    )

    obstacle_transmission_maps = {
        alpha: torch.zeros_like(obstacle_sum_raw)
        for alpha in OBSTACLE_ALPHAS
    }

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

            for alpha, target_map in obstacle_transmission_maps.items():
                # precomputed inverted saturating / transmission prior
                target_map[0, y1, x1] = torch.exp(-float(alpha) * hit_length)

    #targets = {
    #    "obstacle_sum": minmax(obstacle_sum_raw).cpu(),
    #}
    targets = {}

    for alpha, target_map in obstacle_transmission_maps.items():
        targets[alpha_to_key(alpha)] = target_map.clamp(0.0, 1.0).cpu()

    return targets


def get_split_loader(split_dict, split):
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
            "Dataset must return sample names. This script sets "
            "cfg['data']['return_name']=True automatically."
        )
    return x, names


def save_manifest(save_base, cfg, options):
    #target_keys = ["obstacle_sum"] + [
    #    alpha_to_key(alpha) for alpha in OBSTACLE_ALPHAS
    #]
    target_keys = [
        alpha_to_key(alpha) for alpha in OBSTACLE_ALPHAS
    ]
    print(f"[INFO] target keys       : {target_keys}")

    manifest = {
        "target_keys": target_keys,
        "saturation_alpha_by_key": {
            alpha_to_key(alpha): alpha
            for alpha in OBSTACLE_ALPHAS
        },
        "stored_obstacle_saturating_semantics": "exp(-alpha * obstruction_length), i.e., inverted/transmission prior",
        "dtype": options["dtype_name"],
        "input_mode_inferred_from_target_type": (
            "cars" if cfg["data"]["cars_input"] else "building"
        ),
        "target_type": cfg["data"]["target_type"],
        "tx_channel": options["tx_channel"],
        "obstacle_channels": list(options["obstacle_channels"]),
        "building_threshold": options["building_threshold"],
    }
    with open(os.path.join(save_base, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    options = prepare_precompute_config(cfg)

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg)

    save_base = os.path.join(options["save_root"], options["mode_name"])
    os.makedirs(save_base, exist_ok=True)
    save_manifest(save_base, cfg, options)

    split_dict = build_dataloaders(cfg, return_datasets=True)
    save_dtype = torch.float16 if options["dtype_name"] == "float16" else torch.float32

    print(f"[INFO] device            : {device}")
    print(f"[INFO] save_base         : {save_base}")
    print(f"[INFO] splits            : {options['splits']}")
    print(f"[INFO] target_type       : {cfg['data']['target_type']}")
    print(f"[INFO] cars_input        : {cfg['data']['cars_input']} (inferred)")
    print(f"[INFO] obstacle_channels : {options['obstacle_channels']}")
    print(f"[INFO] tx_channel        : {options['tx_channel']}")
    print(f"[INFO] dtype             : {options['dtype_name']}")
    print(f"[INFO] overwrite         : {options['overwrite']}")

    for split in options["splits"]:
        loader = get_split_loader(split_dict, split)
        save_dir = os.path.join(save_base, split)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[INFO] Precomputing split={split}, save_dir={save_dir}")
        saved = 0
        skipped = 0
        progress = tqdm(loader, desc=f"precompute {split}")

        for batch in progress:
            x, names = unpack_batch(batch)

            for index in range(x.size(0)):
                stem = safe_stem(names[index])
                save_path = os.path.join(save_dir, f"{stem}.pt")

                if os.path.exists(save_path) and not options["overwrite"]:
                    skipped += 1
                    continue

                targets = compute_obstacle_targets(
                    x_i=x[index].to(device, non_blocking=True),
                    tx_channel=options["tx_channel"],
                    obstacle_channels=options["obstacle_channels"],
                    building_threshold=options["building_threshold"],
                )
                targets = {
                    key: value.to(dtype=save_dtype)
                    for key, value in targets.items()
                }

                torch.save(targets, save_path)
                saved += 1

            progress.set_postfix({"saved": saved, "skipped": skipped})

        print(f"[DONE] split={split} saved={saved}, skipped={skipped}")

    print("\nFinished precomputing ray-obstruction targets.")


if __name__ == "__main__":
    main()
