import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch

from datasets.rms_dataset import build_dataloaders
from datasets.physics_targets import PhysicsTargetBuilder
from utils import set_seed, prepare_device


# Only grad/lap are retained as optional label-driven ablation targets.
ONLINE_LABEL_TARGETS = {"grad", "lap", "singularity"}

# Input-driven maps must be loaded from precomputed .pt files.
PRECOMPUTED_INPUT_TARGETS = {
    "obstacle_sum",
    "obstacle_saturating_a003",
    "obstacle_saturating_a005",
    "radial_gain",
    "corner_diffraction",
}

SUPPORTED_TARGETS = ONLINE_LABEL_TARGETS | PRECOMPUTED_INPUT_TARGETS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize RadioMapSeer pretraining targets. grad/lap/singularity are optionally "
            "generated online; input-driven maps are loaded from precomputed .pt files."
        )
    )

    # Paths
    parser.add_argument("--config-path", type=str, default="./configs/hrformer_radiomapseer.json")
    parser.add_argument("--data-root", type=str, default=None, help="RadioMapSeer root directory.")
    parser.add_argument("--save-dir", type=str, default="./save/visual")
    parser.add_argument(
        "--precompute-root",
        "--geo-precompute-root",
        dest="precompute_root",
        type=str,
        default="./data/precomputed_input_driven",
        help=(
            "Root directory of precomputed input-driven targets. Expected layout: "
            "<root>/<mode_name>/<split>/<sample_name>.pt"
        ),
    )
    parser.add_argument(
        "--precompute-mode-name",
        "--geo-mode-name",
        dest="precompute_mode_name",
        type=str,
        default=None,
        help=(
            "Optional mode folder name below --precompute-root. If omitted, "
            "uses '<input_mode>_<target_type>', e.g., cars_carsDPM."
        ),
    )

    # RadioMapSeer mode
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
    parser.add_argument("--split", choices=["train", "val", "valid", "test"], default="train")

    # Sampling
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=None)

    # Targets
    parser.add_argument(
        "--physics-targets",
        type=str,
        default=(
            "grad,lap,singularity,obstacle_sum,obstacle_saturating_a003,"
            "obstacle_saturating_a005,radial_gain,corner_diffraction"
        ),
        help=(
            "Comma-separated targets. Supported online ablation targets: grad,lap,singularity. "
            "Supported precomputed input-driven targets: obstacle_sum,"
            "obstacle_saturating_a003,obstacle_saturating_a005,"
            "radial_gain,corner_diffraction."
        ),
    )

    # Only used if grad/lap are requested.
    parser.add_argument(
        "--field-mode",
        choices=["normalized_power", "db_power", "pathloss_db"],
        default="normalized_power",
    )
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--tx-channel", type=int, default=-1)
    parser.add_argument(
        "--no-normalize-each-sample",
        action="store_true",
        help="Disable sample-wise standardization for online grad/lap maps.",
    )
    parser.add_argument("--radiodiff-pathloss-trunc", type=float, default=-147.0)
    parser.add_argument("--radiodiff-pathloss-max", type=float, default=-47.0)
    parser.add_argument("--radiodiff-source-power-dbm", type=float, default=23.0)
    parser.add_argument("--radiodiff-h", type=float, default=1.0)
    parser.add_argument("--radiodiff-border-value", type=float, default=1.0)
    parser.add_argument("--radiodiff-eps", type=float, default=1e-30)
    parser.add_argument("--radiodiff-smooth-sigma", type=float, default=0.9)

    # Figure
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Also save one PNG per input/label/target map.",
    )

    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def apply_radiomapseer_overrides(cfg, args):
    cfg.setdefault("data", {})

    if args.data_root is not None:
        cfg["data"]["root_dir"] = args.data_root
    if args.num_tx is not None:
        cfg["data"]["num_tx"] = args.num_tx
    if args.thresh is not None:
        cfg["data"]["thresh"] = args.thresh

    cfg["data"]["batch_size"] = args.batch_size
    cfg["data"]["num_workers"] = args.num_workers
    cfg["data"]["pin_memory"] = False
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["return_name"] = True

    if args.input_mode == "building":
        cfg["data"]["cars_input"] = False
        cfg["data"]["target_type"] = args.target_type or "DPM"
    else:
        cfg["data"]["cars_input"] = True
        cfg["data"]["target_type"] = args.target_type or "carsDPM"

    if cfg["data"].get("root_dir") is None:
        raise ValueError(
            "RadioMapSeer root is required. Use --data-root or set cfg['data']['root_dir']."
        )

    return cfg


def get_split_loader(cfg, split):
    try:
        split_dict = build_dataloaders(cfg, return_datasets=True)
    except TypeError as exc:
        raise TypeError(
            "rms_dataset.build_dataloaders must accept return_datasets=True."
        ) from exc

    split = "val" if split == "valid" else split
    key = f"{split}_loader"
    if key not in split_dict:
        raise KeyError(f"Missing {key}. Available keys: {list(split_dict.keys())}")
    return split_dict[key]


def unpack_batch(batch):
    if len(batch) == 3:
        x, y, names = batch
    elif len(batch) == 2:
        x, y = batch
        names = None
    else:
        raise ValueError(f"Unexpected batch format with length {len(batch)}")
    return x, y, names


def parse_target_names(text):
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one target must be provided.")

    names = list(dict.fromkeys(names))
    invalid = [name for name in names if name not in SUPPORTED_TARGETS]
    if invalid:
        raise ValueError(
            f"Unsupported target(s): {invalid}. Supported: {sorted(SUPPORTED_TARGETS)}"
        )
    return names


def safe_stem(name):
    name = str(name)
    name = os.path.basename(name)
    if name.endswith(".png"):
        name = name[:-4]
    return name.replace(os.sep, "_").replace(" ", "_")


def load_precomputed_targets(
    names,
    precompute_root,
    mode_name,
    split,
    requested_targets,
    device,
):
    requested_targets = [
        target for target in requested_targets if target in PRECOMPUTED_INPUT_TARGETS
    ]
    if not requested_targets:
        return {}

    if names is None:
        raise ValueError(
            "Dataset must return sample names to load precomputed target files."
        )

    split = "val" if split == "valid" else split
    base_dir = os.path.join(precompute_root, mode_name, split)

    loaded = {target: [] for target in requested_targets}

    for name in names:
        path = os.path.join(base_dir, f"{safe_stem(name)}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing precomputed target file: {path}\n"
                "Check --precompute-root, --precompute-mode-name, --split, and sample names."
            )

        data = torch.load(path, map_location="cpu")
        for target in requested_targets:
            if target not in data:
                raise KeyError(
                    f"Key '{target}' not found in {path}. "
                    f"Available keys: {list(data.keys())}"
                )
            loaded[target].append(data[target].float())

    return {
        target: torch.stack(values, dim=0).to(device=device)
        for target, values in loaded.items()
    }


def to_numpy_img(tensor):
    if tensor.ndim == 3:
        tensor = tensor[0]
    return tensor.detach().float().cpu().numpy()


def minmax_for_display(tensor, eps=1e-8):
    z = tensor.detach().float()
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    zmin = z.amin()
    zmax = z.amax()
    if float((zmax - zmin).abs()) < eps:
        return torch.zeros_like(z)
    return (z - zmin) / (zmax - zmin + eps)


def infer_channel_titles(x, input_mode):
    c = x.shape[0]
    if input_mode == "building":
        base = ["input ch0: building", "input ch1: building", "input ch2: Tx"]
    else:
        base = ["input ch0: building", "input ch1: cars", "input ch2: Tx"]
    return base[:c] if c <= len(base) else base + [f"input ch{i}" for i in range(len(base), c)]


def pretty_target_name(name):
    aliases = {
        "obstacle_sum": "obstacle: sum",
        "obstacle_saturating_a003": "obstacle: saturating (α=0.03)",
        "obstacle_saturating_a005": "obstacle: saturating (α=0.05)",
        "radial_gain": "radial gain",
        "corner_diffraction": "corner diffraction",
    }
    return aliases.get(name, name)


def save_single_map(path, img, title, cmap="viridis", dpi=180):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    image = plt.imshow(to_numpy_img(minmax_for_display(img)), cmap=cmap)
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_sample_grid(
    save_path,
    x_i,
    y_i,
    targets_i: Dict[str, torch.Tensor],
    input_mode,
    name: Optional[str] = None,
    dpi=180,
):
    panels = []

    titles = infer_channel_titles(x_i, input_mode)
    for channel in range(x_i.shape[0]):
        panels.append((titles[channel], x_i[channel], "gray"))

    panels.append(("label y", y_i[0], "viridis"))

    for target_name, target_map in targets_i.items():
        panels.append((f"physics: {pretty_target_name(target_name)}", target_map[0], "viridis"))

    n = len(panels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.25 * ncols, 3.3 * nrows),
        squeeze=False,
    )

    for ax in axes.flatten():
        ax.axis("off")

    for idx, (title, img, cmap) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        image = ax.imshow(to_numpy_img(minmax_for_display(img)), cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

    if name is not None:
        fig.suptitle(str(name), fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cfg = apply_radiomapseer_overrides(load_config(args.config_path), args)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    device = prepare_device(args.cuda)
    target_names = parse_target_names(args.physics_targets)

    online_names = [name for name in target_names if name in ONLINE_LABEL_TARGETS]
    precomputed_names = [
        name for name in target_names if name in PRECOMPUTED_INPUT_TARGETS
    ]

    mode_name = args.precompute_mode_name
    if mode_name is None:
        mode_name = f"{args.input_mode}_{cfg['data']['target_type']}"

    builder = None
    if online_names:
        builder = PhysicsTargetBuilder(
            target_names=online_names,
            field_mode=args.field_mode,
            gaussian_sigma=args.gaussian_sigma,
            eps=args.eps,
            tx_channel=args.tx_channel,
            normalize_each_sample=not args.no_normalize_each_sample,
            radiodiff_pathloss_trunc=args.radiodiff_pathloss_trunc,
            radiodiff_pathloss_max=args.radiodiff_pathloss_max,
            radiodiff_source_power_dbm=args.radiodiff_source_power_dbm,
            radiodiff_h=args.radiodiff_h,
            radiodiff_border_value=args.radiodiff_border_value,
            radiodiff_eps=args.radiodiff_eps,
            radiodiff_smooth_sigma=args.radiodiff_smooth_sigma,
        ).to(device)
        builder.eval()

    loader = get_split_loader(cfg, args.split)
    os.makedirs(args.save_dir, exist_ok=True)

    saved = 0
    for batch in loader:
        x, y, names = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            targets = {}

            if builder is not None:
                targets.update(builder(x, y))

            if precomputed_names:
                targets.update(
                    load_precomputed_targets(
                        names=names,
                        precompute_root=args.precompute_root,
                        mode_name=mode_name,
                        split=args.split,
                        requested_targets=precomputed_names,
                        device=device,
                    )
                )

        # Preserve user-specified ordering in plots.
        targets = {name: targets[name] for name in target_names}

        for i in range(x.size(0)):
            if saved >= args.num_samples:
                break

            sample_name = names[i] if names is not None else None
            stem = f"sample_{saved:03d}"
            if sample_name is not None:
                stem += f"_{safe_stem(sample_name)}"

            x_i = x[i].detach().cpu()
            y_i = y[i].detach().cpu()
            targets_i = {key: value[i].detach().cpu() for key, value in targets.items()}

            grid_path = os.path.join(args.save_dir, f"{stem}_physics_grid.png")
            save_sample_grid(
                save_path=grid_path,
                x_i=x_i,
                y_i=y_i,
                targets_i=targets_i,
                input_mode=args.input_mode,
                name=sample_name,
                dpi=args.dpi,
            )

            if args.save_individual:
                individual_dir = os.path.join(args.save_dir, stem)
                os.makedirs(individual_dir, exist_ok=True)

                for channel in range(x_i.shape[0]):
                    save_single_map(
                        os.path.join(individual_dir, f"input_ch{channel}.png"),
                        x_i[channel],
                        f"input ch{channel}",
                        cmap="gray",
                        dpi=args.dpi,
                    )

                save_single_map(
                    os.path.join(individual_dir, "label_y.png"),
                    y_i[0],
                    "label y",
                    cmap="viridis",
                    dpi=args.dpi,
                )

                for target_name, target_map in targets_i.items():
                    save_single_map(
                        os.path.join(individual_dir, f"physics_{target_name}.png"),
                        target_map[0],
                        f"physics: {pretty_target_name(target_name)}",
                        cmap="viridis",
                        dpi=args.dpi,
                    )

            print(f"Saved: {grid_path}")
            saved += 1

        if saved >= args.num_samples:
            break

    print(f"Done. Saved {saved} sample grid(s) to: {args.save_dir}")


if __name__ == "__main__":
    main()
