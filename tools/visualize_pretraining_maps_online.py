import os
import sys
import math
import argparse
import json
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from datasets.rms_dataset import build_dataloaders
from datasets.physics_targets import PhysicsTargetBuilder
from utils import set_seed, prepare_device


# -----------------------------------------------------------------------------
# Online visualizer for all pretraining targets.
#
# Label-driven maps, generated from the radio-map label y:
#   grad, lap, singularity
#
# Input-driven maps, generated on-the-fly from input geometry x:
#   obstacle_sum
#   obstacle_saturating_a003
#   obstacle_saturating_a005
#   radial_gain
#   corner_diffraction
#
# This script intentionally does NOT read any precomputed .pt target files.
# It uses the same target definitions as the offline precompute script, so it
# is appropriate for inspecting target quality before precomputing the dataset.
# -----------------------------------------------------------------------------


ONLINE_LABEL_TARGETS = {"grad", "lap", "singularity"}
ONLINE_INPUT_TARGETS = {
    "obstacle_sum",
    "obstacle_saturating_a003",
    "obstacle_saturating_a005",
    "radial_gain",
    "corner_diffraction",
}
SUPPORTED_TARGETS = ONLINE_LABEL_TARGETS | ONLINE_INPUT_TARGETS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize all RadioMapSeer pretraining maps online without loading "
            "precomputed .pt files."
        )
    )

    # Dataset / runtime
    parser.add_argument("--config-path", type=str, default="./configs/hrformer_radiomapseer.json")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./save/visual_online")
    parser.add_argument(
        "--input-mode",
        choices=["building", "cars"],
        default="cars",
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

    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use 1 for online ray-based maps to avoid unnecessary waiting.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=None)

    # Target selection
    parser.add_argument(
        "--physics-targets",
        type=str,
        default=(
            "grad,lap,singularity,obstacle_sum,obstacle_saturating_a003,"
            "obstacle_saturating_a005,radial_gain,corner_diffraction"
        ),
        help=(
            "Comma-separated targets. Supported: grad,lap,singularity,"
            "obstacle_sum,obstacle_saturating_a003,obstacle_saturating_a005,"
            "radial_gain,corner_diffraction."
        ),
    )

    # Label-driven target options
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

    # Singularity / RadioDiff-k2-like target options
    parser.add_argument("--radiodiff-pathloss-trunc", type=float, default=-147.0)
    parser.add_argument("--radiodiff-pathloss-max", type=float, default=-47.0)
    parser.add_argument("--radiodiff-source-power-dbm", type=float, default=23.0)
    parser.add_argument("--radiodiff-h", type=float, default=1.0)
    parser.add_argument("--radiodiff-border-value", type=float, default=1.0)
    parser.add_argument("--radiodiff-eps", type=float, default=1e-30)
    parser.add_argument("--radiodiff-smooth-sigma", type=float, default=0.9)

    # Input-driven geometry options
    parser.add_argument(
        "--obstacle-channels",
        type=str,
        default="0,1",
        help="Comma-separated channels used as obstacles. For [building,cars,Tx], use 0,1.",
    )
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument(
        "--obstacle-alphas",
        type=str,
        default="0.03,0.05",
        help=(
            "Alpha values for saturation maps. The current visualization supports "
            "0.03 and/or 0.05 because their target names are fixed."
        ),
    )
    parser.add_argument("--corner-sigma", type=float, default=3.0)
    parser.add_argument("--corner-posthit-decay", type=float, default=0.03)
    parser.add_argument("--corner-max-corners", type=int, default=128)
    parser.add_argument("--corner-response-threshold", type=float, default=0.05)
    parser.add_argument("--corner-nms-radius", type=int, default=2)
    parser.add_argument("--corner-harris-k", type=float, default=0.04)

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


def parse_int_list(text):
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def parse_float_list(text):
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("--obstacle-alphas must contain at least one alpha.")
    if any(value <= 0 for value in values):
        raise ValueError(f"All --obstacle-alphas must be positive. Got: {values}")
    return tuple(dict.fromkeys(values))


def safe_stem(name):
    name = str(name)
    name = os.path.basename(name)
    if name.endswith(".png"):
        name = name[:-4]
    return name.replace(os.sep, "_").replace(" ", "_")


def select_channel(x, channel_idx):
    """Select one channel from x=[C,H,W], supporting negative indices."""
    c = x.size(0)
    if channel_idx < 0:
        channel_idx = c + channel_idx
    if channel_idx < 0 or channel_idx >= c:
        raise IndexError(f"Invalid channel index {channel_idx} for input with {c} channels.")
    return x[channel_idx:channel_idx + 1]


def build_obstacle_map(x_i, obstacle_channels, threshold):
    """Merge building/car channels into a binary obstacle map [1,H,W]."""
    maps = [select_channel(x_i, ch).float() for ch in obstacle_channels]
    obstacle = torch.stack(maps, dim=0).amax(dim=0)
    return (obstacle > threshold).float()


def tx_center(tx_map):
    """Return integer Tx center (y, x) from tx_map=[1,H,W]."""
    _, h, w = tx_map.shape
    flat = tx_map.flatten().argmax()
    y = int(flat // w)
    x = int(flat % w)
    return y, x


def minmax(z, eps=1e-8):
    zmin = z.amin()
    zmax = z.amax()
    return (z - zmin) / (zmax - zmin + eps)


def make_radial_gain(h, w, tx_y, tx_x, device, dtype=torch.float32):
    """Log-distance free-space-like gain prior in [0,1]."""
    yy = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, w)
    dist = torch.sqrt((yy - float(tx_y)) ** 2 + (xx - float(tx_x)) ** 2)
    max_dist = math.sqrt((h - 1) ** 2 + (w - 1) ** 2)

    gain = 1.0 - torch.log1p(dist) / math.log1p(max_dist)
    return gain.clamp(0.0, 1.0).unsqueeze(0)


def make_corner_affinity(
    obstacle_mask,
    corner_sigma,
    max_corners,
    response_threshold,
    nms_radius,
    harris_k,
):
    """Create a dense [1,H,W] Harris-like obstacle-corner affinity map.

    The affinity is a geometry proxy used by corner_diffraction. It does not
    calculate exact physical diffraction coefficients.
    """
    _, h, w = obstacle_mask.shape
    device = obstacle_mask.device
    dtype = torch.float32

    z = obstacle_mask.float().unsqueeze(0)  # [1,1,H,W]

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0

    gx = F.conv2d(z, sobel_x, padding=1)
    gy = F.conv2d(z, sobel_y, padding=1)

    sxx = F.avg_pool2d(gx * gx, kernel_size=3, stride=1, padding=1)
    syy = F.avg_pool2d(gy * gy, kernel_size=3, stride=1, padding=1)
    sxy = F.avg_pool2d(gx * gy, kernel_size=3, stride=1, padding=1)

    response = sxx * syy - float(harris_k) * (sxx + syy) ** 2
    response = torch.relu(response[0, 0])
    max_response = response.max()

    if float(max_response) <= 1e-12:
        return torch.zeros((1, h, w), device=device, dtype=dtype)

    radius = max(0, int(nms_radius))
    if radius > 0:
        pooled = F.max_pool2d(
            response.view(1, 1, h, w),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[0, 0]
        is_peak = (response >= pooled) & (
            response >= float(response_threshold) * max_response
        )
    else:
        is_peak = response >= float(response_threshold) * max_response

    ys, xs = torch.where(is_peak)
    if ys.numel() == 0:
        return torch.zeros((1, h, w), device=device, dtype=dtype)

    scores = response[ys, xs]
    k = min(int(max_corners), int(scores.numel()))
    if k <= 0:
        return torch.zeros((1, h, w), device=device, dtype=dtype)

    scores, order = torch.topk(scores, k=k, largest=True, sorted=False)
    ys = ys[order].float()
    xs = xs[order].float()
    weights = scores / scores.max().clamp_min(1e-12)

    yy = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
    sigma = max(float(corner_sigma), 1e-6)

    # At 256x256 and max_corners=128, this temporary tensor is ~32 MB float32.
    dist2 = (yy - ys.view(-1, 1, 1)) ** 2 + (xx - xs.view(-1, 1, 1)) ** 2
    affinity = (
        torch.exp(-0.5 * dist2 / (sigma ** 2))
        * weights.view(-1, 1, 1)
    ).amax(dim=0)

    return affinity.clamp(0.0, 1.0).unsqueeze(0)


@torch.no_grad()
def compute_input_driven_targets(
    x_i,
    tx_channel=-1,
    obstacle_channels=(0, 1),
    building_threshold=0.5,
    obstacle_alphas=(0.03, 0.05),
    corner_sigma=3.0,
    corner_posthit_decay=0.03,
    corner_max_corners=128,
    corner_response_threshold=0.05,
    corner_nms_radius=2,
    corner_harris_k=0.04,
):
    """Compute all requested geometry-derived targets online for one sample."""
    x_i = x_i.float()
    obstacle_mask = build_obstacle_map(
        x_i,
        obstacle_channels=obstacle_channels,
        threshold=building_threshold,
    )
    tx_map = select_channel(x_i, tx_channel).float()

    _, h, w = obstacle_mask.shape
    device = x_i.device
    tx_y, tx_x = tx_center(tx_map)

    radial_gain = make_radial_gain(h, w, tx_y, tx_x, device)
    corner_affinity = make_corner_affinity(
        obstacle_mask=obstacle_mask,
        corner_sigma=corner_sigma,
        max_corners=corner_max_corners,
        response_threshold=corner_response_threshold,
        nms_radius=corner_nms_radius,
        harris_k=corner_harris_k,
    )

    obstacle_sum_raw = torch.zeros((1, h, w), device=device, dtype=torch.float32)
    saturating_maps = {
        alpha: torch.zeros((1, h, w), device=device, dtype=torch.float32)
        for alpha in obstacle_alphas
    }
    corner_diffraction = torch.zeros((1, h, w), device=device, dtype=torch.float32)

    for y1 in range(h):
        dy = y1 - tx_y
        for x1 in range(w):
            dx = x1 - tx_x
            n = max(abs(dx), abs(dy), 1) + 1

            rr = torch.linspace(tx_y, y1, n, device=device).round().long().clamp(0, h - 1)
            cc = torch.linspace(tx_x, x1, n, device=device).round().long().clamp(0, w - 1)

            ray_values = obstacle_mask[0, rr, cc]
            hit_length = ray_values.sum()
            obstacle_sum_raw[0, y1, x1] = hit_length

            for alpha, map_tensor in saturating_maps.items():
                map_tensor[0, y1, x1] = 1.0 - torch.exp(-float(alpha) * hit_length)

            # First-blocker corner diffraction potential:
            # first blocker must be corner-like; potential decays with distance
            # from that blocker to the receiver pixel.
            hit_indices = torch.nonzero(ray_values > 0.5, as_tuple=False).flatten()
            if hit_indices.numel() > 0:
                first = int(hit_indices[0].item())
                hit_y = int(rr[first].item())
                hit_x = int(cc[first].item())

                post_hit_distance = math.sqrt(
                    (y1 - hit_y) ** 2 + (x1 - hit_x) ** 2
                )
                corner_diffraction[0, y1, x1] = (
                    corner_affinity[0, hit_y, hit_x]
                    * math.exp(-float(corner_posthit_decay) * post_hit_distance)
                )

    targets = {
        "obstacle_sum": minmax(obstacle_sum_raw),
        "radial_gain": radial_gain,
        "corner_diffraction": corner_diffraction.clamp(0.0, 1.0),
    }

    # The visualizer uses fixed names for alpha=.03 and alpha=.05.
    for alpha, target_map in saturating_maps.items():
        if abs(alpha - 0.03) < 1e-8:
            targets["obstacle_saturating_a003"] = target_map.clamp(0.0, 1.0)
        elif abs(alpha - 0.05) < 1e-8:
            targets["obstacle_saturating_a005"] = target_map.clamp(0.0, 1.0)

    return targets


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

    for idx, (title, image_tensor, cmap) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        image = ax.imshow(to_numpy_img(minmax_for_display(image_tensor)), cmap=cmap)
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

    requested_label_targets = [
        name for name in target_names if name in ONLINE_LABEL_TARGETS
    ]
    requested_input_targets = [
        name for name in target_names if name in ONLINE_INPUT_TARGETS
    ]

    obstacle_channels = parse_int_list(args.obstacle_channels)
    obstacle_alphas = parse_float_list(args.obstacle_alphas)

    missing_alpha_targets = []
    if "obstacle_saturating_a003" in requested_input_targets and 0.03 not in obstacle_alphas:
        missing_alpha_targets.append("obstacle_saturating_a003 requires --obstacle-alphas to include 0.03")
    if "obstacle_saturating_a005" in requested_input_targets and 0.05 not in obstacle_alphas:
        missing_alpha_targets.append("obstacle_saturating_a005 requires --obstacle-alphas to include 0.05")
    if missing_alpha_targets:
        raise ValueError("; ".join(missing_alpha_targets))

    label_builder = None
    if requested_label_targets:
        label_builder = PhysicsTargetBuilder(
            target_names=requested_label_targets,
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
        label_builder.eval()

    loader = get_split_loader(cfg, args.split)
    os.makedirs(args.save_dir, exist_ok=True)

    print("[INFO] All targets are computed online; no .pt files are loaded.")
    print(f"[INFO] Requested targets: {target_names}")
    print(f"[INFO] Obstacle alphas: {obstacle_alphas}")
    print(f"[INFO] Split: {args.split} | samples: {args.num_samples}")

    saved = 0
    for batch in loader:
        x, y, names = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            all_targets = {}

            if label_builder is not None:
                all_targets.update(label_builder(x, y))

            if requested_input_targets:
                per_sample_targets = []
                for sample_index in range(x.size(0)):
                    per_sample_targets.append(
                        compute_input_driven_targets(
                            x_i=x[sample_index],
                            tx_channel=args.tx_channel,
                            obstacle_channels=obstacle_channels,
                            building_threshold=args.building_threshold,
                            obstacle_alphas=obstacle_alphas,
                            corner_sigma=args.corner_sigma,
                            corner_posthit_decay=args.corner_posthit_decay,
                            corner_max_corners=args.corner_max_corners,
                            corner_response_threshold=args.corner_response_threshold,
                            corner_nms_radius=args.corner_nms_radius,
                            corner_harris_k=args.corner_harris_k,
                        )
                    )

                for target_name in requested_input_targets:
                    if any(target_name not in sample_targets for sample_targets in per_sample_targets):
                        raise KeyError(
                            f"Online geometry target '{target_name}' was not created. "
                            "Check --obstacle-alphas and --physics-targets."
                        )
                    all_targets[target_name] = torch.stack(
                        [sample_targets[target_name] for sample_targets in per_sample_targets],
                        dim=0,
                    ).to(device=device)

        # Preserve requested ordering.
        all_targets = {name: all_targets[name] for name in target_names}

        for i in range(x.size(0)):
            if saved >= args.num_samples:
                break

            sample_name = names[i] if names is not None else None
            stem = f"sample_{saved:03d}"
            if sample_name is not None:
                stem += f"_{safe_stem(sample_name)}"

            x_i = x[i].detach().cpu()
            y_i = y[i].detach().cpu()
            targets_i = {
                key: value[i].detach().cpu()
                for key, value in all_targets.items()
            }

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
