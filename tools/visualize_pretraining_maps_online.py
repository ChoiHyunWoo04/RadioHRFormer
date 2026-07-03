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
# Online visualizer for RadioMapSeer physics-pretraining targets.
#
# All experiment-specific options are read from hrt.json. The command line only
# selects the config file and optionally redirects the output directory.
#
# The visualizer keeps the same online target definitions as the previous script:
#   label-driven: grad, lap, singularity
#   geometry-driven: los, obstacle, obstacle_sum, obstacle_saturating_a003,
#                    obstacle_saturating_a005, radial_gain, corner_diffraction
#
# `los` and `obstacle` are included so that the legacy cfg target list
# [grad, lap, los, obstacle] can also be visualized without changing cfg.
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

TARGET_ALIASES = {
    "radial-gain": "radial_gain",
    "corner-diffraction": "corner_diffraction",
    "obstacle-saturating": "obstacle_saturating_a003",
    "obstacle_saturating": "obstacle_saturating_a003",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize randomly sampled RadioMapSeer inputs, labels, and "
            "physics-pretraining targets using values from the JSON config."
        )
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/hrt.json",
        help="Path to the shared HRFormer JSON config.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Optional output-directory override. If omitted, uses "
            "cfg['visualize']['save_dir']."
        ),
    )
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def cfg_get(cfg, keys, default=None):
    current = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_list(value, default):
    if value is None:
        value = default
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def apply_visualizer_config(cfg, args):
    """Prepare a loader-only cfg while preserving shared cfg settings.

    The visualizer needs sample names and a small loader batch. All dataset,
    physics, seed, and runtime values are otherwise read directly from cfg.
    """
    cfg.setdefault("data", {})
    cfg.setdefault("physics", {})
    visualize_cfg = cfg.setdefault("visualize", {})

    if cfg["data"].get("root_dir") is None:
        raise ValueError("cfg['data']['root_dir'] must be set.")

    # A ray-based target is computed per sample. Keep the visualization loader
    # separate from the training batch size to avoid computing unused samples.
    cfg["data"]["batch_size"] = int(visualize_cfg.get("batch_size", 1))
    cfg["data"]["num_workers"] = int(
        visualize_cfg.get("num_workers", cfg["data"].get("num_workers", 0))
    )
    cfg["data"]["pin_memory"] = bool(
        visualize_cfg.get("pin_memory", cfg["data"].get("pin_memory", True))
    )
    cfg["data"]["persistent_workers"] = bool(
        visualize_cfg.get("persistent_workers", cfg["data"].get("persistent_workers", False))
    ) and cfg["data"]["num_workers"] > 0
    cfg["data"]["return_name"] = True

    if args.save_dir is not None:
        visualize_cfg["save_dir"] = args.save_dir

    visualize_cfg.setdefault("save_dir", "./save/visual_online")
    visualize_cfg.setdefault("split", "train")
    visualize_cfg.setdefault("num_samples", 1)
    visualize_cfg.setdefault("dpi", 180)
    visualize_cfg.setdefault("save_grid", True)
    visualize_cfg.setdefault("save_individual", True)

    return cfg


def resolve_visualizer_device(cfg):
    runtime_cfg = cfg_get(cfg, ["runtime"], {}) or {}
    gpu_ids = _as_list(runtime_cfg.get("gpus", [0]), [0])

    if torch.cuda.is_available() and gpu_ids:
        # A visualization task uses the first configured GPU only. DataParallel
        # would add overhead and provides no benefit for a few sampled maps.
        return prepare_device(str(int(gpu_ids[0])))
    return torch.device("cpu")


def resolve_visualizer_options(cfg):
    data_cfg = cfg["data"]
    physics_cfg = cfg.get("physics", {})
    visualize_cfg = cfg.get("visualize", {})

    target_values = visualize_cfg.get("targets", physics_cfg.get("targets"))
    if target_values is None:
        raise KeyError(
            "Set cfg['physics']['targets'] or cfg['visualize']['targets']."
        )

    return {
        "seed": int(cfg.get("seed", 42)),
        "split": str(visualize_cfg["split"]),
        "num_samples": int(visualize_cfg["num_samples"]),
        "save_dir": str(visualize_cfg["save_dir"]),
        "dpi": int(visualize_cfg["dpi"]),
        "save_grid": bool(visualize_cfg["save_grid"]),
        "save_individual": bool(visualize_cfg["save_individual"]),
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "cars_input": bool(data_cfg.get("cars_input", False)),
        "target_type": str(data_cfg.get("target_type", "DPM")),
        "target_names": parse_target_names(target_values),
        "field_mode": str(physics_cfg.get("field_mode", "normalized_power")),
        "gaussian_sigma": float(physics_cfg.get("gaussian_sigma", 1.0)),
        "eps": float(physics_cfg.get("eps", 1e-4)),
        "tx_channel": int(physics_cfg.get("tx_channel", -1)),
        "normalize_each_sample": bool(physics_cfg.get("normalize_each_sample", True)),
        "building_threshold": float(physics_cfg.get("building_threshold", 0.5)),
        "obstacle_channels": parse_int_list(physics_cfg.get("obstacle_channels", [0, 1])),
        "obstacle_alphas": parse_float_list(physics_cfg.get("obstacle_alphas", [0.03, 0.05])),
        "corner_sigma": float(physics_cfg.get("corner_sigma", 3.0)),
        "corner_posthit_decay": float(physics_cfg.get("corner_posthit_decay", 0.03)),
        "corner_max_corners": int(physics_cfg.get("corner_max_corners", 128)),
        "corner_response_threshold": float(physics_cfg.get("corner_response_threshold", 0.05)),
        "corner_nms_radius": int(physics_cfg.get("corner_nms_radius", 2)),
        "corner_harris_k": float(physics_cfg.get("corner_harris_k", 0.04)),
        "radiodiff_pathloss_trunc": float(physics_cfg.get("radiodiff_pathloss_trunc", -147.0)),
        "radiodiff_pathloss_max": float(physics_cfg.get("radiodiff_pathloss_max", -47.0)),
        "radiodiff_source_power_dbm": float(physics_cfg.get("radiodiff_source_power_dbm", 23.0)),
        "radiodiff_h": float(physics_cfg.get("radiodiff_h", 1.0)),
        "radiodiff_border_value": float(physics_cfg.get("radiodiff_border_value", 1.0)),
        "radiodiff_eps": float(physics_cfg.get("radiodiff_eps", 1e-30)),
        "radiodiff_smooth_sigma": float(physics_cfg.get("radiodiff_smooth_sigma", 0.9)),
    }


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


def parse_target_names(values):
    names = _as_list(values, [])
    if not names:
        raise ValueError("At least one visualization target must be provided.")

    normalized_names = []
    for raw_name in names:
        name = TARGET_ALIASES.get(str(raw_name).strip(), str(raw_name).strip())
        if name and name not in normalized_names:
            normalized_names.append(name)

    invalid = [name for name in normalized_names if name not in SUPPORTED_TARGETS]
    if invalid:
        raise ValueError(
            f"Unsupported visualization target(s): {invalid}. "
            f"Supported: {sorted(SUPPORTED_TARGETS)}"
        )
    return normalized_names


def parse_int_list(values):
    values = _as_list(values, [])
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise ValueError("obstacle_channels must contain at least one channel index.")
    return parsed


def parse_float_list(values):
    values = _as_list(values, [])
    parsed = tuple(dict.fromkeys(float(v) for v in values))
    if not parsed:
        raise ValueError("obstacle_alphas must contain at least one alpha.")
    if any(value <= 0 for value in parsed):
        raise ValueError(f"All obstacle_alphas must be positive. Got: {parsed}")
    return parsed


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

    normalized_obstacle = minmax(obstacle_sum_raw)
    targets = {
        # `obstacle` is kept as a legacy alias for the normalized cumulative
        # obstacle map used in the earlier pretraining configuration.
        "los": (obstacle_sum_raw <= 0).float(),
        "obstacle": normalized_obstacle,
        "obstacle_sum": normalized_obstacle,
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


def infer_channel_titles(x, cars_input):
    c = x.shape[0]
    if cars_input:
        base = ["input ch0: building", "input ch1: cars", "input ch2: Tx"]
    else:
        base = ["input ch0: building", "input ch1: building", "input ch2: Tx"]
    return base[:c] if c <= len(base) else base + [f"input ch{i}" for i in range(len(base), c)]


def pretty_target_name(name):
    aliases = {
        "los": "line of sight",
        "obstacle": "obstacle: cumulative",
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
    plt.imshow(to_numpy_img(minmax_for_display(img)), cmap=cmap)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def save_sample_grid(
    save_path,
    x_i,
    y_i,
    targets_i: Dict[str, torch.Tensor],
    cars_input,
    name: Optional[str] = None,
    dpi=180,
):
    panels = []
    titles = infer_channel_titles(x_i, cars_input)

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


def save_individual_maps(sample_dir, x_i, y_i, targets_i, cars_input, dpi):
    """Save every panel as a standalone PNG in the sample directory."""
    os.makedirs(sample_dir, exist_ok=True)
    titles = infer_channel_titles(x_i, cars_input)

    for channel, title in enumerate(titles):
        save_single_map(
            os.path.join(sample_dir, f"input_ch{channel}.png"),
            x_i[channel],
            title,
            cmap="gray",
            dpi=dpi,
        )

    save_single_map(
        os.path.join(sample_dir, "label_y.png"),
        y_i[0],
        "label y",
        cmap="viridis",
        dpi=dpi,
    )

    for target_name, target_map in targets_i.items():
        save_single_map(
            os.path.join(sample_dir, f"physics_{target_name}.png"),
            target_map[0],
            f"physics: {pretty_target_name(target_name)}",
            cmap="viridis",
            dpi=dpi,
        )


def main():
    args = parse_args()
    cfg = apply_visualizer_config(load_config(args.config_path), args)
    opts = resolve_visualizer_options(cfg)

    if opts["num_samples"] <= 0:
        raise ValueError("cfg['visualize']['num_samples'] must be positive.")

    set_seed(opts["seed"])
    device = resolve_visualizer_device(cfg)

    target_names = opts["target_names"]
    requested_label_targets = [
        name for name in target_names if name in ONLINE_LABEL_TARGETS
    ]
    requested_input_targets = [
        name for name in target_names if name in ONLINE_INPUT_TARGETS
    ]

    missing_alpha_targets = []
    if (
        "obstacle_saturating_a003" in requested_input_targets
        and 0.03 not in opts["obstacle_alphas"]
    ):
        missing_alpha_targets.append(
            "obstacle_saturating_a003 requires physics.obstacle_alphas to include 0.03"
        )
    if (
        "obstacle_saturating_a005" in requested_input_targets
        and 0.05 not in opts["obstacle_alphas"]
    ):
        missing_alpha_targets.append(
            "obstacle_saturating_a005 requires physics.obstacle_alphas to include 0.05"
        )
    if missing_alpha_targets:
        raise ValueError("; ".join(missing_alpha_targets))

    label_builder = None
    if requested_label_targets:
        label_builder = PhysicsTargetBuilder(
            target_names=requested_label_targets,
            field_mode=opts["field_mode"],
            gaussian_sigma=opts["gaussian_sigma"],
            eps=opts["eps"],
            tx_channel=opts["tx_channel"],
            normalize_each_sample=opts["normalize_each_sample"],
            radiodiff_pathloss_trunc=opts["radiodiff_pathloss_trunc"],
            radiodiff_pathloss_max=opts["radiodiff_pathloss_max"],
            radiodiff_source_power_dbm=opts["radiodiff_source_power_dbm"],
            radiodiff_h=opts["radiodiff_h"],
            radiodiff_border_value=opts["radiodiff_border_value"],
            radiodiff_eps=opts["radiodiff_eps"],
            radiodiff_smooth_sigma=opts["radiodiff_smooth_sigma"],
        ).to(device)
        label_builder.eval()

    loader = get_split_loader(cfg, opts["split"])
    os.makedirs(opts["save_dir"], exist_ok=True)

    print("[INFO] All targets are computed online; no .pt files are loaded.")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Target type: {opts['target_type']}")
    print(f"[INFO] Requested targets: {target_names}")
    print(
        f"[INFO] Split: {opts['split']} | samples: {opts['num_samples']} "
        f"| loader batch size: {opts['batch_size']}"
    )

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
                            tx_channel=opts["tx_channel"],
                            obstacle_channels=opts["obstacle_channels"],
                            building_threshold=opts["building_threshold"],
                            obstacle_alphas=opts["obstacle_alphas"],
                            corner_sigma=opts["corner_sigma"],
                            corner_posthit_decay=opts["corner_posthit_decay"],
                            corner_max_corners=opts["corner_max_corners"],
                            corner_response_threshold=opts["corner_response_threshold"],
                            corner_nms_radius=opts["corner_nms_radius"],
                            corner_harris_k=opts["corner_harris_k"],
                        )
                    )

                for target_name in requested_input_targets:
                    if any(target_name not in sample_targets for sample_targets in per_sample_targets):
                        raise KeyError(
                            f"Online geometry target '{target_name}' was not created. "
                            "Check physics.obstacle_alphas and physics.targets."
                        )
                    all_targets[target_name] = torch.stack(
                        [sample_targets[target_name] for sample_targets in per_sample_targets],
                        dim=0,
                    ).to(device=device)

        # Preserve the cfg target ordering in both grid and standalone filenames.
        all_targets = {name: all_targets[name] for name in target_names}

        for i in range(x.size(0)):
            if saved >= opts["num_samples"]:
                break

            sample_name = names[i] if names is not None else None
            stem = f"sample_{saved:03d}"
            if sample_name is not None:
                stem += f"_{safe_stem(sample_name)}"

            sample_dir = os.path.join(opts["save_dir"], stem)
            os.makedirs(sample_dir, exist_ok=True)

            x_i = x[i].detach().cpu()
            y_i = y[i].detach().cpu()
            targets_i = {
                key: value[i].detach().cpu()
                for key, value in all_targets.items()
            }

            if opts["save_grid"]:
                grid_path = os.path.join(sample_dir, "physics_grid.png")
                save_sample_grid(
                    save_path=grid_path,
                    x_i=x_i,
                    y_i=y_i,
                    targets_i=targets_i,
                    cars_input=opts["cars_input"],
                    name=sample_name,
                    dpi=opts["dpi"],
                )

            if opts["save_individual"]:
                save_individual_maps(
                    sample_dir=sample_dir,
                    x_i=x_i,
                    y_i=y_i,
                    targets_i=targets_i,
                    cars_input=opts["cars_input"],
                    dpi=opts["dpi"],
                )

            print(f"Saved sample directory: {sample_dir}")
            saved += 1

        if saved >= opts["num_samples"]:
            break

    print(f"Done. Saved {saved} sampled visualization(s) to: {opts['save_dir']}")


if __name__ == "__main__":
    main()
