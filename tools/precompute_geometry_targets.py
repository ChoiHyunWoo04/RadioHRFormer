import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

from datasets.rms_dataset import build_dataloaders
from utils import set_seed, prepare_device


# Saved target keys by default:
#   obstacle_sum
#   obstacle_saturating_a003
#   obstacle_saturating_a005
#   radial_gain
#   corner_diffraction
#
# Notes:
# - "obstacle_saturating" is represented by one key per alpha value, because
#   alpha changes the map itself. It is not min-max normalized after applying
#   1 - exp(-alpha * obstruction_length), so the alpha effect is preserved.
# - corner_diffraction is a geometry-driven proxy, not exact UTD/knife-edge loss.


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Offline precompute input-driven propagation proxy maps for RadioMapSeer: "
            "obstacle_sum, obstacle_saturating variants, radial_gain, and "
            "corner_diffraction."
        )
    )

    # Paths
    p.add_argument("--config-path", type=str, default="./configs/hrformer_radiomapseer.json")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--save-root", type=str, default="./data/precomputed_input_driven")

    # RadioMapSeer mode
    p.add_argument("--input-mode", choices=["building", "cars"], default="cars")
    p.add_argument("--target-type", choices=["DPM", "carsDPM"], default=None)
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--num-tx", type=int, default=None)
    p.add_argument("--thresh", type=float, default=None)

    # Runtime
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cuda", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float16")

    # Geometry input interpretation
    p.add_argument("--tx-channel", type=int, default=-1)
    p.add_argument(
        "--obstacle-channels",
        type=str,
        default="0,1",
        help="Comma-separated obstacle channels. For [building, cars, Tx], use 0,1.",
    )
    p.add_argument("--building-threshold", type=float, default=0.5)

    # Obstruction maps
    p.add_argument(
        "--obstacle-alphas",
        type=str,
        default="0.03,0.05",
        help=(
            "Comma-separated alpha values for saturation maps. "
            "For example: 0.03,0.05. Each alpha is saved under a distinct key."
        ),
    )

    # Corner-diffraction proxy
    p.add_argument(
        "--corner-sigma",
        type=float,
        default=3.0,
        help="Spatial Gaussian spread in pixels for selected corner candidates.",
    )
    p.add_argument(
        "--corner-posthit-decay",
        type=float,
        default=0.03,
        help=(
            "Decay in exp(-decay * post_hit_distance) for the "
            "corner_diffraction proxy."
        ),
    )
    p.add_argument(
        "--corner-max-corners",
        type=int,
        default=128,
        help="Maximum number of non-maximum-suppressed corner candidates.",
    )
    p.add_argument(
        "--corner-response-threshold",
        type=float,
        default=0.05,
        help="Keep Harris-like corner candidates above this fraction of max response.",
    )
    p.add_argument(
        "--corner-nms-radius",
        type=int,
        default=2,
        help="Non-maximum suppression radius for corner candidate selection.",
    )
    p.add_argument(
        "--corner-harris-k",
        type=float,
        default=0.04,
        help="Harris response coefficient.",
    )

    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


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
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def parse_float_list(text):
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("--obstacle-alphas must contain at least one positive alpha.")
    if any(alpha <= 0.0 for alpha in values):
        raise ValueError(f"All obstacle alphas must be positive. Got: {values}")
    return tuple(dict.fromkeys(values))


def alpha_key(alpha):
    """0.03 -> obstacle_saturating_a003, 0.05 -> obstacle_saturating_a005."""
    tag = f"{float(alpha):.6f}".rstrip("0").rstrip(".")
    tag = tag.replace(".", "")
    return f"obstacle_saturating_a{tag.zfill(3)}"


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

    # Map-diagonal normalization keeps the target comparable across Tx locations.
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
    """Build dense [1,H,W] affinity map for obstacle corners.

    This identifies Harris-like corner candidates on the binary obstacle map and
    spreads their influence spatially with Gaussian kernels. It is a geometric
    approximation used by corner_diffraction, not a physical diffraction solver.
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

    # [K,H,W], K <= max_corners. With K=128 and 256x256, this is ~32 MB float32.
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
    """Compute input-driven propagation proxy maps for one sample.

    Returns:
        obstacle_sum:
            Sample-wise min-max normalized ray obstruction integral.

        obstacle_saturating_aXXX:
            One map per alpha: 1 - exp(-alpha * obstruction_length).
            These remain in their natural [0,1] range without min-max.

        radial_gain:
            Tx-distance prior with log-distance decay.

        corner_diffraction:
            First-blocker corner-diffraction potential proxy.
            It is not exact UTD or knife-edge diffraction loss.
    """
    x_i = x_i.float()
    obstacle_mask = build_obstacle_map(
        x_i,
        obstacle_channels=obstacle_channels,
        threshold=building_threshold,
    )
    tx = select_channel(x_i, tx_channel).float()

    _, h, w = obstacle_mask.shape
    device = x_i.device
    tx_y, tx_x = tx_center(tx)

    radial_gain = make_radial_gain(h, w, tx_y, tx_x, device=device)
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

            ray_vals = obstacle_mask[0, rr, cc]
            hit_len = ray_vals.sum()
            obstacle_sum_raw[0, y1, x1] = hit_len

            for alpha, target_map in saturating_maps.items():
                target_map[0, y1, x1] = 1.0 - torch.exp(-float(alpha) * hit_len)

            # First-obstruction corner-diffraction potential:
            # a corner-like first blocker can mediate NLoS propagation; the
            # potential decays as the receiver moves farther beyond that blocker.
            hit_idx = torch.nonzero(ray_vals > 0.5, as_tuple=False).flatten()
            if hit_idx.numel() > 0:
                first = int(hit_idx[0].item())
                hit_y = int(rr[first].item())
                hit_x = int(cc[first].item())

                post_hit_distance = math.sqrt(
                    (y1 - hit_y) ** 2 + (x1 - hit_x) ** 2
                )
                corner_score = corner_affinity[0, hit_y, hit_x]
                corner_diffraction[0, y1, x1] = (
                    corner_score
                    * math.exp(-float(corner_posthit_decay) * post_hit_distance)
                )

    targets = {
        "obstacle_sum": minmax(obstacle_sum_raw).cpu(),
        "radial_gain": radial_gain.cpu(),
        "corner_diffraction": corner_diffraction.clamp(0.0, 1.0).cpu(),
    }

    for alpha, target_map in saturating_maps.items():
        targets[alpha_key(alpha)] = target_map.clamp(0.0, 1.0).cpu()

    return targets


def get_split_loader(split_dict, split):
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
        raise ValueError(f"Unexpected batch format: len={len(batch)}")

    if names is None:
        raise ValueError(
            "Dataset must return sample names. Set cfg['data']['return_name']=True."
        )

    return x, y, names


def save_manifest(save_base, args, cfg, obstacle_alphas):
    saturation_keys = {alpha_key(alpha): alpha for alpha in obstacle_alphas}
    manifest = {
        "target_keys": [
            "obstacle_sum",
            *saturation_keys.keys(),
            "radial_gain",
            "corner_diffraction",
        ],
        "saturation_alpha_by_key": saturation_keys,
        "dtype": args.dtype,
        "input_mode": args.input_mode,
        "target_type": cfg["data"]["target_type"],
        "tx_channel": args.tx_channel,
        "obstacle_channels": list(parse_int_list(args.obstacle_channels)),
        "building_threshold": args.building_threshold,
        "corner_sigma": args.corner_sigma,
        "corner_posthit_decay": args.corner_posthit_decay,
        "corner_max_corners": args.corner_max_corners,
        "corner_response_threshold": args.corner_response_threshold,
        "corner_nms_radius": args.corner_nms_radius,
        "corner_harris_k": args.corner_harris_k,
        "corner_diffraction_note": (
            "First-blocker corner-diffraction potential proxy; "
            "not exact UTD or knife-edge diffraction loss."
        ),
    }

    with open(os.path.join(save_base, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = apply_overrides(load_config(args.config_path), args)
    device = prepare_device(args.cuda)

    obstacle_channels = parse_int_list(args.obstacle_channels)
    obstacle_alphas = parse_float_list(args.obstacle_alphas)

    mode_name = f"{args.input_mode}_{cfg['data']['target_type']}"
    save_base = os.path.join(args.save_root, mode_name)
    os.makedirs(save_base, exist_ok=True)
    save_manifest(save_base, args, cfg, obstacle_alphas)

    split_dict = build_dataloaders(cfg, return_datasets=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    target_keys = [
        "obstacle_sum",
        *[alpha_key(alpha) for alpha in obstacle_alphas],
        "radial_gain",
        "corner_diffraction",
    ]

    print(f"[INFO] save_base                 : {save_base}")
    print(f"[INFO] splits                    : {splits}")
    print(f"[INFO] saved target keys         : {target_keys}")
    print(f"[INFO] obstacle_channels         : {obstacle_channels}")
    print(f"[INFO] obstacle_alphas           : {obstacle_alphas}")
    print(f"[INFO] tx_channel                : {args.tx_channel}")
    print(f"[INFO] dtype                     : {args.dtype}")

    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    for split in splits:
        loader = get_split_loader(split_dict, split)
        split_key = "val" if split == "valid" else split
        save_dir = os.path.join(save_base, split_key)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[INFO] Precomputing split={split_key}, save_dir={save_dir}")

        saved = 0
        skipped = 0

        pbar = tqdm(loader, desc=f"precompute {split_key}")
        for batch in pbar:
            x, _, names = unpack_batch(batch)

            for i in range(x.size(0)):
                stem = safe_stem(names[i])
                save_path = os.path.join(save_dir, f"{stem}.pt")

                if os.path.exists(save_path) and not args.overwrite:
                    skipped += 1
                    continue

                x_i = x[i].to(device, non_blocking=True)

                targets = compute_input_driven_targets(
                    x_i=x_i,
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

                targets = {key: value.to(dtype=save_dtype) for key, value in targets.items()}
                torch.save(targets, save_path)
                saved += 1

            pbar.set_postfix({"saved": saved, "skipped": skipped})

        print(f"[DONE] split={split_key} saved={saved}, skipped={skipped}")

    print("\nFinished precomputing input-driven propagation proxy maps.")


if __name__ == "__main__":
    main()
