import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import numpy as np
from skimage.metrics import structural_similarity as ssim_metric


################################## NMSE, RMSE, PSNR, SSIM ##################################

def compute_nmse(pred, target):
    numerator = torch.sum((pred - target) ** 2)
    denominator = torch.sum(target ** 2) + 1e-8
    nmse = numerator / denominator
    return nmse

def compute_rmse(pred, target):
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-8))
    return psnr

def compute_ssim(pred, target, data_range=1.0):
    """Compute mean single-scale SSIM using skimage.metrics.structural_similarity.

    This matches the RadioUNet / RadioDiff project evaluators.
    Expected input: [N, 1, H, W] or [N, H, W].
    """
    if torch.is_tensor(pred):
        pred_np = pred.detach().cpu().float().numpy()
    else:
        pred_np = np.asarray(pred, dtype=np.float32)

    if torch.is_tensor(target):
        target_np = target.detach().cpu().float().numpy()
    else:
        target_np = np.asarray(target, dtype=np.float32)

    if pred_np.ndim == 4 and pred_np.shape[1] == 1:
        pred_np = pred_np[:, 0]
    if target_np.ndim == 4 and target_np.shape[1] == 1:
        target_np = target_np[:, 0]

    return float(
        np.mean([
            ssim_metric(target_np[i], pred_np[i], data_range=data_range)
            for i in range(pred_np.shape[0])
        ])
    )


################################## Boundary, LoS, NLoS, Long-range NLoS ##################################

def get_obstacle_mask(x, target_type):
    """
    x: [B, 3, H, W]
       DPM     -> [building, building, tx]
       carsDPM -> [building, cars, tx]
    """
    building = x[:, 0:1] > 0.5

    if target_type.lower() == "carsdpm":
        cars = x[:, 1:2] > 0.5
        obstacle = building | cars
    else:
        obstacle = building

    return obstacle


def get_tx_distance_map(tx_map):
    """
    tx_map: [B, 1, H, W]
    return: [B, 1, H, W]
    """
    B, _, H, W = tx_map.shape
    device = tx_map.device

    idx = tx_map.reshape(B, -1).argmax(dim=1)

    ty = (idx // W).float().view(B, 1, 1, 1)
    tx = (idx % W).float().view(B, 1, 1, 1)

    yy = torch.arange(H, device=device).float().view(1, 1, H, 1)
    xx = torch.arange(W, device=device).float().view(1, 1, 1, W)

    dist = torch.sqrt((yy - ty) ** 2 + (xx - tx) ** 2)

    return dist


def get_boundary_band(obstacle, radius=3):
    """
    obstacle: [B, 1, H, W], bool

    Outdoor pixels lying within `radius` pixels of an obstacle.
    """
    obs = obstacle.float()

    k = 2 * radius + 1

    dilated = F.max_pool2d(
        obs,
        kernel_size=k,
        stride=1,
        padding=radius,
    ) > 0.5

    outdoor = ~obstacle

    boundary_band = dilated & outdoor

    return boundary_band


def build_region_masks(
    x,
    target_type,
    d0,
    obstacle_accum=None,
    z_tr=None,
    boundary_radius=3,
):
    """
    obstacle_accum:
        A(p), shape [B,1,H,W]

    z_tr:
        exp(-alpha * A(p)), shape [B,1,H,W]
        obstacle_accum이 없다면 이것을 사용.

    d0:
        validation set에서 미리 결정한 long-range threshold
    """

    obstacle = get_obstacle_mask(x, target_type)
    outdoor = ~obstacle

    # 1. Boundary
    boundary = get_boundary_band(
        obstacle,
        radius=boundary_radius,
    )

    # 2. LoS / NLoS
    if obstacle_accum is not None:
        los = obstacle_accum <= 0
    elif z_tr is not None:
        # A=0 -> z_tr=1
        # A>=1 -> z_tr<=exp(-0.07) ~= 0.932
        los = z_tr > 0.99
    else:
        raise ValueError(
            "obstacle_accum or z_tr must be provided."
        )

    los = los & outdoor
    nlos = (~los) & outdoor

    # 3. Tx-to-pixel distance
    tx_map = x[:, -1:, :, :]
    distance = get_tx_distance_map(tx_map)

    # 4. Long-range NLoS
    long_nlos = nlos & (distance >= d0)

    return {
        "boundary": boundary,
        "los": los,
        "nlos": nlos,
        "long_nlos": long_nlos,
    }


def masked_rmse_per_sample(pred, target, mask):
    """
    pred, target: [B,1,H,W]
    mask:         [B,1,H,W]
    """
    sq_error = (pred - target) ** 2

    values = []

    for b in range(pred.shape[0]):
        m = mask[b, 0]

        if m.sum() == 0:
            continue

        mse = sq_error[b, 0][m].mean()
        rmse = torch.sqrt(mse)

        values.append(rmse.item())

    return values


@torch.no_grad()
def update_spatial_metrics(
    metric_store,
    pred,
    target,
    x,
    target_type,
    d0,
    obstacle_accum=None,
    z_tr=None,
    boundary_radius=3,
):
    masks = build_region_masks(
        x=x,
        target_type=target_type,
        d0=d0,
        obstacle_accum=obstacle_accum,
        z_tr=z_tr,
        boundary_radius=boundary_radius,
    )

    for region_name, mask in masks.items():
        vals = masked_rmse_per_sample(
            pred,
            target,
            mask,
        )
        metric_store[region_name].extend(vals)


def summarize_spatial_metrics(metric_store):
    result = {}

    for name, values in metric_store.items():
        t = torch.tensor(values, dtype=torch.float32)

        result[name] = {
            "rmse": t.mean().item(),
            "std": t.std(unbiased=False).item(),
            "num_samples": len(values),
        }

    return result