import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from skimage.metrics import structural_similarity as ssim_metric


################################## Metric ##################################
# NMSA (NMSE)
def compute_nmse(pred, target):
    numerator = torch.sum((pred - target) ** 2)
    denominator = torch.sum(target ** 2) + 1e-8
    nmse = numerator / denominator
    return nmse

# RMSE (RMSA)
def compute_rmse(pred, target):
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

# PSNR
def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-8))
    return psnr

# SSIM
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