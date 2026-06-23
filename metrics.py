import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


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
def compute_ssim(pred, target):
    return ssim(pred, target, data_range=1.0)
