import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


# l1_loss
def MAE(y_pred, y_true):
    loss = torch.mean(torch.abs(torch.sub(y_true, y_pred)))
    return loss

def MSE(y_pred, y_true):
    loss = torch.mean(((y_pred - y_true) ** 2))
    return loss


class JointLoss(nn.Module):
    def __init__(
        self,
        mae_weight=0.5,
        grad_weight=0.3,
        ssim_weight=0.2,
        data_range=1.0,
    ):
        super().__init__()

        self.mae_weight = mae_weight
        self.grad_weight = grad_weight
        self.ssim_weight = ssim_weight

        self.data_range = data_range

        self.l1 = nn.L1Loss()

        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, x):
        _, c, _, _ = x.shape

        sobel_x = self.sobel_x.repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=c)

        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

    def compute_ms_ssim_loss(self, pred, target):
        """
        MS-SSIM은 반드시 FP32에서 계산.
        pred/target shape: [B, C, H, W]
        target range: [0, 1] 가정
        """

        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()

            ms_ssim_score = ms_ssim(
                pred_fp32,
                target_fp32,
                data_range=self.data_range,
                size_average=True,
            )

            loss_ms_ssim = 1.0 - ms_ssim_score

        return loss_ms_ssim

    def forward(self, pred, target):
        # Pixel-level MAE
        loss_mae = self.l1(pred, target)

        # Gradient consistency
        grad_pred = self.gradient(pred)
        grad_target = self.gradient(target)
        loss_grad = self.l1(grad_pred, grad_target)

        # Structural similarity: FP32 only
        loss_ssim = self.compute_ms_ssim_loss(pred, target)

        loss = (
            self.mae_weight * loss_mae
            + self.grad_weight * loss_grad
            + self.ssim_weight * loss_ssim
        )

        return loss