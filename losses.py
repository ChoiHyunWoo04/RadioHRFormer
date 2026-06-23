import torch
import torch.nn as nn
import torch.nn.functional as F

# l1_loss
def MAE(y_pred, y_true):
    loss = torch.mean(torch.abs(torch.sub(y_true, y_pred)))
    return loss

def MAE_MSE(y_pred, y_true):
    loss_mae = torch.mean(torch.abs(y_pred - y_true))
    loss_mse = torch.mean(((y_pred - y_true) ** 2))
    loss = (loss_mae + loss_mse) / 2
    return loss


class JointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        sobel_x = torch.tensor([[1,0,-1],
                                [2,0,-2],
                                [1,0,-1]], dtype=torch.float32).view(1,1,3,3)

        sobel_y = torch.tensor([[1,2,1],
                                [0,0,0],
                                [-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, x):
        B, C, H, W = x.shape

        sobel_x = self.sobel_x.repeat(C,1,1,1)
        sobel_y = self.sobel_y.repeat(C,1,1,1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)

        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return grad

    def forward(self, pred, target):

        # MAE
        L_MAE = self.l1(pred, target)

        # MSE
        #L_MSE = self.mse(pred, target)

        # Gradient loss
        grad_pred = self.gradient(pred)
        grad_target = self.gradient(target)
        L_Grad = self.l1(grad_pred, grad_target)

        loss = 0.7 * L_MAE + 0.3 * L_Grad# 0.2 * L_MSE + + 0.2 * L_SSIM

        return loss
