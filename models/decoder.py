import torch
import torch.nn as nn
import torch.nn.functional as F

### for ablation study (skip connection X) ###
class RegressionDecoder(nn.Module):
    def __init__(self, backbone_channels, out_channels=1):
        super(RegressionDecoder, self).__init__()
        
        extended_dim = backbone_channels * 2 # 64
        
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 64x64 -> 128x128
            nn.Conv2d(backbone_channels, extended_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(extended_dim),
            nn.ReLU(),
            
            nn.Conv2d(extended_dim, extended_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(extended_dim), 
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 128x128 -> 256x256
            nn.Conv2d(extended_dim, extended_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(extended_dim),
            nn.ReLU(),
            
            nn.Conv2d(extended_dim, extended_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(extended_dim), 
            nn.ReLU(),
            
            nn.Conv2d(extended_dim, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):  # x: (B, C, 64, 64)
        return self.conv(x)  # (B, 1, 256, 256)

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(),
        )

    def forward(self, x):
        return self.block(x)


class HRFormerSkipDecoderBody(nn.Module):
    """Shared dense decoder body for HRFormer branch-0 features.

    This module contains all layers except the final task-specific 1x1 head.
    It is used by both downstream regression and physics pretraining.
    """

    def __init__(
        self,
        backbone_channels=48,
        stem2_channels=64,
        stem1_channels=64,
        hidden_channels=64,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.fuse_64 = nn.Sequential(
            ConvBNAct(backbone_channels + stem2_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )

        self.fuse_128 = nn.Sequential(
            ConvBNAct(hidden_channels + stem1_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )

        self.fuse_256_body = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )

    def forward(self, x, stem1, stem2):
        x = torch.cat([x, stem2], dim=1)
        x = self.fuse_64(x)

        x = F.interpolate(
            x,
            size=stem1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, stem1], dim=1)
        x = self.fuse_128(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.fuse_256_body(x)
        return x


class HRFormerSkipDecoder(nn.Module):
    """Downstream decoder: shared body + a single radio-map head."""

    def __init__(
        self,
        backbone_channels=48,
        stem2_channels=64,
        stem1_channels=64,
        hidden_channels=64,
        out_channels=1,
    ):
        super().__init__()
        self.body = HRFormerSkipDecoderBody(
            backbone_channels=backbone_channels,
            stem2_channels=stem2_channels,
            stem1_channels=stem1_channels,
            hidden_channels=hidden_channels,
        )
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward_features(self, x, stem1, stem2):
        return self.body(x, stem1=stem1, stem2=stem2)

    def forward(self, x, stem1, stem2):
        z = self.forward_features(x, stem1=stem1, stem2=stem2)
        return self.head(z)

    def reset_head(self, out_channels=1):
        in_channels = self.head.in_channels
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)


class MultiHeadPhysicsDecoder(nn.Module):
    """Pretraining decoder: one shared body and lightweight target-specific heads.

    Args:
        head_specs: dict such as {"grad": 1, "lap": 1, "k2": 1, "kneg": 1}
    Returns:
        dict[name, Tensor[B, C_name, H, W]]
    """

    def __init__(
        self,
        backbone_channels=48,
        stem2_channels=64,
        stem1_channels=64,
        hidden_channels=64,
        head_specs=None,
    ):
        super().__init__()
        if not head_specs:
            raise ValueError("head_specs must be a non-empty dict, e.g. {'grad': 1, 'lap': 1}.")

        self.body = HRFormerSkipDecoderBody(
            backbone_channels=backbone_channels,
            stem2_channels=stem2_channels,
            stem1_channels=stem1_channels,
            hidden_channels=hidden_channels,
        )
        self.heads = nn.ModuleDict({
            name: nn.Conv2d(hidden_channels, int(out_ch), kernel_size=1)
            for name, out_ch in head_specs.items()
        })

    def forward_features(self, x, stem1, stem2):
        return self.body(x, stem1=stem1, stem2=stem2)

    def forward(self, x, stem1, stem2):
        z = self.forward_features(x, stem1=stem1, stem2=stem2)
        return {name: head(z) for name, head in self.heads.items()}
