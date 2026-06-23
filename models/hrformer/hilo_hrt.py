# --------------------------------------------------------
# HiLoHRT: High-Low Resolution High Resolution Transformer
# Modified from HRT / HRFormer
# --------------------------------------------------------

import math
import torch
import torch.nn as nn

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import constant_init, normal_init
from mmengine.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from models.utils.ops import resize
from models.hrformer.modules.bottleneck_block import Bottleneck
from models.hrformer.modules.transformer_block import GeneralTransformerBlock


class ScaleAwareHighResolutionTransformerModule(nn.Module):
    """
    Two-branch or multi-branch HRT module with explicit branch stride information.

    branch_strides:
        Relative stride of each branch with respect to the highest-resolution branch.
        For this model:
            branch0: stride 1  -> 64x64
            branch1: stride 8  -> 8x8
    """

    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        in_channels,
        num_channels,
        multiscale_output,
        branch_strides,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        num_heads=None,
        num_window_sizes=None,
        num_mlp_ratios=None,
        drop_paths=0.0,
    ):
        super().__init__()

        self._check_branches(num_branches, num_blocks, in_channels, num_channels)
        self._check_branch_strides(num_branches, branch_strides)

        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.branch_strides = branch_strides

        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        self.branches = self._make_branches(
            num_branches=num_branches,
            block=blocks,
            num_blocks=num_blocks,
            num_channels=num_channels,
            num_heads=num_heads,
            num_window_sizes=num_window_sizes,
            num_mlp_ratios=num_mlp_ratios,
            drop_paths=drop_paths,
        )

        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios

    def _check_branches(self, num_branches, num_blocks, in_channels, num_channels):
        if num_branches != len(num_blocks):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})"
            )

        if num_branches != len(num_channels):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})"
            )

        if num_branches != len(in_channels):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> IN_CHANNELS({len(in_channels)})"
            )

    def _check_branch_strides(self, num_branches, branch_strides):
        if num_branches != len(branch_strides):
            raise ValueError(
                f"NUM_BRANCHES({num_branches}) <> BRANCH_STRIDES({len(branch_strides)})"
            )

        for s in branch_strides:
            if s < 1 or int(s) != s:
                raise ValueError(f"Invalid branch stride: {s}")

        for i in range(len(branch_strides) - 1):
            if branch_strides[i + 1] % branch_strides[i] != 0:
                raise ValueError(
                    f"branch_strides must be divisible: {branch_strides}"
                )

    @staticmethod
    def _num_downsample_steps(src_stride, dst_stride):
        """
        src_stride < dst_stride means spatial downsampling is required.
        Example:
            src_stride=1, dst_stride=8 -> 3 stride-2 downsamples.
        """
        ratio = dst_stride // src_stride
        if ratio < 1 or dst_stride % src_stride != 0:
            raise ValueError(f"Invalid stride ratio: {src_stride} -> {dst_stride}")

        steps = int(math.log2(ratio))
        if 2 ** steps != ratio:
            raise ValueError(
                f"Only power-of-two stride ratios are supported. Got ratio={ratio}."
            )
        return steps

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
        stride=1,
    ):
        downsample = None
        expected_out_channels = num_channels[branch_index] * block.expansion

        if stride != 1 or self.in_channels[branch_index] != expected_out_channels:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    expected_out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, expected_out_channels)[1],
            )

        layers = []

        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                drop_path=drop_paths[0],
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
            )
        )

        self.in_channels[branch_index] = expected_out_channels

        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    branch_index=i,
                    block=block,
                    num_blocks=num_blocks,
                    num_channels=num_channels,
                    num_heads=num_heads,
                    num_window_sizes=num_window_sizes,
                    num_mlp_ratios=num_mlp_ratios,
                    drop_paths=drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """
        Build scale-aware fusion layers.

        For branch_strides=[1, 8]:
            low -> high:
                8x8 -> 64x64 upsample by 8
            high -> low:
                64x64 -> 8x8 downsample by 2 three times
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels

        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1

        for i in range(num_out_branches):
            fuse_layer = []

            dst_stride = self.branch_strides[i]
            dst_channels = in_channels[i]

            for j in range(num_branches):
                src_stride = self.branch_strides[j]
                src_channels = in_channels[j]

                if j == i:
                    fuse_layer.append(None)

                elif src_stride > dst_stride:
                    # lower-resolution branch -> higher-resolution branch
                    # Example: 8x8 -> 64x64
                    upsample_scale = src_stride // dst_stride

                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                src_channels,
                                dst_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, dst_channels)[1],
                            nn.Upsample(
                                scale_factor=upsample_scale,
                                mode="bilinear",
                                align_corners=False,
                            ),
                        )
                    )

                else:
                    # higher-resolution branch -> lower-resolution branch
                    # Example: 64x64 -> 8x8
                    num_downsamples = self._num_downsample_steps(
                        src_stride=src_stride,
                        dst_stride=dst_stride,
                    )

                    conv_downsamples = []
                    cur_channels = src_channels

                    for k in range(num_downsamples):
                        is_last = k == num_downsamples - 1
                        out_channels = dst_channels if is_last else cur_channels

                        conv_downsamples.append(
                            nn.Sequential(
                                build_conv_layer(
                                    self.conv_cfg,
                                    cur_channels,
                                    cur_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=cur_channels,
                                    bias=False,
                                ),
                                build_norm_layer(self.norm_cfg, cur_channels)[1],
                                build_conv_layer(
                                    self.conv_cfg,
                                    cur_channels,
                                    out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                                build_norm_layer(self.norm_cfg, out_channels)[1],
                                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
                            )
                        )

                        cur_channels = out_channels

                    fuse_layer.append(nn.Sequential(*conv_downsamples))

            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])

            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    fused = self.fuse_layers[i][j](x[j])
                    if fused.shape[-2:] != x[i].shape[-2:]:
                        fused = resize(
                            fused,
                            size=x[i].shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    y = y + fused

            x_fuse.append(self.relu(y))

        return x_fuse


class HiLoHRT(nn.Module):
    """
    HiLoHRT backbone.

    Designed for 256x256 radio-map input.

    Resolution flow:
        input 256x256
        conv1 -> 128x128
        conv2 -> 64x64
        stage1 -> 64x64

        transition1:
            branch0: 64x64, high-resolution local branch
            branch1: 8x8, low-resolution global branch

        stage2/3/4:
            two branches repeatedly perform window attention and scale-aware fusion.

        final:
            stage4 multiscale_output=False returns fused branch0 only.
    """

    blocks_dict = {
        "BOTTLENECK": Bottleneck,
        "TRANSFORMER_BLOCK": GeneralTransformerBlock,
    }

    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
    ):
        super().__init__()

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # branch0 stride=1 means 64x64.
        # branch1 stride=8 means 8x8.
        self.branch_strides = self.extra.get("branch_strides", [1, 8])

        # Stem
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)

        # Drop path schedule
        depth_s2 = (
            self.extra["stage2"]["num_blocks"][0]
            * self.extra["stage2"]["num_modules"]
        )
        depth_s3 = (
            self.extra["stage3"]["num_blocks"][0]
            * self.extra["stage3"]["num_modules"]
        )
        depth_s4 = (
            self.extra["stage4"]["num_blocks"][0]
            * self.extra["stage4"]["num_modules"]
        )

        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = self.extra["drop_path_rate"]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Stage 1
        self.stage1_cfg = self.extra["stage1"]
        stage1_channels = self.stage1_cfg["num_channels"][0]
        stage1_block_type = self.stage1_cfg["block"]
        stage1_num_blocks = self.stage1_cfg["num_blocks"][0]

        block = self.blocks_dict[stage1_block_type]
        stage1_out_channels = stage1_channels * block.expansion

        self.layer1 = self._make_layer(
            block=block,
            inplanes=64,
            planes=stage1_channels,
            blocks=stage1_num_blocks,
        )

        # Stage 2
        self.stage2_cfg = self.extra["stage2"]
        self._check_hilo_stage_config(self.stage2_cfg, "stage2")

        stage2_channels = self._expanded_channels(self.stage2_cfg)

        # Custom transition:
        # [B, 256, 64, 64] ->
        #   branch0: [B, 32, 64, 64]
        #   branch1: [B, 256, 8, 8]
        self.transition1 = self._make_initial_hilo_transition(
            in_channels=stage1_out_channels,
            out_channels_list=stage2_channels,
        )

        self.stage2, pre_stage_channels = self._make_stage(
            layer_config=self.stage2_cfg,
            in_channels=stage2_channels,
            multiscale_output=True,
            drop_paths=dpr[0:depth_s2],
        )

        # Stage 3
        self.stage3_cfg = self.extra["stage3"]
        self._check_hilo_stage_config(self.stage3_cfg, "stage3")

        stage3_channels = self._expanded_channels(self.stage3_cfg)
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,
            stage3_channels,
        )

        self.stage3, pre_stage_channels = self._make_stage(
            layer_config=self.stage3_cfg,
            in_channels=stage3_channels,
            multiscale_output=True,
            drop_paths=dpr[depth_s2 : depth_s2 + depth_s3],
        )

        # Stage 4
        self.stage4_cfg = self.extra["stage4"]
        self._check_hilo_stage_config(self.stage4_cfg, "stage4")

        stage4_channels = self._expanded_channels(self.stage4_cfg)
        self.transition3 = self._make_transition_layer(
            pre_stage_channels,
            stage4_channels,
        )

        self.stage4, pre_stage_channels = self._make_stage(
            layer_config=self.stage4_cfg,
            in_channels=stage4_channels,
            multiscale_output=self.stage4_cfg.get("multiscale_output", False),
            drop_paths=dpr[depth_s2 + depth_s3 :],
        )

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def _check_hilo_stage_config(self, cfg, stage_name):
        if cfg["num_branches"] != 2:
            raise ValueError(
                f"{stage_name} must have exactly 2 branches for HiLoHRT. "
                f"Got {cfg['num_branches']}."
            )

        required_keys = [
            "num_blocks",
            "num_channels",
            "num_heads",
            "num_mlp_ratios",
            "num_window_sizes",
        ]

        for key in required_keys:
            if len(cfg[key]) != 2:
                raise ValueError(
                    f"{stage_name}.{key} must have length 2. Got {len(cfg[key])}."
                )

    def _expanded_channels(self, stage_cfg):
        block = self.blocks_dict[stage_cfg["block"]]
        return [c * block.expansion for c in stage_cfg["num_channels"]]

    def _make_initial_hilo_transition(self, in_channels, out_channels_list):
        """
        Build the initial two-branch transition before stage2.

        Input:
            [B, 256, 64, 64]

        Output:
            branch0:
                [B, 32, 64, 64]

            branch1:
                [B, 256, 8, 8]
        """
        high_channels = out_channels_list[0]
        low_channels = out_channels_list[1]

        high_branch = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                high_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self.norm_cfg, high_channels)[1],
            nn.ReLU(inplace=True),
        )

        low_branch = nn.Sequential(
            # 64 -> 32
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                low_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self.norm_cfg, low_channels)[1],
            nn.ReLU(inplace=True),

            # 32 -> 16
            build_conv_layer(
                self.conv_cfg,
                low_channels,
                low_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self.norm_cfg, low_channels)[1],
            nn.ReLU(inplace=True),

            # 16 -> 8
            build_conv_layer(
                self.conv_cfg,
                low_channels,
                low_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            build_norm_layer(self.norm_cfg, low_channels)[1],
            nn.ReLU(inplace=True),
        )

        return nn.ModuleList([high_branch, low_branch])

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """
        Transition between stage2 -> stage3 and stage3 -> stage4.

        In the recommended HiLoHRT config, both stages have channels [32, 256],
        so this usually returns [None, None].

        This is kept for flexibility in case stage channels are changed later.
        """
        if len(num_channels_pre_layer) != 2 or len(num_channels_cur_layer) != 2:
            raise ValueError("HiLoHRT transition expects exactly two branches.")

        transition_layers = []

        for i in range(2):
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                transition_layers.append(
                    nn.Sequential(
                        build_conv_layer(
                            self.conv_cfg,
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1],
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                transition_layers.append(None)

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        stride=1,
        num_heads=1,
        window_size=8,
        mlp_ratio=4.0,
    ):
        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
            )
        )

        inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config,
        in_channels,
        multiscale_output=True,
        drop_paths=0.0,
    ):
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = self.blocks_dict[layer_config["block"]]

        num_heads = layer_config["num_heads"]
        num_window_sizes = layer_config["num_window_sizes"]
        num_mlp_ratios = layer_config["num_mlp_ratios"]

        hr_modules = []

        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            start = num_blocks[0] * i
            end = num_blocks[0] * (i + 1)

            hr_modules.append(
                ScaleAwareHighResolutionTransformerModule(
                    num_branches=num_branches,
                    blocks=block,
                    num_blocks=num_blocks,
                    in_channels=in_channels,
                    num_channels=num_channels,
                    multiscale_output=reset_multiscale_output,
                    branch_strides=self.branch_strides,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    num_heads=num_heads,
                    num_window_sizes=num_window_sizes,
                    num_mlp_ratios=num_mlp_ratios,
                    drop_paths=drop_paths[start:end],
                )
            )

        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            ckpt = load_checkpoint(self, pretrained, strict=False)
            if "model" in ckpt:
                self.load_state_dict(ckpt["model"], strict=False)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)

        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, return_stem_features=False):
        """
        Return:
            y_list:
                If stage4.multiscale_output=False:
                    [branch0_fused]
                    branch0_fused shape: [B, 32, 64, 64]

                If stage4.multiscale_output=True:
                    [branch0_fused, branch1_fused]
        """
        # Stem
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        stem1 = x  # [B, 64, 128, 128]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        stem2 = x  # [B, 64, 64, 64]

        # Stage 1
        x = self.layer1(x)  # [B, 256, 64, 64]

        # Initial Hi-Lo transition before stage2
        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            x_list.append(self.transition1[i](x))

        # stage2
        y_list = self.stage2(x_list)

        # stage3 transition
        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[i]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)

        # stage4 transition
        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[i]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage4(x_list)

        if return_stem_features:
            return y_list, {
                "stem1": stem1,
                "stem2": stem2,
            }

        return y_list

    def train(self, mode=True):
        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()