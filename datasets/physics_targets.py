from typing import Dict, Iterable, Optional
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsTargetBuilder(nn.Module):
    """Build input-driven auxiliary targets for pretraining ablation.

    Supported targets
    -----------------
    radial_gain:
        Simple Tx-centered distance prior.
        High near the Tx and decreases linearly with Euclidean distance.

    los:
        Binary line-of-sight prior.
        Derived from the same precomputed obstruction-transmittance map
        used by the proposed target.

    obstacle_saturating_a007:
        Proposed path-wise obstruction-transmittance target:
            exp(-0.07 * A(p))
        where A(p) is accumulated obstacle length along the Tx-to-pixel ray.
    """

    VALID_TARGETS = {
        "radial_gain",
        "los",
        "obstacle_saturating_a007",
    }

    TRANSMITTANCE_KEY = "obstacle_saturating_a007"

    def __init__(
        self,
        target_names: Iterable[str] = ("obstacle_saturating_a007",),
        tx_channel: int = -1,
        geo_precompute_root: Optional[str] = None,
        geo_mode_name: Optional[str] = None,
        geo_split: str = "train",
        transmittance_alpha: float = 0.07,
    ):
        super().__init__()

        self.target_names = self.normalize_target_names(target_names)
        self._validate_target_names(self.target_names)

        self.tx_channel = int(tx_channel)

        self.geo_precompute_root = geo_precompute_root
        self.geo_mode_name = geo_mode_name
        self.geo_split = "val" if geo_split == "valid" else str(geo_split)

        self.transmittance_alpha = float(transmittance_alpha)

    # ---------------------------------------------------------
    # Common helpers
    # ---------------------------------------------------------

    @classmethod
    def normalize_target_names(cls, target_names: Iterable[str]):
        names = [
            str(name).strip()
            for name in target_names
            if str(name).strip()
        ]
        return tuple(dict.fromkeys(names))

    @classmethod
    def _validate_target_names(cls, target_names):
        invalid = [
            name
            for name in target_names
            if name not in cls.VALID_TARGETS
        ]

        if invalid:
            raise ValueError(
                f"Unsupported pretraining target(s): {invalid}. "
                f"Valid targets are: {sorted(cls.VALID_TARGETS)}"
            )

    @staticmethod
    def head_specs(target_names):
        target_names = PhysicsTargetBuilder.normalize_target_names(
            target_names
        )
        PhysicsTargetBuilder._validate_target_names(target_names)

        return {
            name: 1
            for name in target_names
        }

    @staticmethod
    def _safe_stem(name):
        name = str(name)
        name = os.path.basename(name)

        if name.endswith(".png"):
            name = name[:-4]

        return (
            name
            .replace(os.sep, "_")
            .replace(" ", "_")
        )

    def _select_channel(self, x, channel_idx: int):
        """Safely select one channel, including negative indices."""

        c = x.size(1)

        if channel_idx < 0:
            channel_idx = c + channel_idx

        if channel_idx < 0 or channel_idx >= c:
            raise IndexError(
                f"Invalid channel index {channel_idx} "
                f"for input with {c} channels."
            )

        return x[:, channel_idx:channel_idx + 1]

    # ---------------------------------------------------------
    # Precomputed transmittance
    # ---------------------------------------------------------

    def _precomputed_geo_path(self, name: str):
        if self.geo_precompute_root is None:
            raise ValueError(
                "geo_precompute_root must be provided for "
                "'los' or 'obstacle_saturating_a007'."
            )

        if self.geo_mode_name is None:
            raise ValueError(
                "geo_mode_name must be provided when loading "
                "precomputed geometry targets. "
                "Example: cars_carsDPM or building_DPM."
            )

        stem = self._safe_stem(name)

        return os.path.join(
            self.geo_precompute_root,
            self.geo_mode_name,
            self.geo_split,
            f"{stem}.pt",
        )

    @staticmethod
    def _normalize_loaded_map(value, path):
        """Convert one stored map to [1, H, W]."""

        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        value = value.float()

        # [H, W] -> [1, H, W]
        if value.ndim == 2:
            value = value.unsqueeze(0)

        # [1, 1, H, W] -> [1, H, W]
        elif value.ndim == 4 and value.size(0) == 1:
            value = value.squeeze(0)

        if value.ndim != 3 or value.size(0) != 1:
            raise ValueError(
                f"Expected stored target to have shape "
                f"[H,W], [1,H,W], or [1,1,H,W], "
                f"but got {tuple(value.shape)} in {path}"
            )

        return value

    def _load_transmittance(
        self,
        names,
        device,
        dtype,
    ):
        """Load exp(-0.07 * A(p)) maps as [B, 1, H, W]."""

        if names is None:
            raise ValueError(
                "Sample names are required for LoS/transmittance "
                "pretraining. Set cfg['data']['return_name']=True."
            )

        loaded = []

        for name in names:
            path = self._precomputed_geo_path(name)

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing precomputed geometry target: {path}"
                )

            data = torch.load(
                path,
                map_location="cpu",
            )

            if not isinstance(data, dict):
                raise TypeError(
                    f"Expected a dict in {path}, "
                    f"but got {type(data)}."
                )

            if self.TRANSMITTANCE_KEY not in data:
                raise KeyError(
                    f"'{self.TRANSMITTANCE_KEY}' "
                    f"was not found in {path}. "
                    f"Available keys: {list(data.keys())}"
                )

            value = self._normalize_loaded_map(
                data[self.TRANSMITTANCE_KEY],
                path,
            )

            loaded.append(value)

        target = torch.stack(
            loaded,
            dim=0,
        )

        return target.to(
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)

    # ---------------------------------------------------------
    # Ablation targets
    # ---------------------------------------------------------

    def _build_radial_gain(self, x):
        """Simple normalized Tx-distance prior.

        radial_gain = 1 - d(Tx, p) / d_max

        d_max is the diagonal length of the image, giving a fixed
        normalization independent of the Tx location.
        """

        tx_map = self._select_channel(
            x,
            self.tx_channel,
        )

        B, _, H, W = tx_map.shape
        device = x.device
        dtype = x.dtype

        # Robust even if the Tx map is Gaussian rather than binary.
        flat_idx = tx_map.reshape(B, -1).argmax(dim=1)

        tx_y = (
            flat_idx // W
        ).to(dtype=dtype).view(B, 1, 1, 1)

        tx_x = (
            flat_idx % W
        ).to(dtype=dtype).view(B, 1, 1, 1)

        yy = torch.arange(
            H,
            device=device,
            dtype=dtype,
        ).view(1, 1, H, 1)

        xx = torch.arange(
            W,
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, W)

        distance = torch.sqrt(
            (yy - tx_y) ** 2
            + (xx - tx_x) ** 2
        )

        d_max = math.sqrt(
            float((H - 1) ** 2 + (W - 1) ** 2)
        )

        d_max = max(d_max, 1.0)

        radial_gain = (
            1.0 - distance / d_max
        ).clamp(0.0, 1.0)

        return radial_gain

    def _build_los_from_transmittance(
        self,
        transmittance,
    ):
        """Recover binary LoS from exp(-alpha * A).

        A = 0 -> LoS
        A >= 1 -> NLoS

        Recovering A rather than comparing exactly against 1.0 makes
        the operation robust to float16 storage.
        """

        eps = 1e-6

        z = transmittance.clamp(
            min=eps,
            max=1.0,
        )

        obstruction = (
            -torch.log(z)
            / self.transmittance_alpha
        )

        # Integer obstruction accumulation:
        # A = 0 -> LoS
        # A >= 1 -> NLoS
        los = (
            obstruction < 0.5
        ).to(dtype=transmittance.dtype)

        return los

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(
        self,
        x,
        names=None,
    ) -> Dict[str, torch.Tensor]:

        target_names = self.normalize_target_names(
            self.target_names
        )
        self._validate_target_names(
            target_names
        )

        targets = {}

        # LoS and proposed transmittance share exactly
        # the same precomputed ray-obstruction source.
        need_transmittance = (
            "los" in target_names
            or "obstacle_saturating_a007" in target_names
        )

        transmittance = None

        if need_transmittance:
            transmittance = self._load_transmittance(
                names=names,
                device=x.device,
                dtype=x.dtype,
            )

        for name in target_names:

            if name == "radial_gain":
                targets[name] = self._build_radial_gain(x)

            elif name == "los":
                targets[name] = (
                    self._build_los_from_transmittance(
                        transmittance
                    )
                )

            elif name == "obstacle_saturating_a007":
                targets[name] = transmittance

            else:
                # Should already be caught by validation,
                # but keep this branch defensive.
                raise RuntimeError(
                    f"Unhandled target: {name}"
                )

        return targets

class PhysicsPretrainLoss(nn.Module):
    """Multi-task regression loss for propagation-prior pretraining."""

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.loss_weights = loss_weights or {}
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        total = None
        logs = {}

        for name, target in targets.items():
            if name not in preds:
                raise KeyError(
                    f"Prediction for target '{name}' is missing. "
                    f"Available predictions: {list(preds.keys())}"
                )

            pred = preds[name]

            if pred.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(
                    target,
                    size=pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = self.reg_loss(pred, target)
            weight = float(self.loss_weights.get(name, 1.0))
            weighted_loss = weight * loss

            logs[name] = float(loss.detach().cpu())
            total = weighted_loss if total is None else total + weighted_loss

        if total is None:
            raise ValueError("No targets were provided to PhysicsPretrainLoss.")

        logs["loss"] = float(total.detach().cpu())
        return total, logs