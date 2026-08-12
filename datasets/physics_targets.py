from typing import Dict, Iterable, Optional
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsTargetBuilder(nn.Module):
    """Load precomputed obstacle transmission maps.

    Expected keys:
        obstacle_saturating_a005
        obstacle_saturating_a006
        obstacle_saturating_a007
        obstacle_saturating_a008
        obstacle_saturating_a009

    Each map is already stored as:
        exp(-alpha * obstruction_length)
    """

    PRECOMPUTED_GEO_TARGETS = {
        "obstacle_saturating_a005",
        "obstacle_saturating_a006",
        "obstacle_saturating_a007",
        "obstacle_saturating_a008",
        "obstacle_saturating_a009",
    }

    VALID_TARGETS = {
        "radial_gain",
        "obstacle_saturating_a005",
        "obstacle_saturating_a006",
        "obstacle_saturating_a007",
        "obstacle_saturating_a008",
        "obstacle_saturating_a009",
    }

    def __init__(
        self,
        target_names: Iterable[str] = ("radial_gain", "obstacle_saturating_a007"),
        tx_channel: int = -1,
        geo_precompute_root: Optional[str] = None,
        geo_mode_name: Optional[str] = None,
        geo_split: str = "train",
    ):
        super().__init__()

        self.target_names = self.normalize_target_names(target_names)
        self._validate_target_names(self.target_names)

        self.tx_channel = int(tx_channel)

        self.geo_precompute_root = geo_precompute_root
        self.geo_mode_name = geo_mode_name
        self.geo_split = "val" if geo_split == "valid" else str(geo_split)

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
        invalid = [name for name in target_names if name not in cls.VALID_TARGETS]
        if invalid:
            raise ValueError(
                f"Unsupported physics target(s): {invalid}. "
                f"Valid targets are: {sorted(cls.VALID_TARGETS)}"
            )

    @staticmethod
    def head_specs(target_names):
        target_names = PhysicsTargetBuilder.normalize_target_names(target_names)
        PhysicsTargetBuilder._validate_target_names(target_names)
        return {name: 1 for name in target_names}

    @staticmethod
    def _safe_stem(name):
        name = str(name)
        name = os.path.basename(name)
        if name.endswith(".png"):
            name = name[:-4]
        return name.replace(os.sep, "_").replace(" ", "_")

    def _precomputed_geo_path(self, name: str):
        if self.geo_precompute_root is None:
            raise ValueError("geo_precompute_root is None.")
        if self.geo_mode_name is None:
            raise ValueError(
                "geo_mode_name must be provided when geo_precompute_root is used. "
                "Example: cars_carsDPM or building_DPM."
            )

        stem = self._safe_stem(name)
        return os.path.join(
            self.geo_precompute_root,
            self.geo_mode_name,
            self.geo_split,
            f"{stem}.pt",
        )

    def _load_precomputed_geo(self, names, requested_targets, device, dtype):
        """Load precomputed ray-obstruction maps.

        Compatibility behavior:
            - `obstacle_sum` loads the new `obstacle_sum` key.
            - `obstacle_saturating_a003` / `_a005` load their exact keys.
            - legacy `obstacle` first tries `obstacle`; if absent, it falls back
              to `obstacle_sum` so older training commands remain usable.
        """
        if names is None:
            raise ValueError(
                "names must be provided to load precomputed geometry targets. "
                "Set cfg['data']['return_name']=True in the pretraining pipeline."
            )

        requested_targets = [
            target for target in requested_targets
            if target in self.PRECOMPUTED_GEO_TARGETS
        ]
        loaded = {key: [] for key in requested_targets}

        for name in names:
            path = self._precomputed_geo_path(name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing precomputed geometry target: {path}\n"
                    "Check --geo-precompute-root, --geo-mode-name, --eval-split, and sample names."
                )

            data = torch.load(path, map_location="cpu")

            for requested_key in requested_targets:
                if requested_key not in data:
                    raise KeyError(
                        f"'{requested_key}' was not found in {path}. "
                        f"Available keys: {list(data.keys())}"
                    )

                loaded[requested_key].append(data[requested_key].float())

        return {
            key: torch.stack(values, dim=0).to(device=device, dtype=dtype)
            for key, values in loaded.items()
        }

    def _select_channel(self, x, channel_idx: int):
        """Return x[:, channel_idx:channel_idx+1] with safe negative-index support."""
        c = x.size(1)

        if channel_idx < 0:
            channel_idx = c + channel_idx

        if channel_idx < 0 or channel_idx >= c:
            raise IndexError(
                f"Invalid channel index {channel_idx} for input with {c} channels."
            )

        return x[:, channel_idx:channel_idx + 1]

    @torch.no_grad()
    def _radial_gain(self, x):
        """Build a label-free log-distance gain prior from Tx locations.

        Output semantics:
            - 1.0 at the transmitter pixel.
            - Monotonic decay with Tx-pixel distance.
            - Map-diagonal normalization, rather than per-sample max distance,
              keeps the target scale comparable across Tx locations.

        This map is intentionally computed online: it requires only O(HW)
        arithmetic and is negligible compared with backbone forward/backward cost.
        """
        tx = self._select_channel(x, self.tx_channel).float()
        batch_size, _, height, width = tx.shape
        device = x.device
        dtype = x.dtype

        flat = tx.flatten(2).argmax(dim=-1).squeeze(1)
        tx_y = (flat // width).to(dtype=torch.float32)
        tx_x = (flat % width).to(dtype=torch.float32)

        yy = torch.arange(height, device=device, dtype=torch.float32).view(1, 1, height, 1)
        xx = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, 1, width)

        distance = torch.sqrt(
            (yy - tx_y.view(batch_size, 1, 1, 1)) ** 2
            + (xx - tx_x.view(batch_size, 1, 1, 1)) ** 2
        )

        max_distance = math.sqrt((height - 1) ** 2 + (width - 1) ** 2)
        radial_gain = 1.0 - torch.log1p(distance) / math.log1p(max_distance)
        return radial_gain.clamp(0.0, 1.0).to(dtype=dtype)

    def forward(self, x, names=None) -> Dict[str, torch.Tensor]:
        target_names = self.normalize_target_names(self.target_names)
        self._validate_target_names(target_names)

        targets = {}

        # ------------------------------------------------------------------
        # Input-driven global propagation prior. This never reads y or .pt files.
        # ------------------------------------------------------------------
        if "radial_gain" in target_names:
            targets["radial_gain"] = self._radial_gain(x)

        # ------------------------------------------------------------------
        # Ray-obstruction targets. New scripts precompute obstacle_sum and two
        # saturating variants. `obstacle` remains a legacy alias / fallback.
        # ------------------------------------------------------------------
        requested_geo = [
            target for target in target_names
            if target in self.PRECOMPUTED_GEO_TARGETS
        ]
        if requested_geo:
            if self.geo_precompute_root is not None:
                geo_targets = self._load_precomputed_geo(
                    names=names,
                    requested_targets=requested_geo,
                    device=x.device,
                    dtype=x.dtype,
                )

            for name in requested_geo:
                target = geo_targets[name].clamp(0.0, 1.0)
                targets[name] = target

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