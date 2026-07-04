from typing import Dict, Iterable, Optional
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsTargetBuilder(nn.Module):
    """Build physics-pretraining targets for RadioMapSeer batches.

    Supported targets:
        grad: |nabla u|, first-order spatial transition of the label-derived field.
        lap: |Delta u|, second-order local curvature of the label-derived field.
        singularity: RadioDiff-k2-inspired inverted negative-k2 map.
        radial_gain: label-free log-distance Tx gain prior computed online from x.
        obstacle_sum: precomputed normalized ray-obstruction integral.
        obstacle_saturating_a003: precomputed 1-exp(-0.03 * obstruction length).
        obstacle_saturating_a005: precomputed 1-exp(-0.05 * obstruction length).

    Obstacle inversion:
        Set invert_obstacle_targets=True to transform any loaded obstacle map
        into 1 - map. This turns obstruction intensity into a transmission-like
        prior: high on unobstructed paths and low after stronger blockage.

    Expected input layout:
        cars mode     : [building, cars, Tx]
        building mode : [building, building, Tx]

    Precomputed geometry layout:
        <geo_precompute_root>/<geo_mode_name>/<geo_split>/<sample_name>.pt

    Current obstacle-only .pt files should contain:
        {
            "obstacle_sum": Tensor[1,H,W],
            "obstacle_saturating_a003": Tensor[1,H,W],
            "obstacle_saturating_a005": Tensor[1,H,W],
        }

    `radial_gain` is deliberately generated online because it only requires the
    Tx location and a lightweight distance-grid calculation.
    """

    SINGULARITY_ALIASES = {
        "singularity",
    }

    PRECOMPUTED_GEO_TARGETS = {
        "obstacle_sum",
        "obstacle_saturating_a003",
        "obstacle_saturating_a005",
    }

    VALID_TARGETS = {
        "grad",
        "lap",
        "singularity",
        "radial_gain",
        "obstacle_sum",
        "obstacle_saturating_a003",
        "obstacle_saturating_a005",
    }

    def __init__(
        self,
        target_names: Iterable[str] = ("radial_gain", "obstacle_saturating_a005"),
        field_mode: str = "normalized_power",
        gaussian_sigma: float = 2.0,
        eps: float = 1e-4,
        tx_channel: int = -1,
        building_threshold: float = 0.5,
        obstacle_channels=(0, 1),
        normalize_each_sample: bool = True,
        ray_stride: int = 1,
        invert_obstacle_targets: bool = False,
        radiodiff_pathloss_trunc: float = -147.0,
        radiodiff_pathloss_max: float = -47.0,
        radiodiff_source_power_dbm: float = 23.0,
        radiodiff_h: float = 1.0,
        radiodiff_border_value: float = 1.0,
        radiodiff_eps: float = 1e-30,
        radiodiff_smooth_sigma: float = 0.9,
        geo_precompute_root: Optional[str] = None,
        geo_mode_name: Optional[str] = None,
        geo_split: str = "train",
    ):
        super().__init__()

        self.target_names = self.normalize_target_names(target_names)
        self._validate_target_names(self.target_names)

        self.field_mode = field_mode
        self.gaussian_sigma = float(gaussian_sigma)
        self.eps = float(eps)

        self.tx_channel = int(tx_channel)
        self.building_threshold = float(building_threshold)
        self.obstacle_channels = tuple(int(c) for c in obstacle_channels)
        self.normalize_each_sample = bool(normalize_each_sample)
        self.ray_stride = max(1, int(ray_stride))
        self.invert_obstacle_targets = bool(invert_obstacle_targets)

        self.radiodiff_pathloss_trunc = float(radiodiff_pathloss_trunc)
        self.radiodiff_pathloss_max = float(radiodiff_pathloss_max)
        self.radiodiff_source_power_dbm = float(radiodiff_source_power_dbm)
        self.radiodiff_h = float(radiodiff_h)
        self.radiodiff_border_value = float(radiodiff_border_value)
        self.radiodiff_eps = float(radiodiff_eps)
        self.radiodiff_smooth_sigma = float(radiodiff_smooth_sigma)
        
        self.geo_precompute_root = geo_precompute_root
        self.geo_mode_name = geo_mode_name
        self.geo_split = "val" if geo_split == "valid" else str(geo_split)

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0],
                 [-2.0, 0.0, 2.0],
                 [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 8.0,
            persistent=False,
        )

        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0],
                 [0.0, 0.0, 0.0],
                 [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 8.0,
            persistent=False,
        )

        self.register_buffer(
            "lap_kernel",
            torch.tensor(
                [[0.0, 1.0, 0.0],
                 [1.0, -4.0, 1.0],
                 [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )

    @classmethod
    def canonical_target_name(cls, name: str) -> str:
        name = str(name).strip()
        if name in cls.SINGULARITY_ALIASES:
            return "singularity"
        return name

    @classmethod
    def normalize_target_names(cls, target_names: Iterable[str]):
        names = [
            cls.canonical_target_name(name)
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
                if requested_key == "obstacle":
                    candidate_keys = ("obstacle", "obstacle_sum")
                elif requested_key == "obstacle_sum":
                    candidate_keys = ("obstacle_sum", "obstacle")
                else:
                    candidate_keys = (requested_key,)

                source_key = next(
                    (key for key in candidate_keys if key in data),
                    None,
                )
                if source_key is None:
                    raise KeyError(
                        f"None of {candidate_keys} was found in {path}. "
                        f"Available keys: {list(data.keys())}"
                    )

                loaded[requested_key].append(data[source_key].float())

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

    def _build_obstacle_map(self, x):
        maps = [self._select_channel(x, ch).float() for ch in self.obstacle_channels]
        obstacle = torch.stack(maps, dim=0).amax(dim=0)
        return (obstacle > self.building_threshold).float()

    def _gaussian_blur(self, y, sigma=None):
        sigma = float(self.gaussian_sigma if sigma is None else sigma)
        if sigma <= 0:
            return y

        radius = max(1, math.ceil(3.0 * sigma))
        coords = torch.arange(-radius, radius + 1, device=y.device, dtype=y.dtype)
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kx = kernel_1d.view(1, 1, 1, -1)
        ky = kernel_1d.view(1, 1, -1, 1)

        y = F.pad(y, (radius, radius, 0, 0), mode="reflect")
        y = F.conv2d(y, kx)
        y = F.pad(y, (0, 0, radius, radius), mode="reflect")
        y = F.conv2d(y, ky)
        return y

    def _to_field_amplitude(self, y):
        """Convert label y to a stable field proxy for grad/lap targets."""
        if self.field_mode == "normalized_power":
            return torch.clamp(y, min=self.eps)
        if self.field_mode == "db_power":
            return torch.pow(10.0, y / 20.0).clamp_min(self.eps)
        if self.field_mode == "pathloss_db":
            return torch.pow(10.0, -y / 20.0).clamp_min(self.eps)
        raise ValueError(f"Unsupported field_mode: {self.field_mode}")

    def _gradient(self, u):
        gx = F.conv2d(u, self.sobel_x.to(dtype=u.dtype), padding=1)
        gy = F.conv2d(u, self.sobel_y.to(dtype=u.dtype), padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def _laplacian(self, u):
        return F.conv2d(u, self.lap_kernel.to(dtype=u.dtype), padding=1)

    def _standardize(self, z):
        if not self.normalize_each_sample:
            return z
        mean = z.mean(dim=(2, 3), keepdim=True)
        std = z.std(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return (z - mean) / std

    def _minmax(self, z):
        zmin = z.amin(dim=(2, 3), keepdim=True)
        zmax = z.amax(dim=(2, 3), keepdim=True)
        return (z - zmin) / (zmax - zmin + self.eps)

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

    def _singularity_target(self, y, x):
        y = y.float().clamp(0.0, 1.0)
        if self.radiodiff_smooth_sigma > 0:
            y = self._gaussian_blur(y, sigma=self.radiodiff_smooth_sigma)
            y = y.clamp(0.0, 1.0)

        tx = self._select_channel(x, self.tx_channel).float()
        pathloss_db = self.radiodiff_pathloss_trunc + (
            self.radiodiff_pathloss_max - self.radiodiff_pathloss_trunc
        ) * y
        source_power_db = self.radiodiff_source_power_dbm - 30.0
        power_db = source_power_db + pathloss_db

        # Use field-amplitude-like scaling for stable pretraining target.
        # This is intentionally more stable than the original power-scale 10^(P/10).
        u = torch.pow(10.0, power_db / 20.0)
        center = u[:, :, 1:-1, 1:-1]
        delta_u = (
            u[:, :, 2:, 1:-1]
            + u[:, :, :-2, 1:-1]
            + u[:, :, 1:-1, 2:]
            + u[:, :, 1:-1, :-2]
            - 4.0 * center
        ) / (self.radiodiff_h ** 2)

        tx_center = tx[:, :, 1:-1, 1:-1]
        tmp = (delta_u + tx_center * center) / center.clamp_min(self.radiodiff_eps)
        k2_neg = torch.where(tmp < 0.0, tmp, torch.zeros_like(tmp))

        zmin = k2_neg.amin(dim=(2, 3), keepdim=True)
        zmax = k2_neg.amax(dim=(2, 3), keepdim=True)
        k2_neg_norm = (k2_neg - zmin) / (zmax - zmin + self.eps)

        k2_neg_norm = F.pad(
            k2_neg_norm,
            pad=(1, 1, 1, 1),
            mode="constant",
            value=self.radiodiff_border_value,
        ).clamp(0.0, 1.0)

        # Inverted direction:
        #   important negative-k2 structures are high/bright.
        singularity = 1.0 - k2_neg_norm
        return singularity.clamp(0.0, 1.0)

    @torch.no_grad()
    def _tx_centers(self, tx_map):
        b, _, h, w = tx_map.shape
        flat = tx_map.flatten(2).argmax(dim=-1).squeeze(1)
        yy = (flat // w).long()
        xx = (flat % w).long()
        return yy, xx

    @torch.no_grad()
    def _visibility_and_obstacle(self, x):
        obstacle_mask = self._build_obstacle_map(x)
        tx = self._select_channel(x, self.tx_channel)
        b, _, h, w = obstacle_mask.shape
        device = x.device
        yy_tx, xx_tx = self._tx_centers(tx)

        obs = torch.zeros((b, 1, h, w), device=device, dtype=x.dtype)
        ys = torch.arange(0, h, self.ray_stride, device=device)
        xs = torch.arange(0, w, self.ray_stride, device=device)

        for bi in range(b):
            y0 = int(yy_tx[bi].item())
            x0 = int(xx_tx[bi].item())
            for y1 in ys.tolist():
                for x1 in xs.tolist():
                    dx = x1 - x0
                    dy = y1 - y0
                    n = max(abs(dx), abs(dy), 1) + 1
                    rr = torch.linspace(y0, y1, n, device=device).round().long().clamp(0, h - 1)
                    cc = torch.linspace(x0, x1, n, device=device).round().long().clamp(0, w - 1)
                    hit = obstacle_mask[bi, 0, rr, cc].sum()
                    obs[bi, 0, y1, x1] = hit

        if self.ray_stride > 1:
            obs = F.interpolate(obs, size=(h, w), mode="bilinear", align_corners=False)
        return obs

    def forward(self, x, y, names=None) -> Dict[str, torch.Tensor]:
        target_names = self.normalize_target_names(self.target_names)
        self._validate_target_names(target_names)

        targets = {}

        # ------------------------------------------------------------------
        # Label-driven ablation targets.
        # ------------------------------------------------------------------
        need_label_targets = any(t in target_names for t in ("grad", "lap"))
        if need_label_targets:
            u = self._to_field_amplitude(y.float())
            u = self._gaussian_blur(u)
            if "grad" in target_names:
                targets["grad"] = self._standardize(self._gradient(u))
            if "lap" in target_names:
                targets["lap"] = self._standardize(self._laplacian(u).abs())

        if "singularity" in target_names:
            targets["singularity"] = self._singularity_target(y, x)

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
            else:
                # Online fallback for debugging only. Full training should use
                # precomputed targets because ray traversal is expensive.
                obstacle_raw = self._visibility_and_obstacle(x.float())
                obstacle_sum = self._minmax(obstacle_raw)

                geo_targets = {
                    "obstacle_sum": obstacle_sum.to(dtype=x.dtype),
                    "obstacle_saturating_a003": (
                        1.0 - torch.exp(-0.03 * obstacle_raw)
                    ).clamp(0.0, 1.0).to(dtype=x.dtype),
                    "obstacle_saturating_a005": (
                        1.0 - torch.exp(-0.05 * obstacle_raw)
                    ).clamp(0.0, 1.0).to(dtype=x.dtype),
                }

            for name in requested_geo:
                target = geo_targets[name]
                if self.invert_obstacle_targets:
                    target = 1.0 - target.clamp(0.0, 1.0)

                targets[name] = target

        return targets



class PhysicsPretrainLoss(nn.Module):
    """Multi-task loss for physics-map pretraining.

    Regression targets:
        grad, lap, radial_gain, obstacle_sum,
        obstacle_saturating_a003, obstacle_saturating_a005

    BCE targets:
        singularity
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        bce_names=("los", "singularity"),
    ):
        super().__init__()

        self.loss_weights = loss_weights or {}
        self.bce_names = set(bce_names)
        self.reg_loss = nn.SmoothL1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

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

            if name in self.bce_names:
                loss = self.bce_loss(pred, target)
            else:
                loss = self.reg_loss(pred, target)

            weight = float(self.loss_weights.get(name, 1.0))
            weighted_loss = weight * loss
            logs[name] = float(loss.detach().cpu())
            total = weighted_loss if total is None else total + weighted_loss

        if total is None:
            raise ValueError("No targets were provided to PhysicsPretrainLoss.")

        logs["loss"] = float(total.detach().cpu())
        return total, logs