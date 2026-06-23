import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from skimage import io


class RadioMapSeerDataset(Dataset):
    """
    RadioMapSeer dataset for HRFormer-based radio map regression.

    Input x: [3, H, W]
        cars_input=False: [building, building, Tx-Gaussian]
        cars_input=True : [building, cars, Tx-Gaussian]

    Target y: [1, H, W]
        target_type='DPM'     : no-car DPM radio map
        target_type='carsDPM' : car-aware DPM radio map

    Augmentation:
        If augment=True, the same random resized crop and flips are applied
        to both x and y to preserve spatial alignment.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        maps_inds=None,
        num_tx: int = 80,
        target_type: str = "DPM",
        cars_input: bool = False,
        thresh: float = 0.0,
        return_name: bool = True,
        augment: bool = False,
        crop_scale=(0.8, 1.0),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        use_tx_gaussian_map: bool = True,
        tx_gaussian_sigma: float = 6.0,
    ):
        super().__init__()

        if maps_inds is None:
            self.maps_inds = np.arange(0, 700, dtype=np.int16)
            rng = np.random.RandomState(42)
            rng.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if split == "train":
            self.ind1 = 0
            self.ind2 = 549
        elif split in ["val", "valid"]:
            self.ind1 = 550
            self.ind2 = 599
        elif split == "test":
            self.ind1 = 600
            self.ind2 = 699
        else:
            raise ValueError(f"Unsupported split: {split}")

        if target_type not in ["DPM", "carsDPM"]:
            raise ValueError("target_type must be either 'DPM' or 'carsDPM'.")

        self.root_dir = root_dir
        self.num_tx = num_tx
        self.target_type = target_type
        self.cars_input = cars_input
        self.thresh = thresh
        self.return_name = return_name

        self.augment = augment
        self.crop_scale = crop_scale
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        
        self.use_tx_gaussian_map = use_tx_gaussian_map
        self.tx_gaussian_sigma = tx_gaussian_sigma

        self.dir_buildings = os.path.join(root_dir, "png", "buildings_complete")
        self.dir_tx = os.path.join(root_dir, "png", "antennas")
        self.dir_gain = os.path.join(root_dir, "gain", target_type)
        self.dir_cars = os.path.join(root_dir, "png", "cars") if cars_input else None

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.num_tx

    @staticmethod
    def _read_gray(path: str) -> np.ndarray:
        img = io.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _check_same_shape(*arrays):
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) != 1:
            raise ValueError(f"Shape mismatch among loaded maps: {shapes}")

    @staticmethod
    def _tx_to_gaussian(tx_img: np.ndarray, sigma: float = 6.0) -> np.ndarray:
        """
        Convert Tx antenna image to Gaussian heatmap.

        tx_img: [H, W], usually sparse binary/gray Tx image.
        return: [H, W], Gaussian map normalized to [0, 1].
        """
        h, w = tx_img.shape

        # Tx 위치 추정: nonzero pixel들의 centroid 사용
        ys, xs = np.where(tx_img > 0)

        if len(xs) == 0:
            # 혹시 Tx 이미지가 완전히 비어 있으면 최댓값 위치 사용
            cy, cx = np.unravel_index(np.argmax(tx_img), tx_img.shape)
        else:
            cy = ys.mean()
            cx = xs.mean()

        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )

        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        gaussian = gaussian.astype(np.float32)

        max_val = gaussian.max()
        if max_val > 0:
            gaussian = gaussian / max_val

        return gaussian

    def _augment_xy(self, x: torch.Tensor, y: torch.Tensor):
        """
        Apply the same square random resized crop and flips to x and y.

        x: [3, H, W]
        y: [1, H, W]
        """
        '''_, h, w = x.shape

        min_scale, max_scale = self.crop_scale
        scale = float(torch.empty(1).uniform_(min_scale, max_scale).item())

        # 정사각형 crop
        base_size = min(h, w)
        crop_size = max(1, int(round(base_size * scale)))

        top = int(torch.randint(0, h - crop_size + 1, (1,)).item())
        left = int(torch.randint(0, w - crop_size + 1, (1,)).item())

        # 동일한 crop parameter를 x, y에 적용
        x = TF.resized_crop(
            x,
            top=top,
            left=left,
            height=crop_size,
            width=crop_size,
            size=[h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        y = TF.resized_crop(
            y,
            top=top,
            left=left,
            height=crop_size,
            width=crop_size,
            size=[h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )'''

        if torch.rand(1).item() < self.hflip_prob:
            x = TF.hflip(x)
            y = TF.hflip(y)

        if torch.rand(1).item() < self.vflip_prob:
            x = TF.vflip(x)
            y = TF.vflip(y)

        return x, y

    def __getitem__(self, idx):
        map_idx_in_split = idx // self.num_tx
        tx_idx_in_map = idx % self.num_tx
        dataset_map_ind = int(self.maps_inds[self.ind1 + map_idx_in_split]) + 1

        building_name = f"{dataset_map_ind}.png"
        sample_name = f"{dataset_map_ind}_{tx_idx_in_map}.png"

        building = self._read_gray(os.path.join(self.dir_buildings, building_name))
        tx = self._read_gray(os.path.join(self.dir_tx, sample_name))
        gain = self._read_gray(os.path.join(self.dir_gain, sample_name))

        # Tx image -> Gaussian map
        if self.use_tx_gaussian_map:
            tx = self._tx_to_gaussian(tx, sigma=self.tx_gaussian_sigma)

        if self.cars_input:
            second_channel = self._read_gray(os.path.join(self.dir_cars, building_name))
        else:
            second_channel = building

        self._check_same_shape(building, second_channel, tx, gain)

        if self.thresh > 0:
            gain = np.maximum(gain, self.thresh)
            gain = (gain - self.thresh) / (1.0 - self.thresh + 1e-8)

        x_np = np.stack([building, second_channel, tx], axis=0)  # [3, H, W]
        y_np = gain[None, :, :]                                  # [1, H, W]

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        if self.augment:
            x, y = self._augment_xy(x, y)

        if self.return_name:
            return x, y, sample_name
        return x, y


def build_dataloaders(cfg, return_datasets: bool = False):
    data_cfg = cfg["data"]

    cars_input = data_cfg.get("cars_input", False)
    target_type = data_cfg.get("target_type", "carsDPM" if cars_input else "DPM")
    num_tx = data_cfg.get("num_tx", 80)
    thresh = data_cfg.get("thresh", 0.0)
    return_name = data_cfg.get("return_name", False)

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    persistent_workers = data_cfg.get("persistent_workers", False) and num_workers > 0

    augment = data_cfg.get("augment", False)
    crop_scale = crop_scale = tuple(data_cfg.get("crop_scale", (0.8, 1.0)))
    hflip_prob = data_cfg.get("hflip_prob", 0.5)
    vflip_prob = data_cfg.get("vflip_prob", 0.5)
    use_tx_gaussian_map = data_cfg.get("use_tx_gaussian_map", True)
    tx_gaussian_sigma = data_cfg.get("tx_gaussian_sigma", 6.0)

    train_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        split="train",
        num_tx=num_tx,
        target_type=target_type,
        cars_input=cars_input,
        thresh=thresh,
        return_name=return_name,
        augment=augment,
        crop_scale=crop_scale,
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
        use_tx_gaussian_map=use_tx_gaussian_map,
        tx_gaussian_sigma=tx_gaussian_sigma,
    )

    val_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        split="val",
        num_tx=num_tx,
        target_type=target_type,
        cars_input=cars_input,
        thresh=thresh,
        return_name=return_name,
        augment=False,
        use_tx_gaussian_map=use_tx_gaussian_map,
        tx_gaussian_sigma=tx_gaussian_sigma,
    )

    test_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        split="test",
        num_tx=num_tx,
        target_type=target_type,
        cars_input=cars_input,
        thresh=thresh,
        return_name=return_name,
        augment=False,
        use_tx_gaussian_map=use_tx_gaussian_map,
        tx_gaussian_sigma=tx_gaussian_sigma,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    if return_datasets:
        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }

    return train_loader, val_loader, test_loader