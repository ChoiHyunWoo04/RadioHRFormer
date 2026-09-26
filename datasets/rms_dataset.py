import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from skimage import io


def normalize_target_type(target_type: str) -> str:
    """Normalize config target names to the canonical RadioMapSeer folder names."""
    key = str(target_type).strip().lower()
    mapping = {
        "dpm": "DPM",
        "carsdpm": "carsDPM",
        "irt4": "IRT4",
        "carsirt4": "carsIRT4",
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported target_type='{target_type}'. "
            "Use one of: DPM, carsDPM, IRT4, carsIRT4."
        )
    return mapping[key]


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
        thresh: float = 0.0,
        return_name: bool = True
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

        target_type = normalize_target_type(target_type)
        if target_type not in ["DPM", "carsDPM"]:
            raise ValueError(
                "RadioMapSeerDataset handles only DPM/carsDPM. "
                "Use RadioMapSeerIRT4Dataset for IRT4/carsIRT4."
            )

        self.root_dir = root_dir
        self.num_tx = num_tx
        self.target_type = target_type
        self.thresh = thresh
        self.return_name = return_name

        self.dir_buildings = os.path.join(root_dir, "png", "buildings_complete")
        self.dir_tx = os.path.join(root_dir, "png", "antennas")
        self.dir_gain = os.path.join(root_dir, "gain", target_type)
        self.dir_cars = os.path.join(root_dir, "png", "cars") if target_type == "carsDPM" else None

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

    def __getitem__(self, idx):
        map_idx_in_split = idx // self.num_tx
        tx_idx_in_map = idx % self.num_tx
        dataset_map_ind = int(self.maps_inds[self.ind1 + map_idx_in_split]) + 1

        building_name = f"{dataset_map_ind}.png"
        sample_name = f"{dataset_map_ind}_{tx_idx_in_map}.png"

        building = self._read_gray(os.path.join(self.dir_buildings, building_name))
        tx = self._read_gray(os.path.join(self.dir_tx, sample_name))
        gain = self._read_gray(os.path.join(self.dir_gain, sample_name))

        if self.target_type == "carsDPM":
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

        if self.return_name:
            return x, y, sample_name
        return x, y


def build_dataloaders(cfg, return_datasets: bool = False):
    data_cfg = cfg["data"]

    target_type = normalize_target_type(data_cfg.get("target_type", "carsDPM"))
    num_tx = data_cfg.get("num_tx", 80)
    thresh = data_cfg.get("thresh", 0.0)
    return_name = data_cfg.get("return_name", False)

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    persistent_workers = data_cfg.get("persistent_workers", False) and num_workers > 0

    train_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        split="train",
        num_tx=num_tx,
        target_type=target_type,
        thresh=thresh,
        return_name=return_name
    )

    val_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        split="val",
        num_tx=num_tx,
        target_type=target_type,
        thresh=thresh,
        return_name=return_name
    )

    test_dataset = RadioMapSeerDataset(
        root_dir=data_cfg["root_dir"],
        split="test",
        num_tx=num_tx,
        target_type=target_type,
        thresh=thresh,
        return_name=return_name
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



class RadioMapSeerIRT4Dataset(Dataset):
    """RadioMapSeer IRT4 dataset using the same map-level split as DPM/carsDPM.

    IRT4 is available only for Tx indices 0 and 1 of each urban map.

    Input x: [3, H, W]
        target_type='IRT4'     : [building, building, Tx]
        target_type='carsIRT4' : [building, cars, Tx]

    Target y: [1, H, W]
        target_type='IRT4'     : gain/IRT4
        target_type='carsIRT4' : gain/carsIRT4

    target_type directly identifies the requested prediction target.
    Thus DPM, carsDPM, IRT4, and carsIRT4 can be selected through one
    config field without a separate eval_dataset switch.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        maps_inds=None,
        num_tx: int = 2,
        target_type: str = "carsIRT4",
        thresh: float = 0.0,
        return_name: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.split = "val" if split == "valid" else split
        self.num_tx = int(num_tx)
        self.target_type = normalize_target_type(target_type)
        self.thresh = float(thresh)
        self.return_name = bool(return_name)

        if self.target_type not in ["IRT4", "carsIRT4"]:
            raise ValueError(
                "RadioMapSeerIRT4Dataset handles only IRT4/carsIRT4."
            )

        if self.num_tx < 1 or self.num_tx > 2:
            raise ValueError(
                f"IRT4 provides only Tx indices 0 and 1; num_tx must be 1 or 2, got {self.num_tx}."
            )

        if maps_inds is None:
            self.maps_inds = np.arange(0, 700, dtype=np.int16)
            rng = np.random.RandomState(int(seed))
            rng.shuffle(self.maps_inds)
        else:
            self.maps_inds = np.asarray(maps_inds)

        # Same 550/50/100 map split used by the main RadioMapSeer experiments.
        if self.split == "train":
            self.ind1, self.ind2 = 0, 549
        elif self.split == "val":
            self.ind1, self.ind2 = 550, 599
        elif self.split == "test":
            self.ind1, self.ind2 = 600, 699
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        self.dir_buildings = os.path.join(root_dir, "png", "buildings_complete")
        self.dir_tx = os.path.join(root_dir, "png", "antennas")

        self.dir_gain = os.path.join(root_dir, "gain", self.target_type)
        self.dir_cars = (
            os.path.join(root_dir, "png", "cars")
            if self.target_type == "carsIRT4"
            else None
        )

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.num_tx

    @staticmethod
    def _read_gray(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"IRT4 file not found: {path}")
        img = io.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _check_same_shape(*arrays):
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) != 1:
            raise ValueError(f"Shape mismatch among loaded IRT4 maps: {shapes}")

    def __getitem__(self, idx):
        map_idx_in_split = idx // self.num_tx
        tx_idx_in_map = idx % self.num_tx
        dataset_map_ind = int(self.maps_inds[self.ind1 + map_idx_in_split]) + 1

        building_name = f"{dataset_map_ind}.png"
        sample_name = f"{dataset_map_ind}_{tx_idx_in_map}.png"

        building = self._read_gray(os.path.join(self.dir_buildings, building_name))
        tx = self._read_gray(os.path.join(self.dir_tx, sample_name))
        gain = self._read_gray(os.path.join(self.dir_gain, sample_name))

        if self.target_type == "carsIRT4":
            second_channel = self._read_gray(os.path.join(self.dir_cars, building_name))
        else:
            second_channel = building

        self._check_same_shape(building, second_channel, tx, gain)

        # Keep the same target preprocessing as the DPM/carsDPM loader.
        if self.thresh > 0:
            gain = np.maximum(gain, self.thresh)
            gain = (gain - self.thresh) / (1.0 - self.thresh + 1e-8)

        x_np = np.stack([building, second_channel, tx], axis=0)
        y_np = gain[None, :, :]

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

        if self.return_name:
            return x, y, sample_name
        return x, y


def build_irt4_dataloaders(cfg, return_datasets: bool = False):
    """Build train/val/test loaders for IRT4 or carsIRT4.

    This mirrors build_dataloaders() so training and evaluation code can switch
    builders without changing the returned dictionary/tuple interface.
    """
    data_cfg = cfg["data"]

    target_type = normalize_target_type(data_cfg.get("target_type", "carsIRT4"))
    num_tx = int(data_cfg.get("num_tx", 2))
    thresh = data_cfg.get("thresh", 0.0)
    return_name = data_cfg.get("return_name", False)
    seed = int(cfg.get("seed", 42))

    if target_type not in ["IRT4", "carsIRT4"]:
        raise ValueError("IRT4 builder requires target_type='IRT4' or 'carsIRT4'.")
    if num_tx < 1 or num_tx > 2:
        raise ValueError(
            f"IRT4 provides only Tx indices 0 and 1; set data.num_tx to 1 or 2, got {num_tx}."
        )

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", False)) and num_workers > 0

    common_kwargs = dict(
        root_dir=data_cfg["root_dir"],
        num_tx=num_tx,
        target_type=target_type,
        thresh=thresh,
        return_name=return_name,
        seed=seed,
    )

    train_dataset = RadioMapSeerIRT4Dataset(split="train", **common_kwargs)
    val_dataset = RadioMapSeerIRT4Dataset(split="val", **common_kwargs)
    test_dataset = RadioMapSeerIRT4Dataset(split="test", **common_kwargs)

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

    # Keep batch=1 for final test evaluation/profiling, matching the main loader.
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
