import os
import torch
import random
import numpy as np
from collections import defaultdict
from pathlib import Path


def get_default_run_name(config_path):
    """Derive a stable experiment name from the config filename."""
    stem = Path(config_path).stem

    for suffix in ("_pretrain", "_downstream"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem

# Seed-fixing utility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def prepare_device(cuda_visible_devices="0"):
    """
    Set CUDA_VISIBLE_DEVICES and return selected device.

    Args:
        cuda_visible_devices:
            "0", "1", "0,1" 등 GPU visible id.
            "", "cpu", "none", "-1"이면 CPU 사용.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if str(cuda_visible_devices).lower() in ["", "cpu", "none", "-1"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[Runtime] requested cuda        : {cuda_visible_devices}")
    print(f"[Runtime] CUDA_VISIBLE_DEVICES  : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Runtime] torch cuda available  : {torch.cuda.is_available()}")
    print(f"[Runtime] selected device       : {device}")

    if is_cuda_device(device):
        torch.cuda.empty_cache()
        print(f"[Runtime] current cuda device   : {torch.cuda.current_device()}")
        print(f"[Runtime] cuda device name      : {torch.cuda.get_device_name(torch.cuda.current_device())}")

    return device


def get_amp_device_type(device) -> str:
    return "cuda" if is_cuda_device(device) else "cpu"


def summarize_trainable_by_module(model):
    stats = defaultdict(lambda: {"trainable": 0, "frozen": 0})

    for name, param in model.named_parameters():
        module = name.split('.')[0]
        if param.requires_grad:
            stats[module]["trainable"] += param.numel()
        else:
            stats[module]["frozen"] += param.numel()

    print(f"{'Module':30s} | {'Trainable':>12s} | {'Frozen':>12s}")
    print("-" * 60)
    for m, v in stats.items():
        print(f"{m:30s} | {v['trainable']:12,d} | {v['frozen']:12,d}")