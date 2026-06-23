import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Seed-fixing utility
def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FormattingString:
    @staticmethod
    def show(values):
        raise NotImplementedError("The formatting string used is not implemented.")
  
       
class DefaultFormattingString(FormattingString):
    @staticmethod
    def show(values):
        gb_formatting = lambda value : value / 1024**3
        return f"""Memory Amount Used
 - Alloccated Memory: {gb_formatting(values[0]):.2f} GB
 - Reserved Memory: {gb_formatting(values[1]):.2f} GB"""


def show_current_cuda_memory(formatting_string=DefaultFormattingString):
    """
    Check current CUDA memory usage.
    """
    cuda = torch.cuda
    formatting = lambda value : value / 1024**3
    allocated_memory = cuda.memory_allocated()
    reserved_memory = cuda.memory_reserved()
    print(formatting_string.show((allocated_memory, reserved_memory)))
#     print(f"Allocated memory: {formatting(allocated_memory):.2f} GB")
#     print(f"Reserved memory: {formatting(reserved_memory):.2f} GB")


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


def use_amp_on_device(device, amp_enabled=True) -> bool:
    return bool(amp_enabled and is_cuda_device(device))


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