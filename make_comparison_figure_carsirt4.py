from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import numpy as np


# ============================================================
# 1. Basic settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/ailab/Desktop/data/radiomapseer")

# Replace these with the two carsIRT4 samples you want to show.
SAMPLES = [
    "202_0",
    "367_0",
]


# ============================================================
# 2. Prediction sources
# ============================================================
#
# Each entry can point to:
#   - a prediction root containing npy/png/rgb subfolders, or
#   - the npy/png/rgb folder itself.
#
# Search priority:
#   npy -> grayscale image -> existing RGB image
#
# Missing prediction -> blank cell.
# ============================================================

MODEL_DIRS = OrderedDict([
    (
        "RadioUNet",
        PROJECT_ROOT
        / "baselines"
        / "RadioUNet"
        / "runs_eval"
        / "radiounet_carsirt4"
        / "pred_png",
    ),
    (
        "RadioMamba",
        PROJECT_ROOT
        / "baselines"
        / "RadioMamba"
        / "results"
        / "predictions_carsirt4"
        / "pred_png",
    ),
    (
        "RadioHRFormer",
        PROJECT_ROOT
        / "save_eval"
        / "carsirt4_finetuned",
    ),
])


# ============================================================
# 3. Output
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_ROOT = OUTPUT_DIR / "carsirt4_rgb"
RGB_ROOT.mkdir(parents=True, exist_ok=True)

GT_RGB_DIR = RGB_ROOT / "GT"
GT_RGB_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = OUTPUT_DIR / "comparison_carsirt4.pdf"


# ============================================================
# 4. RadioMapSeer paths
# ============================================================

GT_SUBDIR = "gain/carsIRT4"
BUILDING_SUBDIR = "png/buildings_complete"
CARS_SUBDIR = "png/cars"

OBSTACLE_THRESHOLD = 0.5


# ============================================================
# 5. Figure style
# ============================================================

FONT_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
    "STIXGeneral",
]


def find_available_font():
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in FONT_CANDIDATES:
        if font_name in installed_fonts:
            print(f"[Font] Using: {font_name}")
            return font_name

    print("[Font] Times-like font not found. Using matplotlib serif fallback.")
    return "serif"


PAPER_FONT = find_available_font()

plt.rcParams.update({
    "font.family": PAPER_FONT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
})

MODEL_LABEL_FONTSIZE = 10
WSPACE = 0.015
HSPACE = 0.015
COLUMN_WIDTH = 1.70
ROW_HEIGHT = 1.70

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
]


# ============================================================
# 6. Common helpers
# ============================================================

def safe_name(name: str) -> str:
    return (
        str(name)
        .replace("$", "")
        .replace("^", "")
        .replace("{", "")
        .replace("}", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_gray01(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_npy01(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D prediction after squeeze, got {arr.shape}: {path}"
        )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return np.clip(arr, 0.0, 1.0)


def find_geometry_path(subdir: str, sample_name: str) -> Path:
    map_id = str(sample_name).split("_")[0]

    candidates = [
        DATA_ROOT / subdir / f"{map_id}.png",
        DATA_ROOT / subdir / f"{sample_name}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Geometry map not found for {sample_name}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def build_carsirt4_rgb(
    field01: np.ndarray,
    sample_name: str,
    obstacle_threshold: float = OBSTACLE_THRESHOLD,
) -> np.ndarray:
    """Common paper rendering:
       radio = black->yellow, building = blue, car = red.
    """
    field01 = np.asarray(field01, dtype=np.float32)
    field01 = np.squeeze(field01)

    if field01.ndim != 2:
        raise ValueError(
            f"Expected 2-D radio map for {sample_name}, got {field01.shape}"
        )

    field01 = np.nan_to_num(
        field01,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    field01 = np.clip(field01, 0.0, 1.0)

    building = (
        load_gray01(
            find_geometry_path(
                BUILDING_SUBDIR,
                sample_name,
            )
        )
        > obstacle_threshold
    )

    cars = (
        load_gray01(
            find_geometry_path(
                CARS_SUBDIR,
                sample_name,
            )
        )
        > obstacle_threshold
    )

    if building.shape != field01.shape:
        raise ValueError(
            f"Building/field mismatch for {sample_name}: "
            f"{building.shape} vs {field01.shape}"
        )

    if cars.shape != field01.shape:
        raise ValueError(
            f"Cars/field mismatch for {sample_name}: "
            f"{cars.shape} vs {field01.shape}"
        )

    rgb = np.zeros((*field01.shape, 3), dtype=np.uint8)
    intensity = np.rint(field01 * 255.0).astype(np.uint8)

    rgb[..., 0] = intensity
    rgb[..., 1] = intensity

    rgb[building] = [0, 0, 255]
    rgb[cars] = [255, 0, 0]

    return rgb


def save_rgb(rgb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


# ============================================================
# 7. Prediction loading + RGB conversion
# ============================================================

def find_prediction_source(directory: Path, sample_name: str):
    npy_candidates = [
        directory / "npy" / f"{sample_name}.npy",
        directory / "pred_npy" / f"{sample_name}.npy",
        directory / f"{sample_name}.npy",
    ]

    for path in npy_candidates:
        if path.exists():
            return "npy", path

    image_dirs = [
        directory / "png",
        directory / "pred_png",
        directory / "rgb",
        directory,
    ]

    for image_dir in image_dirs:
        for ext in IMAGE_EXTENSIONS:
            path = image_dir / f"{sample_name}{ext}"
            if path.exists():
                return "image", path

        for ext in IMAGE_EXTENSIONS:
            candidates = sorted(
                image_dir.glob(f"{sample_name}_mae_*{ext}")
            )
            if candidates:
                return "image", candidates[0]

    return None


def image_is_rgb(path: Path) -> bool:
    with Image.open(path) as image:
        return image.mode in {"RGB", "RGBA"}


def load_model_prediction_rgb(
    model_name: str,
    directory: Path,
    sample_name: str,
):
    source = find_prediction_source(
        directory,
        sample_name,
    )

    if source is None:
        print(
            f"[Warning] Missing prediction: "
            f"model={model_name}, sample={sample_name}, "
            f"directory={directory}"
        )
        return None

    source_type, source_path = source

    rgb_save_path = (
        RGB_ROOT
        / safe_name(model_name)
        / f"{sample_name}.png"
    )

    if source_type == "npy":
        field01 = load_npy01(source_path)
        rgb = build_carsirt4_rgb(field01, sample_name)
        save_rgb(rgb, rgb_save_path)
        return rgb

    # Grayscale image -> convert with the common renderer.
    if not image_is_rgb(source_path):
        field01 = load_gray01(source_path)
        rgb = build_carsirt4_rgb(field01, sample_name)
        save_rgb(rgb, rgb_save_path)
        return rgb

    # Existing RGB is used as a fallback (e.g. already-rendered HRFormer PNG).
    with Image.open(source_path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

    save_rgb(rgb, rgb_save_path)
    return rgb


# ============================================================
# 8. GT
# ============================================================

def load_gt_rgb(sample_name: str):
    gt_path = (
        DATA_ROOT
        / GT_SUBDIR
        / f"{sample_name}.png"
    )

    if not gt_path.exists():
        print(f"[Warning] Missing GT: {gt_path}")
        return None

    gt = load_gray01(gt_path)
    rgb = build_carsirt4_rgb(gt, sample_name)

    save_rgb(
        rgb,
        GT_RGB_DIR / f"{sample_name}.png",
    )

    return rgb


# ============================================================
# 9. 2 x 4 comparison figure
# ============================================================

def make_carsirt4_comparison_pdf():
    model_names = list(MODEL_DIRS.keys()) + ["GT"]

    n_rows = len(SAMPLES)
    n_cols = len(model_names)

    if n_rows != 2:
        print(
            f"[Warning] SAMPLES has {n_rows} samples. "
            "Use exactly two for a 2x4 figure."
        )

    fig_width = COLUMN_WIDTH * n_cols
    fig_height = ROW_HEIGHT * n_rows + 0.40

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for row_idx, sample_name in enumerate(SAMPLES):

        # RadioUNet / RadioDiff-k2 / RadioHRFormer
        for col_idx, (model_name, pred_dir) in enumerate(MODEL_DIRS.items()):
            ax = axes[row_idx, col_idx]

            rgb = load_model_prediction_rgb(
                model_name=model_name,
                directory=pred_dir,
                sample_name=sample_name,
            )

            # Missing prediction -> blank cell.
            if rgb is None:
                ax.axis("off")
                continue

            ax.imshow(rgb)
            ax.axis("off")
            ax.set_aspect("equal")

        # GT at the far right.
        gt_ax = axes[row_idx, n_cols - 1]
        gt_rgb = load_gt_rgb(sample_name)

        if gt_rgb is None:
            gt_ax.axis("off")
        else:
            gt_ax.imshow(gt_rgb)
            gt_ax.axis("off")
            gt_ax.set_aspect("equal")

    plt.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.995,
        bottom=0.09,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    for col_idx, model_name in enumerate(model_names):
        bottom_ax = axes[-1, col_idx]
        bbox = bottom_ax.get_position()

        x_center = (bbox.x0 + bbox.x1) / 2
        label_y = bbox.y0 - 0.028

        fig.text(
            x_center,
            label_y,
            model_name,
            ha="center",
            va="top",
            fontsize=MODEL_LABEL_FONTSIZE,
            fontfamily=PAPER_FONT,
        )

    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.015,
        dpi=300,
    )

    plt.close(fig)

    print(f"[Saved] {OUTPUT_PDF}")
    print(f"[Saved RGB panels] {RGB_ROOT}")


if __name__ == "__main__":
    make_carsirt4_comparison_pdf()
