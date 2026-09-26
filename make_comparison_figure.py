from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import numpy as np

# ============================================================
# 1. 기본 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path("/home/ailab/Desktop/data/radiomapseer")

SAMPLES = [
    "65_13",
    "72_11",
    "122_79",
    "190_51",
]


# ============================================================
# 2. Baseline prediction 경로
# ============================================================

# 실제 폴더명이 다르면 오른쪽 경로만 수정하면 됨.
BASELINE_DIRS_DPM = OrderedDict([
    ("RadioUNet", PROJECT_ROOT / "baselines" / "RadioUNet" / "runs" / "radiounet_dpm" / "pred" / "test" / "rgb"),
    ("RME-GAN", PROJECT_ROOT / "baselines" / "RME-GAN" / "runs" / "rmegan_dpm" / "pred" / "test" / "rgb"),
    ("RadioDiff", PROJECT_ROOT / "baselines" / "RadioDiff" / "runs" / "radiodiff_dpm" / "pred" / "test" / "rgb"),
    ("RadioDiff-$k^2$", PROJECT_ROOT / "baselines" / "RadioDiff-k" / "runs" / "radiodiffk2_dpm" / "pred" / "test" / "rgb"),
    ("RadioMamba", PROJECT_ROOT / "baselines" / "RadioMamba" / "src" / "results" / "predictions_nocars_rgb"),
])

BASELINE_DIRS_CARSDPM = OrderedDict([
    ("RadioUNet", PROJECT_ROOT / "baselines" / "RadioUNet" / "runs" / "radiounet_carsdpm" / "pred" / "test" / "rgb"),
    ("RME-GAN", PROJECT_ROOT / "baselines" / "RME-GAN" / "runs" / "rmegan_carsdpm" / "pred" / "test" / "rgb"),
    ("RadioDiff", PROJECT_ROOT / "baselines" / "RadioDiff" / "runs" / "radiodiff_carsdpm" / "pred" / "test" / "rgb"),
    ("RadioDiff-$k^2$", PROJECT_ROOT / "baselines" / "RadioDiff-k" / "runs" / "radiodiffk2_carsdpm" / "pred" / "test" / "rgb"),
    ("RadioMamba", PROJECT_ROOT / "baselines" / "RadioMamba" / "src" / "results" / "predictions_withcars_rgb"),
])


# ============================================================
# 3. Ours prediction 경로
# ============================================================

OURS_DIRS = {
    "dpm": PROJECT_ROOT / "save_eval" / "ours_dpm" / "pred_png",
    "carsdpm": PROJECT_ROOT / "save_eval" / "ours_carsdpm" / "pred_png",
}

# ============================================================
# 4. Output
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GT_RGB_DIR = OUTPUT_DIR / "gt_rgb"
GT_RGB_DIR.mkdir(parents=True, exist_ok=True)


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
COLUMN_WIDTH = 1.55
ROW_HEIGHT = 1.55


# ============================================================
# 6. Image loading
# ============================================================

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
]


def find_image(directory: Path, sample_name: str):
    """Find exact sample image; return None instead of raising when missing."""

    # Exact filenames used by baseline RGB folders.
    for ext in IMAGE_EXTENSIONS:
        path = directory / f"{sample_name}{ext}"
        if path.exists():
            return path

    # RadioHRFormer evaluation currently may append "_mae_..." to prediction PNGs.
    for ext in IMAGE_EXTENSIONS:
        candidates = sorted(directory.glob(f"{sample_name}_mae_*{ext}"))
        if candidates:
            return candidates[0]

    print(
        f"[Warning] Image not found: sample={sample_name}, "
        f"directory={directory}"
    )
    return None


# ============================================================
# 7. Common RadioMapSeer RGB renderer
# ============================================================

def load_gray01(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

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


def build_radio_rgb(
    field01: np.ndarray,
    sample_name: str,
    target_type: str,
    obstacle_threshold: float = 0.5,
) -> np.ndarray:
    """Same convention as the RadioHRFormer evaluation.

    radio/path-gain : black -> yellow
    buildings       : blue
    cars            : red
    """

    field01 = np.asarray(field01, dtype=np.float32)
    field01 = np.squeeze(field01)
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
                "png/buildings_complete",
                sample_name,
            )
        )
        > obstacle_threshold
    )

    cars = np.zeros_like(building, dtype=bool)

    if str(target_type).lower() == "carsdpm":
        cars = (
            load_gray01(
                find_geometry_path(
                    "png/cars",
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


def load_gt_rgb(sample_name: str, target_type: str) -> np.ndarray:
    gain_subdir = (
        "gain/carsDPM"
        if str(target_type).lower() == "carsdpm"
        else "gain/DPM"
    )

    gt_path = DATA_ROOT / gain_subdir / f"{sample_name}.png"
    gt = load_gray01(gt_path)

    return build_radio_rgb(
        field01=gt,
        sample_name=sample_name,
        target_type=target_type,
    )


def save_gt_rgb(sample_name: str, target_type: str, rgb: np.ndarray):
    mode = "carsdpm" if str(target_type).lower() == "carsdpm" else "dpm"
    save_dir = GT_RGB_DIR / mode
    save_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb, mode="RGB").save(
        save_dir / f"{sample_name}.png"
    )


# ============================================================
# 8. Comparison figure
# ============================================================

def make_comparison_pdf(
    samples,
    baseline_dirs,
    ours_dir,
    output_path,
    target_type,
    ours_label="Ours",
):
    """
    rows    = samples
    columns = baselines + ours + GT
    """

    model_dirs = OrderedDict(baseline_dirs)
    model_dirs[ours_label] = ours_dir

    model_names = list(model_dirs.keys()) + ["GT"]

    n_rows = len(samples)
    n_cols = len(model_names)

    fig_width = COLUMN_WIDTH * n_cols
    fig_height = ROW_HEIGHT * n_rows + 0.40

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for row_idx, sample_name in enumerate(samples):

        # ----------------------------------------------------
        # Baselines + ours
        # ----------------------------------------------------
        for col_idx, (model_name, pred_dir) in enumerate(model_dirs.items()):
            ax = axes[row_idx, col_idx]

            img_path = find_image(
                pred_dir,
                sample_name,
            )

            # Missing model prediction -> blank cell.
            if img_path is None:
                ax.axis("off")
                continue

            with Image.open(img_path) as img:
                image = img.convert("RGB").copy()

            ax.imshow(image)
            ax.axis("off")
            ax.set_aspect("equal")

        # ----------------------------------------------------
        # GT at the far right
        # ----------------------------------------------------
        gt_col = n_cols - 1
        gt_ax = axes[row_idx, gt_col]

        gt_rgb = load_gt_rgb(
            sample_name=sample_name,
            target_type=target_type,
        )
        save_gt_rgb(
            sample_name=sample_name,
            target_type=target_type,
            rgb=gt_rgb,
        )

        gt_ax.imshow(gt_rgb)
        gt_ax.axis("off")
        gt_ax.set_aspect("equal")

    plt.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.995,
        bottom=0.07,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    for col_idx, model_name in enumerate(model_names):
        bottom_ax = axes[-1, col_idx]
        bbox = bottom_ax.get_position()

        x_center = (bbox.x0 + bbox.x1) / 2
        label_y = bbox.y0 - 0.022

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
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.015,
        dpi=300,
    )

    plt.close(fig)
    print(f"[Saved] {output_path}")


# ============================================================
# 9. Run
# ============================================================

if __name__ == "__main__":

    make_comparison_pdf(
        samples=SAMPLES,
        baseline_dirs=BASELINE_DIRS_DPM,
        ours_dir=OURS_DIRS["dpm"],
        output_path=OUTPUT_DIR / "qualitative_dpm.pdf",
        target_type="DPM",
        ours_label="RadioHRFormer",
    )

    make_comparison_pdf(
        samples=SAMPLES,
        baseline_dirs=BASELINE_DIRS_CARSDPM,
        ours_dir=OURS_DIRS["carsdpm"],
        output_path=OUTPUT_DIR / "qualitative_carsdpm.pdf",
        target_type="carsDPM",
        ours_label="RadioHRFormer",
    )
