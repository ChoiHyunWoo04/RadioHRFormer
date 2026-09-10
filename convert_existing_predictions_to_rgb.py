from pathlib import Path
import argparse

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert existing 1-channel/NPY RadioMapSeer predictions to the "
            "paper RGB convention without rerunning model inference."
        )
    )
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ailab/Desktop/data/radiomapseer"),
    )
    parser.add_argument(
        "--dataset",
        choices=["dpm", "carsdpm"],
        required=True,
    )
    parser.add_argument("--obstacle-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D prediction, got {arr.shape}: {path}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def find_geometry_path(data_root: Path, subdir: str, stem: str) -> Path:
    map_id = stem.split("_")[0]
    candidates = [
        data_root / subdir / f"{map_id}.png",
        data_root / subdir / f"{stem}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Geometry map not found for {stem}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def load_mask(
    data_root: Path,
    subdir: str,
    stem: str,
    threshold: float,
) -> np.ndarray:
    path = find_geometry_path(data_root, subdir, stem)

    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    return arr > threshold


def build_rgb(
    field: np.ndarray,
    data_root: Path,
    stem: str,
    dataset: str,
    threshold: float,
) -> np.ndarray:
    building = load_mask(
        data_root,
        "png/buildings_complete",
        stem,
        threshold,
    )

    cars = np.zeros_like(building, dtype=bool)
    if dataset == "carsdpm":
        cars = load_mask(
            data_root,
            "png/cars",
            stem,
            threshold,
        )

    if building.shape != field.shape:
        raise ValueError(
            f"Building/field shape mismatch for {stem}: "
            f"{building.shape} vs {field.shape}"
        )

    if cars.shape != field.shape:
        raise ValueError(
            f"Cars/field shape mismatch for {stem}: "
            f"{cars.shape} vs {field.shape}"
        )

    rgb = np.zeros((*field.shape, 3), dtype=np.uint8)
    intensity = np.rint(field * 255.0).astype(np.uint8)

    # radio = yellow
    rgb[..., 0] = intensity
    rgb[..., 1] = intensity

    # building = blue, cars = red
    rgb[building] = [0, 0, 255]
    rgb[cars] = [255, 0, 0]

    return rgb


def main():
    args = parse_args()

    if not args.pred_dir.exists():
        raise FileNotFoundError(args.pred_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.pred_dir.glob("*.npy"))
    if not files:
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            files.extend(sorted(args.pred_dir.glob(ext)))

    if not files:
        raise FileNotFoundError(
            f"No prediction .npy/image files found in {args.pred_dir}"
        )

    for idx, path in enumerate(files, 1):
        stem = path.stem

        # RadioHRFormer-style names may include "_mae_...".
        if "_mae_" in stem:
            stem = stem.split("_mae_", 1)[0]

        field = load_prediction(path)
        rgb = build_rgb(
            field=field,
            data_root=args.data_root,
            stem=stem,
            dataset=args.dataset,
            threshold=args.obstacle_threshold,
        )

        Image.fromarray(rgb, mode="RGB").save(
            args.out_dir / f"{stem}.png"
        )

        if idx % 500 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] converted")

    print(f"[Done] RGB predictions saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
