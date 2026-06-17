#!/usr/bin/env python3
"""Convert a CT NIfTI volume to the UniSpine-GS/X-Gaussian pickle format."""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "ctspine1k_spine.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_TIGRE_PYTHON = PROJECT_ROOT / "third_party" / "TIGRE-2.3" / "Python"


def import_tigre():
    try:
        import tigre  # type: ignore
        from tigre.utilities.geometry import Geometry  # type: ignore
    except Exception as exc:
        fallback = str(VENDORED_TIGRE_PYTHON)
        if VENDORED_TIGRE_PYTHON.exists() and fallback not in sys.path:
            sys.path.insert(0, fallback)
        try:
            import tigre  # type: ignore
            from tigre.utilities.geometry import Geometry  # type: ignore
        except Exception as fallback_exc:
            raise RuntimeError(
                "Cannot import TIGRE. Build the vendored TIGRE source from "
                "`third_party/TIGRE-2.3/Python` and verify that "
                "`python -c 'import tigre'` works before running conversion."
            ) from fallback_exc
    return tigre, Geometry


def convert_to_attenuation(data: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert CT Hounsfield units to an attenuation-like scalar field."""
    hu = data * slope + intercept
    mu_water = 0.206
    mu_air = 0.0004
    return mu_water + (mu_water - mu_air) / 1000.0 * hu


def select_3d_volume(data: np.ndarray, time_axis: Optional[int], time_index: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.squeeze(data)

    if data.ndim == 4:
        if time_axis is None:
            # CTSpine1K is expected to be 3D. For an unexpected 4D input, use
            # the smallest axis as the temporal/channel axis, which is the most
            # common layout for medical volumes with a short extra dimension.
            time_axis = int(np.argmin(data.shape))
        data = np.take(data, indices=time_index, axis=time_axis)
        data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError("Expected a 3D NIfTI volume after squeeze/select, got shape {}".format(data.shape))

    return data.astype(np.float32)


def load_nifti(path: Path, time_axis: Optional[int], time_index: int) -> np.ndarray:
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency nibabel. Install it with `pip install nibabel`.") from exc

    nii = nib.load(str(path))
    return select_3d_volume(nii.get_fdata(dtype=np.float32), time_axis, time_index)


def resize_volume(volume: np.ndarray, target_xyz: Tuple[int, int, int]) -> np.ndarray:
    try:
        import scipy.ndimage as ndi  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency scipy. Install it with `pip install scipy`.") from exc

    zoom = [target_xyz[i] / float(volume.shape[i]) for i in range(3)]
    if any(abs(z - 1.0) > 1e-6 for z in zoom):
        print("Resize CT volume from {} to {}".format(volume.shape, target_xyz))
        volume = ndi.zoom(volume, zoom=zoom, order=3, prefilter=False)
    return volume.astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    return volume.astype(np.float32)


def build_tigre_geometry(cfg: Dict, Geometry):
    geo = Geometry()
    geo.DSD = cfg["DSD"] / 1000.0
    geo.DSO = cfg["DSO"] / 1000.0

    geo.nDetector = np.array(cfg["nDetector"])
    geo.dDetector = np.array(cfg["dDetector"]) / 1000.0
    geo.sDetector = geo.nDetector * geo.dDetector

    geo.nVoxel = np.array(cfg["nVoxel"][::-1])
    geo.dVoxel = np.array(cfg["dVoxel"][::-1]) / 1000.0
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    geo.offOrigin = np.array(cfg["offOrigin"][::-1]) / 1000.0
    geo.offDetector = np.array([cfg["offDetector"][1], cfg["offDetector"][0], 0]) / 1000.0

    geo.accuracy = cfg["accuracy"]
    geo.mode = cfg["mode"]
    geo.filter = cfg["filter"]
    return geo


def build_angles(cfg: Dict, rng: np.random.Generator, n_train: int, n_val: int) -> Dict[str, np.ndarray]:
    total_rad = float(cfg["totalAngle"]) / 180.0 * np.pi
    start_rad = float(cfg["startAngle"]) / 180.0 * np.pi

    if bool(cfg.get("randomAngle", False)):
        train_angles = np.sort(rng.random(n_train) * total_rad) + start_rad
    else:
        train_angles = np.linspace(0.0, total_rad, n_train + 1, dtype=np.float64)[:-1] + start_rad

    if n_train > 1:
        interval = float(train_angles[1] - train_angles[0])
    else:
        interval = total_rad / max(1, n_train)

    val_midpoints = train_angles + interval / 2.0
    if n_val <= len(val_midpoints):
        val_angles = val_midpoints[:n_val]
    else:
        extra_needed = n_val - len(val_midpoints)
        extra_angles = val_midpoints[:extra_needed] + interval / 4.0
        val_angles = np.concatenate([val_midpoints, extra_angles], axis=0)

    return {
        "train": train_angles.astype(np.float64),
        "val": val_angles.astype(np.float64),
    }


def maybe_add_noise(projections: np.ndarray, cfg: Dict) -> np.ndarray:
    noise = float(cfg.get("noise", 0))
    if noise == 0 or not bool(cfg.get("normalize", True)):
        return projections

    try:
        from tigre.utilities import CTnoise  # type: ignore
    except Exception as exc:
        raise RuntimeError("Config requests noise, but TIGRE CTnoise cannot be imported.") from exc

    return CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, noise])).astype(np.float32)


def project_split(tigre, geo, volume_xyz: np.ndarray, angles: np.ndarray, cfg: Dict, split: str) -> np.ndarray:
    if angles.size == 0:
        h = int(cfg["nDetector"][0])
        w = int(cfg["nDetector"][1])
        return np.zeros((0, h, w), dtype=np.float32)

    print("Projecting {} views: {}".format(split, len(angles)))
    volume_zyx = np.transpose(volume_xyz, (2, 1, 0)).copy()
    projections = tigre.Ax(volume_zyx, geo, angles)[:, ::-1, :].astype(np.float32)
    projections = maybe_add_noise(projections, cfg)
    return projections.astype(np.float32)


def save_preview_image(path: Path, image: np.ndarray) -> None:
    try:
        import imageio.v2 as iio  # type: ignore
    except Exception:
        return

    arr = image.astype(np.float32)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), arr)


def save_visualization(vis_dir: Optional[Path], volume_xyz: np.ndarray, out_data: Dict) -> None:
    if vis_dir is None:
        return

    vis_dir.mkdir(parents=True, exist_ok=True)
    save_preview_image(vis_dir / "ct_mid_xy.png", volume_xyz[:, :, volume_xyz.shape[2] // 2])
    save_preview_image(vis_dir / "ct_mid_xz.png", volume_xyz[:, volume_xyz.shape[1] // 2, :])

    for split in ["train", "val"]:
        projections = np.asarray(out_data[split]["projections"])
        if projections.shape[0] == 0:
            continue
        step = max(1, projections.shape[0] // 5)
        for out_idx, proj_idx in enumerate(range(0, projections.shape[0], step)):
            if out_idx >= 5:
                break
            save_preview_image(vis_dir / "{}_proj_{:02d}.png".format(split, out_idx), projections[proj_idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CTSpine1K .nii/.nii.gz file to UniSpine-GS pickle format."
    )
    parser.add_argument("--input_nii_gz", required=True, help="Input CT volume in .nii or .nii.gz format.")
    parser.add_argument("--output_pickle", required=True, help="Output pickle path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Cone-beam geometry YAML.")
    parser.add_argument("--num_train", type=int, default=None, help="Override config numTrain.")
    parser.add_argument("--num_val", type=int, default=None, help="Override config numVal.")
    parser.add_argument("--gpu_id", default=None, help="CUDA_VISIBLE_DEVICES value used by TIGRE.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time_axis", type=int, default=None, help="Axis to index if the NIfTI file is 4D.")
    parser.add_argument("--time_index", type=int, default=0, help="Index to select if the NIfTI file is 4D.")
    parser.add_argument("--vis_dir", default=None, help="Optional directory for conversion sanity images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    input_path = Path(args.input_nii_gz).expanduser().resolve()
    output_path = Path(args.output_pickle).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    vis_dir = Path(args.vis_dir).expanduser().resolve() if args.vis_dir else None

    if not input_path.exists():
        raise FileNotFoundError("Input NIfTI does not exist: {}".format(input_path))
    if not config_path.exists():
        raise FileNotFoundError("Config YAML does not exist: {}".format(config_path))

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_train = int(args.num_train if args.num_train is not None else cfg["numTrain"])
    n_val = int(args.num_val if args.num_val is not None else cfg["numVal"])

    rng = np.random.default_rng(args.seed)
    tigre, Geometry = import_tigre()
    geo = build_tigre_geometry(cfg, Geometry)

    volume = load_nifti(input_path, args.time_axis, args.time_index)
    print("Loaded CT volume: shape={}, min={:.6f}, max={:.6f}".format(volume.shape, float(volume.min()), float(volume.max())))

    if bool(cfg.get("convert", True)):
        print("Convert from HU to attenuation")
        volume = convert_to_attenuation(
            volume,
            float(cfg.get("rescale_slope", 1.0)),
            float(cfg.get("rescale_intercept", 0.0)),
        )

    target_xyz = tuple(int(v) for v in cfg["nVoxel"])
    volume = resize_volume(volume, target_xyz)

    if bool(cfg.get("normalize", True)):
        print("Normalize CT volume to [0, 1]")
        volume = normalize_volume(volume)

    angles = build_angles(cfg, rng, n_train=n_train, n_val=n_val)

    out_data = dict(cfg)
    out_data["numTrain"] = n_train
    out_data["numVal"] = n_val
    out_data["image"] = volume.astype(np.float32)

    for split in ["train", "val"]:
        projections = project_split(tigre, geo, volume, angles[split], cfg, split)
        out_data[split] = {
            "angles": angles[split].astype(np.float64),
            "projections": projections.astype(np.float32),
        }
        print(
            "{} projections: shape={}, min={:.6f}, max={:.6f}".format(
                split,
                projections.shape,
                float(projections.min()) if projections.size else 0.0,
                float(projections.max()) if projections.size else 0.0,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(out_data, f, protocol=4)

    save_visualization(vis_dir, volume, out_data)
    print("Saved pickle: {}".format(output_path))
    if vis_dir is not None:
        print("Saved visualization: {}".format(vis_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Conversion failed: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
