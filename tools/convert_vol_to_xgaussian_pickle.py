#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "tools" / "configs" / "ctspine1k_spine.yaml"
VENDORED_TIGRE_PYTHON = PROJECT_ROOT / "third_party" / "TIGRE-2.3" / "Python"


def _setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("vol2xgaussian")


def _import_tigre_or_fail(logger: logging.Logger):
    try:
        import tigre  # type: ignore
        from tigre.utilities.geometry import Geometry  # type: ignore
        return tigre, Geometry
    except Exception:
        fallback_str = str(VENDORED_TIGRE_PYTHON)
        if VENDORED_TIGRE_PYTHON.exists() and fallback_str not in sys.path:
            sys.path.insert(0, fallback_str)
        try:
            import tigre  # type: ignore
            from tigre.utilities.geometry import Geometry  # type: ignore
            logger.info("Using TIGRE from fallback path: %s", fallback_str)
            return tigre, Geometry
        except Exception as e:
            logger.error("Cannot import TIGRE. Install the vendored TIGRE package from third_party/TIGRE-2.3/Python.")
            raise RuntimeError(f"TIGRE import failed: {e}")


@dataclass
class ReadCandidate:
    score: float
    shape_xyz: Tuple[int, int, int]
    dtype: np.dtype
    offset: int
    order: str
    volume_xyz: np.ndarray


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _score_volume(v: np.ndarray) -> float:
    # v: (X,Y,Z)
    p1, p99 = np.percentile(v, [1.0, 99.0])
    dyn = float(p99 - p1)

    zc = v.shape[2] // 2
    mid = v[:, :, zc]
    yc = v.shape[1] // 2
    mid_xz = v[:, yc, :]

    corr = _safe_corr(mid[:, :-1], mid[:, 1:]) + _safe_corr(mid[:-1, :], mid[1:, :]) + _safe_corr(mid_xz[:, :-1], mid_xz[:, 1:])

    gx = float(np.mean(np.abs(np.diff(mid, axis=1)))) + 1e-6
    gy = float(np.mean(np.abs(np.diff(mid, axis=0)))) + 1e-6
    stripe_penalty = abs(gx - gy) / (gx + gy)

    sparsity = float(np.mean(v <= np.percentile(v, 5)))
    return dyn + 0.12 * corr - 2.5 * stripe_penalty + 0.05 * sparsity


def parse_vol(
    input_vol: str,
    vol_shape_xyz: Optional[Tuple[int, int, int]],
    vol_dtype: Optional[str],
    vol_offset: Optional[int],
    vol_order: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, Dict[str, object]]:
    p = Path(input_vol)
    if not p.exists():
        raise FileNotFoundError(f"Input .vol not found: {input_vol}")

    file_size = p.stat().st_size
    with p.open("rb") as f:
        head = f.read(64)
    is_kretz = head.startswith(b"KRETZFILE")

    logger.info("Input vol: %s", str(p))
    logger.info("File size: %d bytes | KRETZ header: %s", file_size, is_kretz)

    # Strong prior for known fetal-spine file in this project.
    if file_size == 16310767 and vol_shape_xyz is None and vol_dtype is None and vol_offset is None:
        vol_shape_xyz = (141, 282, 205)
        vol_dtype = "uint16"
        vol_offset = 7027
        logger.info("Apply known preset for size 16310767: shape_xyz=%s dtype=%s offset=%d", vol_shape_xyz, vol_dtype, vol_offset)

    raw_bytes = p.read_bytes()

    if vol_shape_xyz is not None:
        dt = np.dtype(vol_dtype if vol_dtype is not None else "uint8")
        off = int(vol_offset) if vol_offset is not None else 0
        if off < 0 or off >= len(raw_bytes):
            raise ValueError(f"Invalid offset: {off}")
        buf = np.frombuffer(raw_bytes[off:], dtype=dt)
        need = int(np.prod(vol_shape_xyz))
        if buf.size < need:
            raise ValueError(f"Specified shape needs {need} values but only {buf.size} available after offset={off}")
        vol = buf[:need].reshape((vol_shape_xyz[2], vol_shape_xyz[1], vol_shape_xyz[0]), order=vol_order).transpose(2, 1, 0)
        meta = {"shape_xyz": vol_shape_xyz, "dtype": str(dt), "offset": off, "order": vol_order, "method": "manual_or_preset"}
        return vol.astype(np.float32), meta

    # Auto search
    dtype_candidates = [np.uint8, np.uint16]
    offset_candidates = {0, 1024, 4096, 7027, 8347}
    if is_kretz:
        offset_candidates.update(range(6800, 7600, 8))

    shape_candidates = [
        (141, 282, 205),
        (282, 141, 205),
        (410, 282, 141),
        (470, 141, 123),
        (141, 470, 123),
        (256, 256, 256),
        (512, 512, 174),
    ]

    best: Optional[ReadCandidate] = None
    for dt in dtype_candidates:
        item = np.dtype(dt).itemsize
        for off in sorted(offset_candidates):
            if off % item != 0 or off < 0 or off >= len(raw_bytes):
                continue
            arr = np.frombuffer(raw_bytes[off:], dtype=dt)
            for sx, sy, sz in shape_candidates:
                need = sx * sy * sz
                if arr.size < need:
                    continue
                try:
                    v = arr[:need].reshape((sz, sy, sx), order=vol_order).transpose(2, 1, 0).astype(np.float32)
                except Exception:
                    continue
                score = _score_volume(v)
                cand = ReadCandidate(score, (sx, sy, sz), np.dtype(dt), off, vol_order, v)
                if best is None or cand.score > best.score:
                    best = cand

    if best is None:
        raise RuntimeError("Failed to auto parse .vol. Please provide --vol_shape_xyz/--vol_dtype/--vol_offset.")

    logger.info(
        "Auto selected: shape_xyz=%s dtype=%s offset=%d order=%s score=%.4f",
        best.shape_xyz,
        best.dtype,
        best.offset,
        best.order,
        best.score,
    )
    meta = {
        "shape_xyz": best.shape_xyz,
        "dtype": str(best.dtype),
        "offset": int(best.offset),
        "order": best.order,
        "method": "auto",
        "score": float(best.score),
    }
    return best.volume_xyz.astype(np.float32), meta


class ConeGeometrySpecial:
    def __init__(self, cfg: Dict, Geometry):
        g = Geometry()
        g.DSD = cfg["DSD"] / 1000.0
        g.DSO = cfg["DSO"] / 1000.0
        g.nDetector = np.array(cfg["nDetector"])
        g.dDetector = np.array(cfg["dDetector"]) / 1000.0
        g.sDetector = g.nDetector * g.dDetector
        g.nVoxel = np.array(cfg["nVoxel"][::-1])
        g.dVoxel = np.array(cfg["dVoxel"][::-1]) / 1000.0
        g.sVoxel = g.nVoxel * g.dVoxel
        g.offOrigin = np.array(cfg["offOrigin"][::-1]) / 1000.0
        g.offDetector = np.array([cfg["offDetector"][1], cfg["offDetector"][0], 0]) / 1000.0
        g.accuracy = cfg["accuracy"]
        g.mode = cfg["mode"]
        g.filter = cfg["filter"]
        self.geo = g


def _normalize_volume(v: np.ndarray) -> np.ndarray:
    p_low, p_high = np.percentile(v, [0.5, 99.5])
    if p_high > p_low:
        v = np.clip(v, p_low, p_high)
        v = (v - p_low) / (p_high - p_low + 1e-8)
    else:
        vmin, vmax = float(v.min()), float(v.max())
        if vmax > vmin:
            v = (v - vmin) / (vmax - vmin)
    return v.astype(np.float32)


def _resize_xyz(v: np.ndarray, target_xyz: Tuple[int, int, int], logger: logging.Logger) -> np.ndarray:
    zoom = [target_xyz[i] / v.shape[i] for i in range(3)]
    if any(abs(z - 1.0) > 1e-6 for z in zoom):
        logger.info("Resize volume from %s to %s (zoom=%s)", v.shape, target_xyz, [round(z, 4) for z in zoom])
        v = ndi.zoom(v, zoom=zoom, order=3, prefilter=False)
    return v.astype(np.float32)


def _build_angle_splits(start_deg: float, total_deg: float, n_train: int, n_val: int, n_test: int) -> Dict[str, np.ndarray]:
    total = n_train + n_val + n_test
    if total <= 0:
        raise ValueError("n_train+n_val+n_test must be > 0")
    full = np.linspace(start_deg / 180.0 * np.pi, (start_deg + total_deg) / 180.0 * np.pi, total, endpoint=False, dtype=np.float64)
    train = full[:n_train]
    val = full[n_train:n_train + n_val]
    test = full[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def _save_img(path: Path, arr2d: np.ndarray) -> None:
    a = arr2d.astype(np.float32)
    if a.size == 0:
        out = np.zeros((256, 256), dtype=np.uint8)
    else:
        lo, hi = np.percentile(a, [1, 99])
        if hi > lo:
            a = np.clip(a, lo, hi)
            a = (a - lo) / (hi - lo + 1e-8)
        else:
            amin, amax = float(a.min()), float(a.max())
            if amax > amin:
                a = (a - amin) / (amax - amin)
        out = (a * 255.0).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), out)


def _visualize(vis_dir: Path, out_data: Dict, ref_vis_dir: Optional[Path], logger: logging.Logger) -> None:
    rng = np.random.default_rng(2026)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        if split not in out_data:
            continue
        proj = np.asarray(out_data[split]["projections"], dtype=np.float32)
        if proj.shape[0] == 0:
            continue
        picks = rng.choice(proj.shape[0], size=min(5, proj.shape[0]), replace=False)
        for i, idx in enumerate(picks):
            _save_img(vis_dir / f"{split}_proj_{i:02d}_idx{idx:03d}.png", proj[idx])

    all_proj = []
    for split in ["train", "val", "test"]:
        if split in out_data:
            all_proj.append(np.asarray(out_data[split]["projections"], dtype=np.float32).ravel())
    if all_proj:
        vals = np.concatenate(all_proj)
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=120)
        plt.title("Projection Intensity Histogram")
        plt.xlabel("intensity")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(vis_dir / "projection_histogram.png", dpi=150)
        plt.close()

    plt.figure(figsize=(6, 4))
    for split, c in [("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]:
        if split in out_data:
            ang = np.asarray(out_data[split]["angles"]) * 180.0 / np.pi
            plt.scatter(np.arange(len(ang)), ang, s=10, label=split, c=c)
    plt.legend()
    plt.title("Angle Distribution")
    plt.xlabel("index")
    plt.ylabel("degree")
    plt.tight_layout()
    plt.savefig(vis_dir / "angle_distribution.png", dpi=150)
    plt.close()

    if ref_vis_dir is not None and ref_vis_dir.exists():
        ref_pngs = sorted([p for p in ref_vis_dir.rglob("*.png")])
        new_pngs = sorted([p for p in vis_dir.glob("train_proj_*.png")])
        if ref_pngs and new_pngs:
            ref = iio.imread(str(ref_pngs[0]))
            new = iio.imread(str(new_pngs[0]))
            if ref.ndim == 2:
                ref = np.stack([ref, ref, ref], axis=-1)
            elif ref.ndim == 3 and ref.shape[-1] > 3:
                ref = ref[..., :3]
            if new.ndim == 2:
                new = np.stack([new, new, new], axis=-1)
            elif new.ndim == 3 and new.shape[-1] > 3:
                new = new[..., :3]
            h = min(ref.shape[0], new.shape[0])
            ref = ref[:h, ...]
            new = new[:h, ...]
            cmp_img = np.concatenate([ref, new], axis=1)
            iio.imwrite(str(vis_dir / "sanity_ref_vs_new_train_proj.png"), cmp_img)
            logger.info("Saved sanity compare image with reference: %s", str(vis_dir / "sanity_ref_vs_new_train_proj.png"))


def validate_pickle_schema(ref_pickle: str, new_pickle: str, logger: logging.Logger) -> bool:
    with open(ref_pickle, "rb") as f:
        ref = pickle.load(f)
    with open(new_pickle, "rb") as f:
        new = pickle.load(f)

    ok = True

    ref_keys = set(ref.keys())
    new_keys = set(new.keys())
    missing = sorted(ref_keys - new_keys)
    extra = sorted(new_keys - ref_keys)

    logger.info("Schema check | missing keys: %s", missing if missing else "None")
    logger.info("Schema check | extra keys: %s", extra if extra else "None")

    if missing:
        ok = False

    for key in ["image"]:
        if key in ref and key in new:
            ra = np.asarray(ref[key])
            na = np.asarray(new[key])
            if ra.ndim != na.ndim:
                logger.error("Key %s ndim mismatch: ref=%d new=%d", key, ra.ndim, na.ndim)
                ok = False
            if ra.dtype != na.dtype:
                logger.warning("Key %s dtype mismatch: ref=%s new=%s (建议保持 float32)", key, ra.dtype, na.dtype)

    for split in ["train", "val"]:
        if split in ref and split in new:
            for k in ["angles", "projections"]:
                if k not in new[split]:
                    logger.error("Missing %s[%s]", split, k)
                    ok = False
                    continue
                rr = np.asarray(ref[split][k])
                nn = np.asarray(new[split][k])
                if rr.ndim != nn.ndim:
                    logger.error("%s[%s] ndim mismatch: ref=%d new=%d", split, k, rr.ndim, nn.ndim)
                    ok = False
                if k == "projections" and rr.shape[1:] != nn.shape[1:]:
                    logger.error("%s[%s] spatial shape mismatch: ref=%s new=%s", split, k, rr.shape[1:], nn.shape[1:])
                    ok = False

    for gk in ["DSD", "DSO", "nVoxel", "dVoxel", "nDetector", "dDetector", "offOrigin", "offDetector", "accuracy", "mode", "filter"]:
        if gk not in new:
            logger.error("Missing geometry key: %s", gk)
            ok = False

    if ok:
        logger.info("Schema validation PASSED")
    else:
        logger.error("Schema validation FAILED")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .vol to UniSpine/X-Gaussian compatible pickle")
    parser.add_argument("--input_vol", required=True)
    parser.add_argument("--output_pickle", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--vis_dir", required=True)
    parser.add_argument("--ref_pickle", default=None, help="Optional reference pickle used for schema comparison.")
    parser.add_argument("--ref_vis_dir", default=None, help="Optional reference visualization directory for sanity comparison.")
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--vol_shape_xyz", default=None, help="e.g. 141,282,205")
    parser.add_argument("--vol_dtype", choices=["uint8", "uint16"], default=None)
    parser.add_argument("--vol_offset", type=int, default=None)
    parser.add_argument("--vol_order", choices=["C", "F"], default="C")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip_loader_check", action="store_true", help="Skip UniSpine-GS dataloader smoke test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = _setup_logger(args.verbose)

    np.random.seed(args.seed)

    input_vol_path = Path(args.input_vol).expanduser().resolve()
    output_pickle_path = Path(args.output_pickle).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    vis_dir_path = Path(args.vis_dir).expanduser().resolve()
    ref_pickle_path = Path(args.ref_pickle).expanduser().resolve() if args.ref_pickle else None
    ref_vis_dir_path = Path(args.ref_vis_dir).expanduser().resolve() if args.ref_vis_dir else None

    if args.vol_shape_xyz is not None:
        parts = [int(x.strip()) for x in args.vol_shape_xyz.split(",")]
        if len(parts) != 3:
            logger.error("--vol_shape_xyz must be X,Y,Z")
            return 2
        vol_shape_xyz = tuple(parts)
    else:
        vol_shape_xyz = None

    if not input_vol_path.exists():
        raise FileNotFoundError(f"Input .vol not found: {input_vol_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")
    if ref_pickle_path is not None and not ref_pickle_path.exists():
        raise FileNotFoundError(f"Reference pickle not found: {ref_pickle_path}")

    tigre, Geometry = _import_tigre_or_fail(logger)

    cfg = yaml.safe_load(config_path.read_text())

    vol_xyz_raw, parse_meta = parse_vol(
        str(input_vol_path),
        vol_shape_xyz=vol_shape_xyz,
        vol_dtype=args.vol_dtype,
        vol_offset=args.vol_offset,
        vol_order=args.vol_order,
        logger=logger,
    )

    logger.info("Parsed volume meta: %s", parse_meta)
    logger.info(
        "Raw volume stats | shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
        vol_xyz_raw.shape,
        float(vol_xyz_raw.min()),
        float(vol_xyz_raw.max()),
        float(vol_xyz_raw.mean()),
        float(vol_xyz_raw.std()),
    )

    vol_xyz = _normalize_volume(vol_xyz_raw)
    target_xyz = tuple(int(v) for v in cfg["nVoxel"])
    vol_xyz = _resize_xyz(vol_xyz, target_xyz, logger)

    logger.info(
        "Final volume stats | shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
        vol_xyz.shape,
        float(vol_xyz.min()),
        float(vol_xyz.max()),
        float(vol_xyz.mean()),
        float(vol_xyz.std()),
    )

    geo = ConeGeometrySpecial(cfg, Geometry).geo

    splits = _build_angle_splits(
        start_deg=float(cfg["startAngle"]),
        total_deg=float(cfg["totalAngle"]),
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
    )

    vol_zyx = np.transpose(vol_xyz, (2, 1, 0)).copy()

    out_data = dict(cfg)
    out_data["image"] = vol_xyz.astype(np.float32)
    out_data["numTrain"] = int(args.n_train)
    out_data["numVal"] = int(args.n_val)

    for split in ["train", "val", "test"]:
        ang = splits[split]
        if ang.size == 0:
            out_data[split] = {"angles": ang.astype(np.float64), "projections": np.zeros((0, int(cfg["nDetector"][1]), int(cfg["nDetector"][0])), dtype=np.float32)}
            continue
        logger.info("Projecting %s views: %d", split, ang.size)
        proj = tigre.Ax(vol_zyx, geo, ang)[:, ::-1, :].astype(np.float32)
        proj = np.clip(proj, a_min=0.0, a_max=None)
        out_data[split] = {"angles": ang.astype(np.float64), "projections": proj.astype(np.float32)}
        logger.info(
            "%s projections stats | shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
            split,
            proj.shape,
            float(proj.min()),
            float(proj.max()),
            float(proj.mean()),
            float(proj.std()),
        )

    # Keep compatibility: train/val are required by existing loaders.
    if "test" in out_data and out_data["test"]["angles"].size > 0:
        logger.info("Non-overlap check train/val/test: %s",
                    len(set(np.round(out_data['train']['angles'], 10)).intersection(set(np.round(out_data['val']['angles'], 10)))) == 0 and
                    len(set(np.round(out_data['train']['angles'], 10)).intersection(set(np.round(out_data['test']['angles'], 10)))) == 0 and
                    len(set(np.round(out_data['val']['angles'], 10)).intersection(set(np.round(out_data['test']['angles'], 10)))) == 0)

    output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pickle_path.open("wb") as f:
        pickle.dump(out_data, f, protocol=4)
    logger.info("Saved pickle: %s", str(output_pickle_path))

    _visualize(vis_dir_path, out_data, ref_vis_dir_path, logger)

    if ref_pickle_path is not None:
        ok = validate_pickle_schema(str(ref_pickle_path), str(output_pickle_path), logger)
    else:
        logger.info("Schema validation skipped: --ref_pickle was not provided.")
        ok = True

    # Dataloader compatibility smoke test
    if not args.skip_loader_check:
        try:
            project_root = str(PROJECT_ROOT)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from scene.dataset_readers import Xray_readCamerasFromTransforms
            _ = Xray_readCamerasFromTransforms(str(output_pickle_path), type="train")
            _ = Xray_readCamerasFromTransforms(str(output_pickle_path), type="val")
            logger.info("Dataloader compatibility smoke test: PASSED")
        except Exception as e:
            logger.warning("Dataloader compatibility smoke test: SKIPPED/FAILED in current env | %s", e)
            logger.warning("Suggestion: run this check inside the unispine_gs env where project dependencies are installed.")

    return 0 if ok else 4


if __name__ == "__main__":
    try:
        code = main()
        raise SystemExit(code)
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.error("Fatal error: %s", exc)
        raise SystemExit(1)
