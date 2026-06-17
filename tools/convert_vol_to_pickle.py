#!/usr/bin/env python3
"""
convert_vol_ax.py

Ultrasound .vol -> pickle (same schema style as generateData_all.py)

Key guarantees:
1) The angles stored in pickle are EXACTLY the angles used to generate projections.
2) Volume-axis alignment (reorder/flip) is applied BEFORE Ax, controlled by config.projection.*.
3) Preview orientation tweaks (post_rot90/flip) are applied ONLY to saved PNGs, NOT to training projections.

Usage:
  python convert_vol_ax.py INPUT.vol OUTPUT.pickle --config path/to/config_ultrasound.yml
"""

import os
import math
import struct
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import yaml
import scipy.ndimage as ndi
import imageio.v2 as iio

import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import CTnoise


# --------------------------------------------------------------------------------
# 1) Geometry (copy from generateData_all.py)
# --------------------------------------------------------------------------------
class ConeGeometry_special(Geometry):
    """
    Cone beam CT geometry (TIGRE).
    """
    def __init__(self, data: Dict[str, Any]):
        Geometry.__init__(self)

        self.DSD = data["DSD"] / 1000.0
        self.DSO = data["DSO"] / 1000.0

        self.nDetector = np.array(data["nDetector"], dtype=np.int32)
        self.dDetector = np.array(data["dDetector"], dtype=np.float32) / 1000.0
        self.sDetector = self.nDetector * self.dDetector

        # TIGRE expects (Z,Y,X) ordering internally
        self.nVoxel = np.array(data["nVoxel"][::-1], dtype=np.int32)
        self.dVoxel = np.array(data["dVoxel"][::-1], dtype=np.float32) / 1000.0
        self.sVoxel = self.nVoxel * self.dVoxel

        self.offOrigin = np.array(data["offOrigin"][::-1], dtype=np.float32) / 1000.0
        self.offDetector = np.array([data["offDetector"][1], data["offDetector"][0], 0], dtype=np.float32) / 1000.0

        self.accuracy = float(data["accuracy"])
        self.mode = data["mode"]
        self.filter = data["filter"]


# --------------------------------------------------------------------------------
# 2) .vol parsing (KRETZ tag stream)
# --------------------------------------------------------------------------------
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
    # v shape: (X,Y,Z)
    if v.size == 0:
        return 0.0
    p1, p99 = np.percentile(v, [1.0, 99.0])
    dyn = float(p99 - p1)

    zc = v.shape[2] // 2
    yc = v.shape[1] // 2
    mid_xy = v[:, :, zc]
    mid_xz = v[:, yc, :]

    corr = (
        _safe_corr(mid_xy[:, :-1], mid_xy[:, 1:])
        + _safe_corr(mid_xy[:-1, :], mid_xy[1:, :])
        + _safe_corr(mid_xz[:, :-1], mid_xz[:, 1:])
    )

    gx = float(np.mean(np.abs(np.diff(mid_xy, axis=1)))) + 1e-6
    gy = float(np.mean(np.abs(np.diff(mid_xy, axis=0)))) + 1e-6
    stripe_penalty = abs(gx - gy) / (gx + gy)

    sparsity = float(np.mean(v <= np.percentile(v, 5)))
    return dyn + 0.12 * corr - 2.5 * stripe_penalty + 0.05 * sparsity


def _parse_kretz_tag_stream(raw: bytes) -> Tuple[Optional[np.ndarray], dict]:
    meta = {
        "method": "kretz_tag_stream",
        "header": raw[:16].decode("ascii", errors="ignore"),
    }
    if not raw.startswith(b"KRETZFILE 1.0"):
        return None, meta

    o = 16
    dims = {"i": 0, "j": 0, "k": 0}
    voxel_offset = None
    voxel_size = None

    while o + 8 <= len(raw):
        g, e, s = struct.unpack_from("<HHI", raw, o)
        o += 8
        if o + s > len(raw):
            break

        if (g, e) == (0xC000, 0x0001) and s >= 2:
            dims["i"] = int(struct.unpack_from("<H", raw, o)[0])
        elif (g, e) == (0xC000, 0x0002) and s >= 2:
            dims["j"] = int(struct.unpack_from("<H", raw, o)[0])
        elif (g, e) == (0xC000, 0x0003) and s >= 2:
            dims["k"] = int(struct.unpack_from("<H", raw, o)[0])
        elif (g, e) == (0xD000, 0x0001):
            voxel_offset = o
            voxel_size = int(s)

        o += s

    i, j, k = dims["i"], dims["j"], dims["k"]
    need = i * j * k

    if voxel_offset is None or voxel_size is None or need <= 0:
        meta["kretz_parse_ok"] = False
        return None, meta

    if voxel_size == need:
        dt = np.uint8
    elif voxel_size == need * 2:
        dt = np.uint16
    elif voxel_size > need and (voxel_size // need) in (1, 2):
        dt = np.uint16 if (voxel_size // need) == 2 else np.uint8
    else:
        meta["kretz_parse_ok"] = False
        return None, meta

    item_bpp = np.dtype(dt).itemsize
    usable = (voxel_size // item_bpp) * item_bpp
    if voxel_offset + usable > len(raw):
        usable = len(raw) - voxel_offset
        usable = (usable // item_bpp) * item_bpp

    arr = np.frombuffer(raw[voxel_offset: voxel_offset + usable], dtype=dt)
    if arr.size < need:
        meta["kretz_parse_ok"] = False
        return None, meta

    # reshape((k,j,i)).transpose(2,1,0) => (i,j,k) == (X,Y,Z)
    vol_xyz = arr[:need].reshape((k, j, i), order="C").transpose(2, 1, 0).astype(np.float32)

    meta.update({
        "kretz_parse_ok": True,
        "dtype": str(np.dtype(dt)),
        "offset": int(voxel_offset),
        "order": "C",
        "shape_xyz": [i, j, k],
    })
    return vol_xyz, meta


def parse_vol(
    vol_path: Path,
    shape_xyz: Optional[Tuple[int, int, int]] = None,
    dtype: Optional[str] = None,
    offset: Optional[int] = None,
    order: str = "C",
) -> Tuple[np.ndarray, dict]:
    raw = vol_path.read_bytes()
    file_size = len(raw)
    is_kretz = raw[:16].startswith(b"KRETZFILE")

    # KRETZ auto
    if is_kretz and shape_xyz is None and dtype is None and offset is None:
        vol_kretz, meta_kretz = _parse_kretz_tag_stream(raw)
        if vol_kretz is not None:
            return vol_kretz, meta_kretz

    # manual
    if shape_xyz is not None:
        dt = np.dtype(dtype if dtype is not None else "uint8")
        off = int(offset) if offset is not None else 0
        arr = np.frombuffer(raw[off:], dtype=dt)
        need = int(np.prod(shape_xyz))
        if arr.size < need:
            raise ValueError(f"Need {need}, got {arr.size}")
        vol = arr[:need].reshape((shape_xyz[2], shape_xyz[1], shape_xyz[0]), order=order).transpose(2, 1, 0)
        return vol.astype(np.float32), {"method": "manual", "shape_xyz": list(shape_xyz), "dtype": str(dt), "offset": off}

    # heuristic fallback (kept minimal)
    dtype_candidates = [np.uint8, np.uint16]
    offset_candidates = {0, 1024, 4096, 7027, 8347}
    if is_kretz:
        offset_candidates.update(range(6800, 7600, 8))

    shape_candidates = [
        (302, 124, 211),
        (141, 282, 205),
        (256, 256, 256),
        (512, 512, 174),
    ]

    best: Optional[ReadCandidate] = None
    for dt in dtype_candidates:
        itemsize = np.dtype(dt).itemsize
        for off in sorted(offset_candidates):
            if off < 0 or off >= file_size or (off % itemsize != 0):
                continue
            arr = np.frombuffer(raw[off:], dtype=dt)
            for sx, sy, sz in shape_candidates:
                need = sx * sy * sz
                if arr.size < need:
                    continue
                try:
                    vol = arr[:need].reshape((sz, sy, sx), order=order).transpose(2, 1, 0).astype(np.float32)
                except Exception:
                    continue
                score = _score_volume(vol)
                cand = ReadCandidate(score, (sx, sy, sz), np.dtype(dt), off, order, vol)
                if best is None or cand.score > best.score:
                    best = cand

    if best is None:
        raise RuntimeError("Auto-parse failed. Provide --shape/--dtype/--offset.")

    return best.volume_xyz.astype(np.float32), {
        "method": "auto",
        "shape_xyz": list(best.shape_xyz),
        "dtype": str(best.dtype),
        "offset": int(best.offset),
        "order": best.order,
        "score": float(best.score),
    }


# --------------------------------------------------------------------------------
# 3) Preprocess
# --------------------------------------------------------------------------------
def normalize_volume_percentile(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32, copy=False)
    p1, p99 = np.percentile(v, [1.0, 99.0])
    if p99 > p1:
        v = np.clip(v, p1, p99)
        v = (v - p1) / (p99 - p1)
    else:
        mn, mx = float(v.min()), float(v.max())
        if mx > mn:
            v = (v - mn) / (mx - mn)
        else:
            v = np.zeros_like(v, dtype=np.float32)
    return np.clip(v, 0.0, 1.0)


def resize_xyz(vol_xyz: np.ndarray, target_xyz: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    sx, sy, sz = vol_xyz.shape
    tx, ty, tz = target_xyz
    if (sx, sy, sz) == (tx, ty, tz):
        return vol_xyz.astype(np.float32, copy=False)
    zoom = (tx / sx, ty / sy, tz / sz)
    return ndi.zoom(vol_xyz.astype(np.float32, copy=False), zoom, order=order, prefilter=False).astype(np.float32, copy=False)


# --------------------------------------------------------------------------------
# 4) Angles
# --------------------------------------------------------------------------------
def make_train_val_angles(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    num_train = int(data["numTrain"])
    num_val = int(data["numVal"])
    total = float(data["totalAngle"])
    start = float(data["startAngle"])
    random_angle = bool(data.get("randomAngle", False))

    if not random_angle:
        train = np.linspace(0, total / 180 * np.pi, num_train + 1)[:-1] + start / 180 * np.pi
    else:
        train = np.sort(np.random.rand(num_train) * total / 180 * np.pi) + start / 180 * np.pi

    if len(train) > 1:
        interval = train[1] - train[0]
    else:
        interval = (total / 180 * np.pi) / max(num_train, 1)

    val_mid = train + (interval / 2)

    if num_val <= len(val_mid):
        val = val_mid[:num_val]
    else:
        extra_needed = num_val - len(val_mid)
        extra_angles = val_mid[:extra_needed] + (interval / 4)
        val = np.concatenate([val_mid, extra_angles])

    return train.astype(np.float32), val.astype(np.float32)


def apply_angle_alignment(angles: np.ndarray, angle_offset_deg: float = 0.0, angle_sign: float = 1.0) -> np.ndarray:
    """
    angles_used = angle_sign * angles + deg2rad(angle_offset_deg)
    IMPORTANT: returned angles must be exactly what we store into pickle.
    """
    return (angles.astype(np.float32) * float(angle_sign) + np.float32(np.deg2rad(float(angle_offset_deg)))).astype(np.float32)


# --------------------------------------------------------------------------------
# 5) Volume axis alignment (critical)
# --------------------------------------------------------------------------------
def apply_axis_order_and_flip(vol_xyz: np.ndarray, volume_axis_order=(0, 1, 2), flip_xyz=(False, False, False)) -> np.ndarray:
    """
    vol_xyz: (X,Y,Z)
    volume_axis_order: reorder axes, output is interpreted as new (X,Y,Z)
    flip_xyz: flips after reorder on the reordered axes.
    """
    v = np.transpose(vol_xyz, tuple(volume_axis_order)).astype(np.float32, copy=False)
    fx, fy, fz = flip_xyz
    if fx:
        v = v[::-1, :, :]
    if fy:
        v = v[:, ::-1, :]
    if fz:
        v = v[:, :, ::-1]
    return v

def fix_dvoxel_keep_physical_size(
    config: Dict[str, Any],
    data: Dict[str, Any],
    volume_axis_order: Tuple[int, int, int],
) -> None:
    """
    Fix data['dVoxel'] (mm) after resize + axis reorder so that the *physical size* sVoxel stays consistent.

    Why needed:
      - You resized volume to some nVoxel (X,Y,Z)
      - Then you reorder axes via volume_axis_order for projection
      - But data['dVoxel'] usually stays as the original config order -> causes sVoxel explode/shrink
      - If sVoxel is larger than detector FOV, Ax only shows a local part (your symptom).

    We do:
      cfg_s_mm = config_nVoxel * config_dVoxel   (in config's X,Y,Z)
      s_mm_aligned = cfg_s_mm[volume_axis_order] (now in aligned X,Y,Z)
      new_dVoxel = s_mm_aligned / aligned_nVoxel
      write back data['dVoxel'] = new_dVoxel (still in aligned X,Y,Z order, mm)
    """
    # config physical size in mm (config's X,Y,Z)
    cfg_n = np.array(config["nVoxel"], dtype=np.float32)     # (X,Y,Z)
    cfg_d = np.array(config["dVoxel"], dtype=np.float32)     # (X,Y,Z) in mm
    cfg_s_mm = cfg_n * cfg_d                                 # (X,Y,Z) mm

    # apply same axis reorder to physical size (aligned X,Y,Z)
    ao = np.array(volume_axis_order, dtype=np.int64)
    if ao.shape[0] != 3:
        raise ValueError(f"volume_axis_order must have 3 elements, got {volume_axis_order}")
    if sorted(ao.tolist()) != [0, 1, 2]:
        raise ValueError(f"volume_axis_order must be a permutation of (0,1,2), got {volume_axis_order}")

    s_mm_aligned = cfg_s_mm[ao]  # aligned (X,Y,Z) mm

    # current aligned voxel counts
    new_n = np.array(data["nVoxel"], dtype=np.float32)  # aligned (X,Y,Z)

    # avoid division by zero
    if np.any(new_n <= 0):
        raise ValueError(f"Invalid aligned nVoxel: {data['nVoxel']}")

    new_d = (s_mm_aligned / new_n).astype(np.float32)  # aligned (X,Y,Z) mm
    data["dVoxel"] = [float(new_d[0]), float(new_d[1]), float(new_d[2])]

    print(
        "[FixDVoxel] keep physical size: "
        f"cfg_sVoxel(mm)={cfg_s_mm.tolist()} -> aligned_sVoxel(mm)={s_mm_aligned.tolist()} ; "
        f"set dVoxel(mm)={data['dVoxel']} for aligned nVoxel={data['nVoxel']}"
    )
# --------------------------------------------------------------------------------
# 6) Projection: TIGRE Ax (line integral)
# --------------------------------------------------------------------------------
def project_with_tigre_ax(mu_xyz_aligned: np.ndarray, geo: Geometry, angles: np.ndarray) -> np.ndarray:
    """
    mu_xyz_aligned is treated as (X,Y,Z) in OUR chosen aligned coordinate.
    TIGRE expects input volume in (Z,Y,X), so we transpose here.
    """
    vol_zyx = np.transpose(mu_xyz_aligned, (2, 1, 0)).copy()
    proj = tigre.Ax(vol_zyx, geo, angles)  # (N, nDetV, nDetU)
    proj = proj[:, ::-1, :]                # match generateData_all orientation
    return proj.astype(np.float32, copy=False)


# --------------------------------------------------------------------------------
# 7) Preview saving (preview-only post transforms)
# --------------------------------------------------------------------------------
def postprocess_for_preview_only(img2d: np.ndarray, rot90_k: int = 0, flip_ud: bool = False, flip_lr: bool = False) -> np.ndarray:
    out = img2d
    k = int(rot90_k) % 4
    if k != 0:
        out = np.rot90(out, k=k)
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return out


def to_uint8_for_vis(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, [1.0, 99.0])
    if hi > lo:
        a = np.clip(a, lo, hi)
        a = (a - lo) / (hi - lo + 1e-8)
    else:
        mn, mx = float(a.min()), float(a.max())
        if mx > mn:
            a = (a - mn) / (mx - mn)
        else:
            a = np.zeros_like(a, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def save_previews(output_pickle: Path, projs: np.ndarray, angles: np.ndarray, prefix: str, proj_cfg: Dict[str, Any], step: int = 5) -> None:
    out_dir = output_pickle.parent
    name = output_pickle.stem
    preview_dir = out_dir / (name + "_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)

    rot90_k = int(proj_cfg.get("post_rot90_k", 0))
    flip_ud = bool(proj_cfg.get("post_flip_ud", False))
    flip_lr = bool(proj_cfg.get("post_flip_lr", False))

    idxs = list(range(0, projs.shape[0], step))
    for i in idxs:
        deg = float(np.degrees(angles[i]))
        fname = f"{prefix}_{i:03d}_{deg:.1f}deg_proj.png"
        p = postprocess_for_preview_only(projs[i], rot90_k=rot90_k, flip_ud=flip_ud, flip_lr=flip_lr)
        iio.imwrite(preview_dir / fname, to_uint8_for_vis(p))

    print(f"[Previews] saved to: {preview_dir}")


# --------------------------------------------------------------------------------
# 8) Main
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert Ultrasound .vol -> pickle (generateData_all-aligned)")
    parser.add_argument("positional_input_path", nargs="?", help="Input .vol path, SAX-NeRF style")
    parser.add_argument("positional_output_pickle", nargs="?", help="Output .pickle path, SAX-NeRF style")
    parser.add_argument("--input_vol", type=str, default=None, help="Input .vol path")
    parser.add_argument("--output_pickle", type=str, default=None, help="Output .pickle path")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "fespine3d_spine.yaml"),
        help="Ultrasound projection config YAML",
    )
    parser.add_argument("--vis_dir", type=str, default=None, help="Accepted for transform_vol.sh compatibility")
    parser.add_argument("--mu_scale", type=float, default=None,
                        help="Override config.projection.us_mu_scale")
    parser.add_argument("--no_resize", action="store_true",
                        help="Do NOT resize to config nVoxel.")
    args = parser.parse_args()

    input_arg = args.input_vol or args.positional_input_path
    output_arg = args.output_pickle or args.positional_output_pickle
    if input_arg is None or output_arg is None:
        parser.error("input and output are required; use positional paths or --input_vol/--output_pickle")

    input_path = Path(input_arg).resolve()
    output_pickle = Path(output_arg).resolve()
    config_path = Path(args.config).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- parse vol ---
    print(f"[Convert] {input_path} -> {output_pickle}")
    print(f"Parsing {input_path} ...")
    vol_xyz, meta = parse_vol(input_path)
    print(f"Parsed shape (X,Y,Z): {vol_xyz.shape} | dtype={vol_xyz.dtype} | meta={meta.get('method')} {meta.get('shape_xyz')}")
    print(f"Raw range: min={float(vol_xyz.min())} max={float(vol_xyz.max())} p1/p99={np.percentile(vol_xyz, [1,99])}")

    # --- prepare data dict ---
    data = dict(config)
    data["vol_meta"] = meta

    # --- resize ---
    target_nVoxel_xyz = tuple(int(x) for x in data["nVoxel"])  # (X,Y,Z)
    if args.no_resize:
        mu_xyz = vol_xyz.astype(np.float32, copy=False)
        data["nVoxel"] = [int(mu_xyz.shape[0]), int(mu_xyz.shape[1]), int(mu_xyz.shape[2])]
        print(f"[NoResize] Using parsed nVoxel={data['nVoxel']}")
    else:
        print(f"[Resize] {vol_xyz.shape} -> target nVoxel {target_nVoxel_xyz}")
        mu_xyz = resize_xyz(vol_xyz, target_nVoxel_xyz, order=1)
        print(f"[Resize] done. new shape={mu_xyz.shape}")

    # --- normalize + mu scale ---
    mu_xyz = normalize_volume_percentile(mu_xyz)

    proj_cfg = data.get("projection", {}) if isinstance(data, dict) else {}
    cfg_mu_scale = float(proj_cfg.get("us_mu_scale", 1.0))
    mu_scale = float(args.mu_scale) if args.mu_scale is not None else cfg_mu_scale
    mu_xyz = (mu_xyz * mu_scale).astype(np.float32)

    # --- apply volume axis alignment used for Ax (THIS affects geometry) ---
    volume_axis_order = tuple(proj_cfg.get("volume_axis_order", [2, 1, 0]))
    flip_xyz = tuple(proj_cfg.get("flip_xyz", [False, False, False]))
    mu_xyz_aligned = apply_axis_order_and_flip(mu_xyz, volume_axis_order=volume_axis_order, flip_xyz=flip_xyz)

    # store image (what training uses) as aligned mu field, consistent with projections
    data["image"] = mu_xyz_aligned.copy()

    # --- angles: generate -> align -> store (MUST match projections) ---
    train_angles, val_angles = make_train_val_angles(data)

    angle_offset_deg = float(proj_cfg.get("angle_offset_deg", 0.0))
    angle_sign = float(proj_cfg.get("angle_sign", 1.0))

    train_angles = apply_angle_alignment(train_angles, angle_offset_deg=angle_offset_deg, angle_sign=angle_sign)
    val_angles   = apply_angle_alignment(val_angles,   angle_offset_deg=angle_offset_deg, angle_sign=angle_sign)

    data["train"] = {"angles": train_angles}
    data["val"]   = {"angles": val_angles}

    # --- geometry ---
    # NOTE: geo uses data["nVoxel"] to set sVoxel; we kept original config nVoxel.
    # But we changed image to aligned axes. If axis_order changes shape, update nVoxel to match aligned volume.
    # This is CRITICAL for consistency.
    # --- geometry ---
    # After axis reorder, update nVoxel to match aligned volume.
    aligned_shape = mu_xyz_aligned.shape  # (X,Y,Z) after reorder/flip
    data["nVoxel"] = [int(aligned_shape[0]), int(aligned_shape[1]), int(aligned_shape[2])]

    # IMPORTANT FIX:
    # Update dVoxel so that physical size sVoxel stays consistent with config,
    # but expressed in the aligned axis order.
    fix_dvoxel_keep_physical_size(config=config, data=data, volume_axis_order=volume_axis_order)

    geo = ConeGeometry_special(data)
    print("[Geo] nVoxel", data["nVoxel"], "dVoxel(mm)", data["dVoxel"])
    print("[Geo] sVoxel(m)", geo.sVoxel, "sDetector(m)", geo.sDetector)
    print("[Geo] offOrigin(mm)", data["offOrigin"], "offDetector(mm)", data["offDetector"])
    # --- projections (Ax) ---
    print(f"[Project] Ax | mu_scale={mu_scale} | axis_order={volume_axis_order} flip_xyz={flip_xyz} | angle_offset_deg={angle_offset_deg} angle_sign={angle_sign}")
    projs_train = project_with_tigre_ax(mu_xyz_aligned, geo, train_angles)
    projs_val   = project_with_tigre_ax(mu_xyz_aligned, geo, val_angles)

    # optional noise (same behavior style)
    if float(data.get("noise", 0)) != 0 and bool(data.get("normalize", True)):
        print("[Noise] Add CTnoise to projections (Poisson+Gaussian) ...")
        sigma = float(data["noise"])
        projs_train = CTnoise.add(projs_train, Poisson=1e5, Gaussian=np.array([0, sigma])).astype(np.float32)
        projs_val   = CTnoise.add(projs_val,   Poisson=1e5, Gaussian=np.array([0, sigma])).astype(np.float32)

    data["train"]["projections"] = projs_train
    data["val"]["projections"]   = projs_val

    # --- previews (preview-only post transforms) ---
    print("[Preview] saving ...")
    save_previews(output_pickle, projs_train, train_angles, prefix="train", proj_cfg=proj_cfg, step=5)
    save_previews(output_pickle, projs_val,   val_angles,   prefix="val",   proj_cfg=proj_cfg, step=5)

    # --- save pickle ---
    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, protocol=4)

    print(f"[Done] wrote: {output_pickle}")
    print(f"train projections: {projs_train.shape}, val projections: {projs_val.shape}")
    print(f"stored nVoxel (X,Y,Z): {data['nVoxel']}")
    print(f"stored nDetector: {data['nDetector']}")


if __name__ == "__main__":
    main()
