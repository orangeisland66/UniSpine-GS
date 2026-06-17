#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

from tools.generate_data_yaml_configs import CONFIG_TEMPLATE


FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def extract_first_float(text: str):
    match = FLOAT_RE.search(text)
    if match:
        return float(match.group(0))
    return None


def parse_log_metrics(log_path: Path):
    if not log_path.exists():
        return {
            "SSIM": None,
            "PSNR": None,
            "total_points": None,
            "Testing Speed": None,
        }

    ssim = None
    psnr = None
    total_points = None
    testing_speed = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()

        if "Evaluating test: SSIM" in line and "PSNR" in line:
            # Support formats such as: SSIM = tensor(0.95, device='cuda:0'), PSNR = tensor(32.1, ...)
            ssim_part = line.split("SSIM =", 1)[1].split(", PSNR", 1)[0]
            psnr_part = line.split("PSNR =", 1)[1]
            cur_ssim = extract_first_float(ssim_part)
            cur_psnr = extract_first_float(psnr_part)
            if cur_ssim is not None:
                ssim = cur_ssim
            if cur_psnr is not None:
                psnr = cur_psnr

        elif "total_points:" in line:
            points_part = line.split("total_points:", 1)[1]
            cur_points = extract_first_float(points_part)
            if cur_points is not None:
                total_points = int(cur_points)

        elif "Testing Speed:" in line and "fps" in line:
            speed_part = line.split("Testing Speed:", 1)[1].split("fps", 1)[0]
            cur_speed = extract_first_float(speed_part)
            if cur_speed is not None:
                testing_speed = cur_speed

    return {
        "SSIM": ssim,
        "PSNR": psnr,
        "total_points": total_points,
        "Testing Speed": testing_speed,
    }


def maybe_make_config(config_path: Path, scene_name: str, pickle_path: Path):
    if config_path.exists():
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_content = CONFIG_TEMPLATE.format(scene=scene_name, source_path=pickle_path.as_posix())
    config_path.write_text(config_content, encoding="utf-8")
    print(f"[Info] Created config automatically: {config_path}")
    return config_path


def average_metric(items, key):
    values = [x[key] for x in items if x.get(key) is not None]
    if not values:
        return None
    return mean(values)


def write_metrics_csv(csv_path: Path, results, avg):
    headers = ["", "SSIM", "PSNR", "total_points", "Testing Speed", "Training Time"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for item in results:
            writer.writerow(
                [
                    item["scene"],
                    item.get("SSIM"),
                    item.get("PSNR"),
                    item.get("total_points"),
                    item.get("Testing Speed"),
                    item.get("Training Time"),
                ]
            )

        writer.writerow(
            [
                "average",
                avg.get("SSIM"),
                avg.get("PSNR"),
                avg.get("total_points"),
                avg.get("Testing Speed"),
                avg.get("Training Time"),
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="Batch train and evaluate all top-level pickle files in data.")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_dir = (project_root / args.data_dir).resolve()
    config_dir = (project_root / args.config_dir).resolve()
    output_root = (project_root / args.output_dir).resolve()

    pickle_files = sorted([p for p in data_dir.glob("*.pickle") if p.is_file()])

    if not pickle_files:
        print(f"[Error] No pickle files found: {data_dir}")
        return 1

    print("=" * 70)
    print(f"Found {len(pickle_files)} pickle files")
    for p in pickle_files:
        print(f"  - {p.name}")
    print("=" * 70)

    results = []
    failed = []

    for idx, pickle_path in enumerate(pickle_files, start=1):
        scene_name = pickle_path.stem
        config_path = maybe_make_config(config_dir / f"{scene_name}.yaml", scene_name, pickle_path)
        model_path = output_root / scene_name

        print("\n" + "-" * 70)
        print(f"[{idx}/{len(pickle_files)}] Starting: {scene_name}")
        print(f"config: {config_path}")
        print(f"model : {model_path}")
        print("-" * 70)

        if model_path.exists():
            print(f"[Info] Removing existing output directory: {model_path}")
            import shutil
            shutil.rmtree(model_path)

        cmd = [
            args.python,
            "train.py",
            "--config",
            str(config_path),
            "--eval",
            "--model_path",
            str(model_path),
            "--gpu_id",
            args.gpu_id,
        ]

        start_time = time.time()
        ret = subprocess.run(cmd, cwd=project_root)
        elapsed = time.time() - start_time

        if ret.returncode != 0:
            print(f"[Error] Training failed: {scene_name}")
            failed.append(scene_name)
            continue

        log_metrics = parse_log_metrics(model_path / "log.txt")

        item = {
            "scene": scene_name,
            "pickle": pickle_path.name,
            "config": str(config_path),
            "output": str(model_path),
            "SSIM": log_metrics["SSIM"],
            "PSNR": log_metrics["PSNR"],
            "total_points": log_metrics["total_points"],
            "Testing Speed": log_metrics["Testing Speed"],
            "Training Time": elapsed,
        }
        results.append(item)

        print(
            f"[Done] {scene_name} | "
            f"SSIM={item['SSIM']} | PSNR={item['PSNR']} | "
            f"total_points={item['total_points']} | "
            f"Testing Speed={item['Testing Speed']} fps | "
            f"Training Time={item['Training Time']:.2f} s"
        )

    avg = {
        "SSIM": average_metric(results, "SSIM"),
        "PSNR": average_metric(results, "PSNR"),
        "total_points": average_metric(results, "total_points"),
        "Testing Speed": average_metric(results, "Testing Speed"),
        "Training Time": average_metric(results, "Training Time"),
    }

    summary = {
        "total_pickle_files": len(pickle_files),
        "successful_runs": len(results),
        "failed_runs": len(failed),
        "failed_scenes": failed,
        "per_scene": results,
        "average": avg,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "batch_eval_summary.json"
    summary_txt = output_root / "batch_eval_summary.txt"
    summary_csv = output_root / "batch_eval_summary.csv"

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_metrics_csv(summary_csv, results, avg)

    lines = []
    lines.append("=" * 70)
    lines.append("Batch Train+Eval Summary")
    lines.append("=" * 70)
    lines.append(f"Total pickle files : {len(pickle_files)}")
    lines.append(f"Successful runs    : {len(results)}")
    lines.append(f"Failed runs        : {len(failed)}")
    if failed:
        lines.append(f"Failed scenes      : {', '.join(failed)}")
    lines.append("-" * 70)

    for item in results:
        lines.append(
            f"{item['scene']}: SSIM={item['SSIM']}, PSNR={item['PSNR']}, "
            f"total_points={item['total_points']}, Testing Speed={item['Testing Speed']} fps, "
            f"Training Time={item['Training Time']:.2f} s"
        )

    lines.append("-" * 70)
    lines.append("Average metrics across successful runs:")
    lines.append(f"SSIM          : {avg['SSIM']}")
    lines.append(f"PSNR          : {avg['PSNR']}")
    lines.append(f"total_points  : {avg['total_points']}")
    lines.append(f"Testing Speed : {avg['Testing Speed']} fps")
    lines.append(f"Training Time : {avg['Training Time']} s")
    lines.append("=" * 70)

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"[Info] Detailed results (JSON): {summary_json}")
    print(f"[Info] Text summary (TXT): {summary_txt}")
    print(f"[Info] Table results (CSV): {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
