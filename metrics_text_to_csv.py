#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import mean

LINE_RE = re.compile(
    r"^\s*(?P<scene>[^:]+):\s*"
    r"SSIM\s*=\s*(?P<ssim>[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?),\s*"
    r"PSNR\s*=\s*(?P<psnr>[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?),\s*"
    r"total_points\s*=\s*(?P<points>\d+),\s*"
    r"Testing\s+Speed\s*=\s*(?P<speed>[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*fps,\s*"
    r"Training\s+Time\s*=\s*(?P<time>[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*s\s*$"
)


def parse_metrics_text(text: str):
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip().rstrip("，").strip()
        if not line:
            continue

        m = LINE_RE.match(line)
        if not m:
            continue

        rows.append(
            {
                "scene": m.group("scene"),
                "SSIM": float(m.group("ssim")),
                "PSNR": float(m.group("psnr")),
                "total_points": int(m.group("points")),
                "Testing Speed": float(m.group("speed")),
                "Training Time": float(m.group("time")),
            }
        )

    return rows


def avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


def write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    average_row = {
        "scene": "average",
        "SSIM": avg(rows, "SSIM"),
        "PSNR": avg(rows, "PSNR"),
        "total_points": avg(rows, "total_points"),
        "Testing Speed": avg(rows, "Testing Speed"),
        "Training Time": avg(rows, "Training Time"),
    }

    headers = ["", "SSIM", "PSNR", "total_points", "Testing Speed", "Training Time"]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(
                [
                    r["scene"],
                    r["SSIM"],
                    r["PSNR"],
                    r["total_points"],
                    r["Testing Speed"],
                    r["Training Time"],
                ]
            )

        writer.writerow(
            [
                average_row["scene"],
                average_row["SSIM"],
                average_row["PSNR"],
                average_row["total_points"],
                average_row["Testing Speed"],
                average_row["Training Time"],
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="将批量评估文本转换为 CSV（含 average 行）")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="输入文本文件路径；不传则从标准输入读取",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/metrics_from_text.csv",
        help="输出 CSV 路径",
    )
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    else:
        if sys.stdin.isatty():
            print("请粘贴多行指标，结束后按 Ctrl-D:")
        text = sys.stdin.read()

    rows = parse_metrics_text(text)
    if not rows:
        raise SystemExit("未解析到任何有效指标行，请检查输入格式。")

    output_csv = Path(args.output).resolve()
    write_csv(rows, output_csv)
    print(f"已生成 CSV: {output_csv}")


if __name__ == "__main__":
    main()
