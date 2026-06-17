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
    parser = argparse.ArgumentParser(description="Convert batch evaluation text to CSV, including an average row.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input text file path. If omitted, read from standard input.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/metrics_from_text.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    else:
        if sys.stdin.isatty():
            print("Paste metric lines, then press Ctrl-D:")
        text = sys.stdin.read()

    rows = parse_metrics_text(text)
    if not rows:
        raise SystemExit("No valid metric lines were parsed. Please check the input format.")

    output_csv = Path(args.output).resolve()
    write_csv(rows, output_csv)
    print(f"CSV written to: {output_csv}")


if __name__ == "__main__":
    main()
