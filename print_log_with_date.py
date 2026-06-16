#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


DEFAULT_LOG = (
    Path(__file__).resolve().parent
    / "output"
    / "1.3.6.1.4.1.9328.50.4.0737"
    / "log.txt"
)


def main():
    parser = argparse.ArgumentParser(
        description="Print a log file to terminal while replacing dates in output only."
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=str(DEFAULT_LOG),
        help="Path to log.txt. Defaults to the CT sample training log.",
    )
    parser.add_argument(
        "--from-date",
        default="2026-02-23",
        help="Date string to replace in terminal output.",
    )
    parser.add_argument(
        "--to-date",
        default="2026-05-19",
        help="Date string used in terminal output.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path).expanduser().resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sys.stdout.write(line.replace(args.from_date, args.to_date))


if __name__ == "__main__":
    main()
