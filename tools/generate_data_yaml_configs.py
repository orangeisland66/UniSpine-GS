#!/usr/bin/env python3
import argparse
from pathlib import Path


CONFIG_TEMPLATE = """scene: {scene}
source_path: {source_path}
iterations: 20000
position_lr_init: 0.00019
position_lr_final: 0.0000019
position_lr_delay_mult: 0.01
position_lr_max_steps: 20000
feature_lr: 0.002
opacity_lr: 0.008
radiodensity_lr: 0.05
scaling_lr: 0.005
rotation_lr: 0.001
percent_dense: 0.01
lambda_dssim: 0.2
densification_interval: 200
opacity_reset_interval: 4000
radiodensity_reset_interval: 2000
densify_from_iter: 500
densify_until_iter: 8000
densify_grad_threshold: 0.000026
random_background: False
"""


def make_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate YAML configs for all pickle files directly under data/."
    )
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--config_dir", type=Path, default=Path("config"))
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing YAML files")
    parser.add_argument("--dry_run", action="store_true", help="show actions without writing files")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    config_dir = args.config_dir if args.config_dir.is_absolute() else project_root / args.config_dir

    pickle_files = sorted(path for path in data_dir.glob("*.pickle") if path.is_file())
    if not pickle_files:
        print(f"[Error] No pickle files found in {data_dir}")
        return 1

    if not args.dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for pickle_path in pickle_files:
        scene = pickle_path.stem
        source_path = make_relative(pickle_path, project_root).as_posix()
        config_path = config_dir / f"{scene}.yaml"

        if config_path.exists() and not args.overwrite:
            print(f"[Skip] {make_relative(config_path, project_root).as_posix()} exists")
            skipped += 1
            continue

        content = CONFIG_TEMPLATE.format(scene=scene, source_path=source_path)
        rel_config_path = make_relative(config_path, project_root).as_posix()
        if args.dry_run:
            action = "overwrite" if config_path.exists() else "create"
            print(f"[DryRun] {action} {rel_config_path}")
        else:
            config_path.write_text(content, encoding="utf-8")
            print(f"[Write] {rel_config_path}")
        written += 1

    print(f"[Done] found={len(pickle_files)}, written={written}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
