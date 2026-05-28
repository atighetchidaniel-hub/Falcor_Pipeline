#!/usr/bin/env python3
"""Generate a Unity-style batch of Falcor synthetic scenes.

Unity's synthetic PVS generation repeatedly clears the scene, creates a fresh
random object layout, then writes one or more GV/PVV samples.  A single Falcor
.pyscene with 1000 camera samples does not reproduce that distribution because
all samples see the same object layout.

This helper creates many smaller .pyscene + camera CSV pairs. Export each part
with NeuralPVSExporter, then merge the exported GV/PVV folders with
merge_neuralpvs_datasets.py before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Root folder that will receive the batch parts.")
    parser.add_argument("--name", required=True,
                        help="Base name for the batch and generated part files.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scene-count", type=int, default=100,
                        help="Number of fresh synthetic scene parts to generate. "
                             "Default is aggressive Unity-parity: 100 layouts.")
    parser.add_argument("--samples-per-scene", type=int, default=10,
                        help="CSV samples per generated scene part.")
    parser.add_argument("--total-samples", type=int, default=None,
                        help="Optional total sample count. Overrides samples-per-scene by distributing samples evenly.")
    parser.add_argument("--fresh-scene-per-sample", action="store_true",
                        help="Most Unity-like mode: make one .pyscene per sample. "
                             "If --total-samples is omitted, uses scene-count * samples-per-scene.")
    parser.add_argument("--seed-stride", type=int, default=1009,
                        help="Seed increment between scene parts.")
    parser.add_argument("--radius", "--view-cell-radius-cm", dest="radius_cm", type=float, default=30.0,
                        help="View-cell radius in centimeters, e.g. 30, 60, or 90.")
    parser.add_argument("--glb-dir", type=Path, default=None)
    parser.add_argument("--generator", type=Path, default=None,
                        help="Path to generate_synthetic_neuralpvs_scene.py. Defaults to sibling script.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write the plan, do not generate part scenes.")
    parser.add_argument("generator_args", nargs=argparse.REMAINDER,
                        help="Extra arguments passed after -- to generate_synthetic_neuralpvs_scene.py.")
    return parser.parse_args()


def part_sample_count(args: argparse.Namespace, index: int) -> int:
    if args.total_samples is None:
        return args.samples_per_scene
    base = args.total_samples // args.scene_count
    rem = args.total_samples % args.scene_count
    return base + (1 if index < rem else 0)


def clean_remainder(args: List[str]) -> List[str]:
    return args[1:] if args and args[0] == "--" else args


def main() -> None:
    args = parse_args()
    if args.fresh_scene_per_sample:
        total_samples = args.total_samples or (args.scene_count * args.samples_per_scene)
        args.scene_count = total_samples
        args.samples_per_scene = 1
        args.total_samples = total_samples

    if args.scene_count < 1:
        raise ValueError("--scene-count must be >= 1")
    if args.samples_per_scene < 1:
        raise ValueError("--samples-per-scene must be >= 1")
    if args.total_samples is not None and args.total_samples < args.scene_count:
        raise ValueError("--total-samples must be >= --scene-count")

    script_dir = Path(__file__).resolve().parent
    generator = args.generator or (script_dir / "generate_synthetic_neuralpvs_scene.py")
    if not generator.is_file():
        raise FileNotFoundError(generator)

    batch_dir = args.out_dir / args.name
    batch_dir.mkdir(parents=True, exist_ok=True)

    extra_args = clean_remainder(args.generator_args)
    rows = []
    total = 0

    for i in range(args.scene_count):
        count = part_sample_count(args, i)
        total += count
        part_name = f"{args.name}_part{i:03d}"
        part_dir = batch_dir / f"part_{i:03d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + i * args.seed_stride

        cmd = [
            sys.executable,
            str(generator),
            "--unity-parity",
            "--out-dir", str(part_dir),
            "--name", part_name,
            "--seed", str(seed),
            "--samples", str(count),
            "--radius", str(args.radius_cm),
        ]
        if args.glb_dir is not None:
            cmd.extend(["--glb-dir", str(args.glb_dir)])
        cmd.extend(extra_args)

        scene_path = part_dir / f"{part_name}.pyscene"
        csv_path = part_dir / f"{part_name}_camera_path.csv"
        manifest_path = part_dir / f"{part_name}_manifest.json"
        rows.append({
            "part": i,
            "dataset_name": part_name,
            "seed": seed,
            "samples": count,
            "scene_path": str(scene_path),
            "camera_path_csv": str(csv_path),
            "manifest": str(manifest_path),
            "command": " ".join(cmd),
        })

        print("=" * 72)
        print(f"Part {i + 1}/{args.scene_count}: {part_name}")
        print(f"Seed: {seed}  Samples: {count}")
        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)

    plan_csv = batch_dir / f"{args.name}_export_plan.csv"
    with plan_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    batch_manifest = {
        "name": args.name,
        "scene_count": args.scene_count,
        "samples_per_scene": args.samples_per_scene,
        "total_samples": total,
        "fresh_scene_per_sample": args.fresh_scene_per_sample,
        "radius_cm": args.radius_cm,
        "generator": str(generator),
        "parts": rows,
    }
    manifest_path = batch_dir / f"{args.name}_batch_manifest.json"
    manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Wrote batch plan : {plan_csv}")
    print(f"Wrote manifest   : {manifest_path}")
    print(f"Total samples    : {total}")
    print("")
    print("Export each part in NeuralPVSExporter using the scene_path and")
    print("camera_path_csv columns, then merge exported datasets with:")
    print("")
    print("python Source/Samples/NeuralPVSExporter/tools/merge_neuralpvs_datasets.py \\")
    print(f"  --plan {plan_csv} \\")
    print("  --source-root <Falcor neuralpvs_export_test/datasets> \\")
    print("  --output-root <training root/datasets> \\")
    print(f"  --dataset-name {args.name}_merged \\")
    print("  --link")


if __name__ == "__main__":
    main()
