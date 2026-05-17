#!/usr/bin/env python3
"""Generate synthetic Falcor scenes and camera paths for NeuralPVS export.

The generated .pyscene file can be loaded by the NeuralPVSExporter sample.
The matching CSV provides viewcell/camera samples for GV/PVV generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Instance:
    name: str
    mesh: str
    material: str
    translation: tuple[float, float, float]
    scaling: tuple[float, float, float]
    rotation: tuple[float, float, float]


@dataclass(frozen=True)
class CameraSample:
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    fov: float


MATERIALS = {
    "floor": (0.55, 0.55, 0.52, 1.0),
    "wall": (0.62, 0.64, 0.66, 1.0),
    "red": (0.72, 0.22, 0.18, 1.0),
    "green": (0.22, 0.55, 0.30, 1.0),
    "blue": (0.22, 0.35, 0.70, 1.0),
    "yellow": (0.80, 0.65, 0.25, 1.0),
    "dark": (0.18, 0.20, 0.22, 1.0),
    "light": (0.86, 0.84, 0.78, 1.0),
}


def f3(value: Iterable[float]) -> str:
    x, y, z = value
    return f"float3({x:.5f}, {y:.5f}, {z:.5f})"


def f4(value: Iterable[float]) -> str:
    x, y, z, w = value
    return f"float4({x:.5f}, {y:.5f}, {z:.5f}, {w:.5f})"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-6:
        return (0.0, 0.0, -1.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def random_in_bounds(rng: random.Random, bounds: tuple[float, float, float]) -> tuple[float, float, float]:
    bx, by, bz = bounds
    return (
        rng.uniform(-0.45 * bx, 0.45 * bx),
        rng.uniform(0.2, max(0.25, 0.75 * by)),
        rng.uniform(-0.45 * bz, 0.45 * bz),
    )


def make_wall_segments(
    rng: random.Random,
    instances: list[Instance],
    bounds: tuple[float, float, float],
    wall_count: int,
) -> None:
    bx, by, bz = bounds
    for i in range(wall_count):
        along_x = rng.random() < 0.5
        height = rng.uniform(2.0, min(5.0, by))
        thickness = rng.uniform(0.12, 0.35)
        total_length = rng.uniform(0.45, 0.85) * (bx if along_x else bz)
        gap_width = rng.uniform(0.15, 0.35) * total_length
        gap_center = rng.uniform(-0.2, 0.2) * total_length
        fixed = rng.uniform(-0.35, 0.35) * (bz if along_x else bx)

        left_length = max(0.25, (total_length - gap_width) * 0.5 + gap_center)
        right_length = max(0.25, (total_length - gap_width) * 0.5 - gap_center)
        left_center = -0.5 * total_length + 0.5 * left_length
        right_center = 0.5 * total_length - 0.5 * right_length

        pieces = [
            ("left", left_center, left_length),
            ("right", right_center, right_length),
        ]
        if rng.random() < 0.55:
            lintel_height = rng.uniform(0.35, 0.9)
            pieces.append(("top", gap_center, gap_width, lintel_height))

        for piece in pieces:
            label = piece[0]
            center = piece[1]
            length = piece[2]
            piece_height = piece[3] if len(piece) == 4 else height
            y = height - piece_height * 0.5 if label == "top" else piece_height * 0.5
            if along_x:
                translation = (center, y, fixed)
                scaling = (length, piece_height, thickness)
            else:
                translation = (fixed, y, center)
                scaling = (thickness, piece_height, length)

            instances.append(
                Instance(
                    name=f"wall_{i:02d}_{label}",
                    mesh="cube",
                    material="wall",
                    translation=translation,
                    scaling=scaling,
                    rotation=(0.0, 0.0, 0.0),
                )
            )


def make_instances(args: argparse.Namespace) -> list[Instance]:
    rng = random.Random(args.seed)
    bounds = (args.bounds_x, args.bounds_y, args.bounds_z)
    instances: list[Instance] = []

    instances.append(
        Instance(
            name="floor",
            mesh="cube",
            material="floor",
            translation=(0.0, -0.06, 0.0),
            scaling=(args.bounds_x, 0.12, args.bounds_z),
            rotation=(0.0, 0.0, 0.0),
        )
    )

    # Thin boundary strips provide stable occluders without fully enclosing the scene.
    edge_height = min(2.5, args.bounds_y)
    edge_thickness = 0.18
    boundary_specs = [
        ("north", (0.0, edge_height * 0.5, 0.5 * args.bounds_z), (args.bounds_x, edge_height, edge_thickness)),
        ("south", (0.0, edge_height * 0.5, -0.5 * args.bounds_z), (args.bounds_x, edge_height, edge_thickness)),
        ("east", (0.5 * args.bounds_x, edge_height * 0.5, 0.0), (edge_thickness, edge_height, args.bounds_z)),
        ("west", (-0.5 * args.bounds_x, edge_height * 0.5, 0.0), (edge_thickness, edge_height, args.bounds_z)),
    ]
    for label, translation, scaling in boundary_specs:
        instances.append(
            Instance(
                name=f"boundary_{label}",
                mesh="cube",
                material="wall",
                translation=translation,
                scaling=scaling,
                rotation=(0.0, 0.0, 0.0),
            )
        )

    make_wall_segments(rng, instances, bounds, args.wall_count)

    cluster_centers = [random_in_bounds(rng, bounds) for _ in range(args.cluster_count)]
    object_materials = ["red", "green", "blue", "yellow", "dark", "light"]
    for i in range(args.object_count):
        if rng.random() < args.cluster_probability and cluster_centers:
            cx, cy, cz = rng.choice(cluster_centers)
            x = clamp(rng.gauss(cx, args.bounds_x * 0.08), -0.45 * args.bounds_x, 0.45 * args.bounds_x)
            z = clamp(rng.gauss(cz, args.bounds_z * 0.08), -0.45 * args.bounds_z, 0.45 * args.bounds_z)
        else:
            x, _, z = random_in_bounds(rng, bounds)

        sx = rng.uniform(args.min_object_scale, args.max_object_scale)
        sy = rng.uniform(args.min_object_scale, args.max_object_scale * 1.8)
        sz = rng.uniform(args.min_object_scale, args.max_object_scale)
        y = sy * 0.5
        if rng.random() < 0.12:
            y += rng.uniform(0.4, max(0.5, args.bounds_y * 0.45))

        instances.append(
            Instance(
                name=f"object_{i:04d}",
                mesh="sphere" if rng.random() < args.sphere_probability else "cube",
                material=rng.choice(object_materials),
                translation=(x, y, z),
                scaling=(sx, sy, sz),
                rotation=(rng.uniform(0, 20), rng.uniform(0, 360), rng.uniform(0, 20)),
            )
        )

    return instances


def make_camera_samples(args: argparse.Namespace) -> list[CameraSample]:
    rng = random.Random(args.seed + 9173)
    samples: list[CameraSample] = []
    radius_min = max(args.bounds_x, args.bounds_z) * 0.18
    radius_max = max(args.bounds_x, args.bounds_z) * 0.55

    for _ in range(args.camera_count):
        angle = rng.uniform(0.0, math.tau)
        radius = rng.uniform(radius_min, radius_max)
        height = rng.uniform(1.0, min(args.bounds_y * 0.85, 5.0))
        target = (
            rng.uniform(-args.bounds_x * 0.2, args.bounds_x * 0.2),
            rng.uniform(0.7, min(args.bounds_y * 0.5, 3.0)),
            rng.uniform(-args.bounds_z * 0.2, args.bounds_z * 0.2),
        )
        position = (
            target[0] + math.cos(angle) * radius,
            height,
            target[2] + math.sin(angle) * radius,
        )
        forward = normalize((target[0] - position[0], target[1] - position[1], target[2] - position[2]))
        samples.append(CameraSample(position=position, forward=forward, fov=args.fov))

    return samples


def write_scene(path: Path, instances: list[Instance], first_camera: CameraSample) -> None:
    lines: list[str] = [
        "# Generated by generate_synthetic_neuralpvs_scene.py",
        "# Load this file in the NeuralPVSExporter sample.",
        "",
        "cubeMesh = TriangleMesh.createCube()",
        "sphereMesh = TriangleMesh.createSphere()",
        "",
    ]

    for name, color in MATERIALS.items():
        lines.extend(
            [
                f"{name}Mat = StandardMaterial('{name}')",
                f"{name}Mat.baseColor = {f4(color)}",
                f"{name}Mat.roughness = 0.75",
                f"{name}Mat.metallic = 0.0",
                "",
            ]
        )

    for name in MATERIALS:
        lines.append(f"{name}CubeMeshID = sceneBuilder.addTriangleMesh(cubeMesh, {name}Mat)")
        lines.append(f"{name}SphereMeshID = sceneBuilder.addTriangleMesh(sphereMesh, {name}Mat)")
    lines.append("")

    for instance in instances:
        mesh_id = f"{instance.material}{'Sphere' if instance.mesh == 'sphere' else 'Cube'}MeshID"
        node_name = instance.name.replace("'", "_")
        lines.append(
            f"{node_name}NodeID = sceneBuilder.addNode("
            f"'{node_name}', "
            f"Transform(scaling={f3(instance.scaling)}, "
            f"translation={f3(instance.translation)}, "
            f"rotationEulerDeg={f3(instance.rotation)}))"
        )
        lines.append(f"sceneBuilder.addMeshInstance({node_name}NodeID, {mesh_id})")
    lines.append("")

    position = first_camera.position
    target = (
        position[0] + first_camera.forward[0],
        position[1] + first_camera.forward[1],
        position[2] + first_camera.forward[2],
    )
    lines.extend(
        [
            "camera = Camera('SyntheticCamera0')",
            f"camera.position = {f3(position)}",
            f"camera.target = {f3(target)}",
            "camera.up = float3(0.0, 1.0, 0.0)",
            "camera.focalLength = 35.0",
            "sceneBuilder.addCamera(camera)",
            "sceneBuilder.selectedCamera = camera",
            "",
            "keyLight = DistantLight('KeyLight')",
            "keyLight.direction = float3(-0.4, -1.0, -0.3)",
            "keyLight.intensity = float3(3.0, 3.0, 3.0)",
            "keyLight.angle = 0.1",
            "sceneBuilder.addLight(keyLight)",
            "",
            "fillLight = DistantLight('FillLight')",
            "fillLight.direction = float3(0.5, -0.6, 0.7)",
            "fillLight.intensity = float3(0.8, 0.9, 1.0)",
            "fillLight.angle = 0.2",
            "sceneBuilder.addLight(fillLight)",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_camera_path(path: Path, samples: list[CameraSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["x", "y", "z", "forward_x", "forward_y", "forward_z", "fov"])
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.position[0]:.6f}",
                    f"{sample.position[1]:.6f}",
                    f"{sample.position[2]:.6f}",
                    f"{sample.forward[0]:.6f}",
                    f"{sample.forward[1]:.6f}",
                    f"{sample.forward[2]:.6f}",
                    f"{sample.fov:.4f}",
                ]
            )


def write_manifest(path: Path, args: argparse.Namespace, scene_path: Path, csv_path: Path, instances: list[Instance]) -> None:
    manifest = {
        "name": args.name,
        "seed": args.seed,
        "scene_path": str(scene_path),
        "camera_path_csv": str(csv_path),
        "object_count": args.object_count,
        "wall_count": args.wall_count,
        "camera_count": args.camera_count,
        "bounds": [args.bounds_x, args.bounds_y, args.bounds_z],
        "suggested_neuralpvs_exporter_settings": {
            "mode": "Generate GV + PVV",
            "volume_size": args.volume_size,
            "volume_depth": args.volume_depth,
            "camera_aspect_ratio": args.aspect_ratio,
            "view_cell_radius": args.view_cell_radius,
            "view_cell_near": args.near,
            "view_cell_far": args.far,
            "pvv_sample_steps": args.pvv_sample_steps,
            "linear_z": args.linear_z,
            "log_depth_scale": args.log_depth_scale,
            "unity_fov_expansion_degrees": args.unity_fov_expansion_degrees,
            "dataset_name": args.name,
        },
        "notes": [
            "The .pyscene contains synthetic geometry only; GV/PVV files are generated by NeuralPVSExporter.",
            "Use the CSV as the exporter path CSV so every row becomes a viewcell/camera sample.",
            "The exporter writes ground-truth PVV files; neural predicted PVVs are produced later by the fVDB backend.",
        ],
        "generated_instance_count": len(instances),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory that receives the scene, path CSV, and manifest.")
    parser.add_argument("--name", default="synthetic_neuralpvs_scene", help="Base name for generated files and dataset.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--object-count", type=int, default=180)
    parser.add_argument("--wall-count", type=int, default=14)
    parser.add_argument("--cluster-count", type=int, default=8)
    parser.add_argument("--cluster-probability", type=float, default=0.72)
    parser.add_argument("--sphere-probability", type=float, default=0.25)
    parser.add_argument("--min-object-scale", type=float, default=0.25)
    parser.add_argument("--max-object-scale", type=float, default=1.25)
    parser.add_argument("--bounds-x", type=float, default=18.0)
    parser.add_argument("--bounds-y", type=float, default=7.0)
    parser.add_argument("--bounds-z", type=float, default=18.0)
    parser.add_argument("--camera-count", type=int, default=128)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--aspect-ratio", type=float, default=1.777778)
    parser.add_argument("--view-cell-radius", type=float, default=0.3)
    parser.add_argument("--near", type=float, default=0.3)
    parser.add_argument("--far", type=float, default=30.0)
    parser.add_argument("--pvv-sample-steps", type=int, default=10)
    parser.add_argument("--linear-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-depth-scale", type=float, default=0.01)
    parser.add_argument("--unity-fov-expansion-degrees", type=float, default=30.0)
    parser.add_argument("--volume-size", type=int, default=256)
    parser.add_argument("--volume-depth", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.object_count < 0 or args.wall_count < 0 or args.camera_count < 1:
        raise ValueError("object-count and wall-count must be non-negative; camera-count must be at least 1.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.out_dir / f"{args.name}.pyscene"
    csv_path = args.out_dir / f"{args.name}_camera_path.csv"
    manifest_path = args.out_dir / f"{args.name}_manifest.json"

    instances = make_instances(args)
    cameras = make_camera_samples(args)
    write_scene(scene_path, instances, cameras[0])
    write_camera_path(csv_path, cameras)
    write_manifest(manifest_path, args, scene_path, csv_path, instances)

    print(f"Wrote scene: {scene_path}")
    print(f"Wrote camera path: {csv_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
