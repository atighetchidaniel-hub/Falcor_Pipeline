#!/usr/bin/env python3
"""Generate synthetic Falcor scenes and camera paths for NeuralPVS export.

Unity-style synthetic scene features inspired by Unity's RuntimeSceneGenerator:
  - Advanced non-spherical Gaussian clustering with per-cluster rotation, density, and position noise
  - Non-uniform per-axis scaling with bias, extreme-scaling probability, aspect-ratio constraints,
    and optional global scale variation
  - True per-object random colours (not a fixed palette)
  - Cylinder and capsule mesh primitives in addition to cubes and spheres
  - Boolean clearing zones (tunnels / roads / spherical cavities) that remove objects they overlap
  - Random thin environmental planes around the scene
  - Orbit-based camera sampling with randomised distance, height, tilt, target offset, and FOV
  - Optional Unity-parity preset matching RuntimeSceneGenerator defaults as closely as Falcor allows

The generated .pyscene file can be loaded by the NeuralPVSExporter sample.
The matching CSV provides viewcell/camera samples for GV/PVV generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional trimesh — used as a fallback for non-GLB 3D formats.
# .glb files are parsed with the built-in pure-Python parser (_parse_glb) below.
try:
    import trimesh as _trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


# ---------------------------------------------------------------------------
# Small vector helpers (no numpy dependency)
# ---------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]


def v3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v3_length(a: Vec3) -> float:
    return math.sqrt(v3_dot(a, a))


def v3_normalize(a: Vec3, fallback: Vec3 = (0.0, 0.0, -1.0)) -> Vec3:
    l = v3_length(a)
    return (a[0] / l, a[1] / l, a[2] / l) if l > 1e-6 else fallback


# 3x3 rotation matrix (row-major, 9 floats)
Mat3 = Tuple[float, ...]


def mat3_from_euler_xyz_deg(rx: float, ry: float, rz: float) -> Mat3:
    """Rotation matrix from Euler angles (degrees) applied X->Y->Z."""
    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    return (
        cy*cz,  sz*sx*sy + cx*cz,  cx*sz*sy - sx*cz,
        -sz*cy, cx*cz - sx*sy*sz,  cx*sy*sz + sx*cz,
        sy,     -sx*cy,             cx*cy,
    )


def mat3_mul_vec3(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
        m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
        m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
    )


# ---------------------------------------------------------------------------
# Normal distribution (Box-Muller, clamped to ±3 sigma)
# ---------------------------------------------------------------------------

def gauss_01(rng: random.Random) -> float:
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return max(-3.0, min(3.0, z))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GaussianCluster:
    center: Vec3
    scale: Vec3      # per-axis sigma multiplier (non-uniform / elliptical)
    rotation: Mat3   # cluster orientation
    density: float   # relative spawn weight


@dataclass
class BooleanZone:
    """Clearing zone that removes objects it overlaps (mirrors Unity's Boolean ops)."""
    shape: str           # "sphere" | "box" | "cylinder" | "capsule"
    position: Vec3
    half_extents: Vec3   # half-size per axis
    rotation_y: float    # degrees


@dataclass
class Instance:
    name: str
    mesh: str        # "cube" | "sphere" | "cylinder" | "capsule"
    color_idx: int
    translation: Vec3
    scaling: Vec3
    rotation: Vec3   # Euler degrees (x, y, z)


@dataclass
class CameraSample:
    position: Vec3
    forward: Vec3
    fov: float


@dataclass
class SceneBounds:
    center: Vec3
    size: Vec3
    min_corner: Vec3
    max_corner: Vec3


@dataclass
class GlbModel:
    """One GLB file from the model library — mirrors Unity's ModelDefinition."""
    name: str             # human-readable name, e.g. "Dodecahedron"
    path: Path            # source .glb file
    category: str         # "Props" | "Containers" | "Buildings"
    size: str             # "Small" | "Medium" | "Large"
    spawn_weight: float   # relative selection weight
    key: str              # Instance.mesh token, e.g. "glb_dodecahedron"
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    normals:  List[Tuple[float, float, float]] = field(default_factory=list)
    faces:    List[Tuple[int, int, int]]        = field(default_factory=list)


# ---------------------------------------------------------------------------
# Falcor pyscene formatting
# ---------------------------------------------------------------------------

def f3(v: Iterable[float]) -> str:
    x, y, z = v
    return f"float3({x:.5f}, {y:.5f}, {z:.5f})"


def f4(v: Iterable[float]) -> str:
    x, y, z, w = v
    return f"float4({x:.5f}, {y:.5f}, {z:.5f}, {w:.5f})"


def euler_abs_extents(rotation_deg: Vec3, half_extents: Vec3) -> Vec3:
    """Axis-aligned half extents after applying the instance Euler rotation."""
    m = mat3_from_euler_xyz_deg(rotation_deg[0], rotation_deg[1], rotation_deg[2])
    hx, hy, hz = half_extents
    return (
        abs(m[0]) * hx + abs(m[1]) * hy + abs(m[2]) * hz,
        abs(m[3]) * hx + abs(m[4]) * hy + abs(m[5]) * hz,
        abs(m[6]) * hx + abs(m[7]) * hy + abs(m[8]) * hz,
    )


# ---------------------------------------------------------------------------
# Random colour generation  (true per-object random, not a fixed palette)
# ---------------------------------------------------------------------------

def make_random_colors(rng: random.Random, n: int) -> List[Tuple[float, float, float, float]]:
    """n distinct random RGBA colours in [0.20, 0.90] — never pure black or white."""
    return [(rng.uniform(0.20, 0.90), rng.uniform(0.20, 0.90), rng.uniform(0.20, 0.90), 1.0)
            for _ in range(n)]


# ---------------------------------------------------------------------------
# Advanced non-spherical Gaussian clustering
# ---------------------------------------------------------------------------

def make_clusters(rng: random.Random, args: argparse.Namespace) -> List[GaussianCluster]:
    bx, by, bz = args.bounds_x, args.bounds_y, args.bounds_z
    sv = args.cluster_shape_variation
    size_lo, size_hi = args.cluster_size_range
    dens_lo, dens_hi = args.cluster_density_range
    pn = args.cluster_position_noise
    rv = args.cluster_rotation_variation
    clusters: List[GaussianCluster] = []

    for _ in range(args.cluster_count):
        # Base centre. Unity's RuntimeSceneGenerator samples the full centered
        # spawn volume; the older Falcor preset biased objects above a floor.
        cx = rng.uniform(-0.40 * bx, 0.40 * bx)
        if args.centered_y_distribution:
            cy = rng.uniform(-0.40 * by, 0.40 * by)
        else:
            cy = rng.uniform(0.10 * by, 0.75 * by)
        cz = rng.uniform(-0.40 * bz, 0.40 * bz)
        # Position noise
        if pn > 0:
            ns = pn * 0.2
            cx += gauss_01(rng) * bx * ns
            cy += gauss_01(rng) * by * ns
            cz += gauss_01(rng) * bz * ns
        cx = clamp(cx, -0.45 * bx, 0.45 * bx)
        if args.centered_y_distribution:
            cy = clamp(cy, -0.45 * by, 0.45 * by)
        else:
            cy = clamp(cy, 0.05 * by, 0.90 * by)
        cz = clamp(cz, -0.45 * bz, 0.45 * bz)

        # Elliptical scale
        base_inf = rng.uniform(size_lo, size_hi)
        def _axis_scale() -> float:
            return base_inf * (rng.uniform(0.3, 1.7) * sv + 1.0 * (1.0 - sv))
        sx, sy_c, sz = _axis_scale(), _axis_scale(), _axis_scale()

        # Random orientation
        rot = mat3_from_euler_xyz_deg(
            rng.uniform(-180, 180) * rv,
            rng.uniform(-180, 180) * rv,
            rng.uniform(-180, 180) * rv,
        )

        clusters.append(GaussianCluster(
            center=(cx, cy, cz),
            scale=(sx, sy_c, sz),
            rotation=rot,
            density=rng.uniform(dens_lo, dens_hi),
        ))
    return clusters


def select_weighted_cluster(rng: random.Random, clusters: List[GaussianCluster]) -> GaussianCluster:
    total = sum(c.density for c in clusters)
    r = rng.uniform(0, total)
    acc = 0.0
    for c in clusters:
        acc += c.density
        if r <= acc:
            return c
    return clusters[-1]


def sample_cluster_position(rng: random.Random, c: GaussianCluster, bounds: Vec3) -> Vec3:
    bx, by, bz = bounds
    noise = (
        gauss_01(rng) * c.scale[0] * bx * 0.15,
        gauss_01(rng) * c.scale[1] * by * 0.15,
        gauss_01(rng) * c.scale[2] * bz * 0.15,
    )
    rotated = mat3_mul_vec3(c.rotation, noise)
    return v3_add(c.center, rotated)


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def generate_position(rng: random.Random, args: argparse.Namespace,
                      clusters: List[GaussianCluster]) -> Vec3:
    bx, by, bz = args.bounds_x, args.bounds_y, args.bounds_z
    if clusters and rng.random() < args.clustering_intensity:
        c = select_weighted_cluster(rng, clusters)
        p = sample_cluster_position(rng, c, (bx, by, bz))
    else:
        y = rng.uniform(-0.50 * by, 0.50 * by) if args.centered_y_distribution else rng.uniform(0.05 * by, 0.90 * by)
        p = (rng.uniform(-0.45*bx, 0.45*bx), y, rng.uniform(-0.45*bz, 0.45*bz))

    y_min, y_max = (-0.50 * by, 0.50 * by) if args.centered_y_distribution else (0.02 * by, 0.95 * by)
    return (clamp(p[0], -0.45*bx, 0.45*bx),
            clamp(p[1], y_min, y_max),
            clamp(p[2], -0.45*bz, 0.45*bz))


# ---------------------------------------------------------------------------
# Scaling  (matches Unity ScalingMode Uniform / NonUniform + all modifiers)
# ---------------------------------------------------------------------------

def _bias(rng: random.Random, bias: float) -> float:
    v = rng.random()
    if bias == 0.5:
        return v
    return v ** (1.0 / (bias * 2.0)) if bias < 0.5 else v ** (bias * 2.0)


def generate_scale(rng: random.Random, args: argparse.Namespace) -> Vec3:
    if args.scaling_mode == "uniform":
        s = args.uniform_scale_min + _bias(rng, args.scaling_bias) * (args.uniform_scale_max - args.uniform_scale_min)
        sx = sy = sz = s
    else:
        sx = args.scale_min_x + _bias(rng, args.scaling_bias) * (args.scale_max_x - args.scale_min_x)
        sy = args.scale_min_y + _bias(rng, args.scaling_bias) * (args.scale_max_y - args.scale_min_y)
        sz = args.scale_min_z + _bias(rng, args.scaling_bias) * (args.scale_max_z - args.scale_min_z)
        # Extreme scaling: amplify one random axis
        if rng.random() < args.extreme_scaling_probability:
            ax = rng.randint(0, 2)
            ex = args.extreme_scale_min + rng.random() * (args.extreme_scale_max - args.extreme_scale_min)
            if ax == 0: sx *= ex
            elif ax == 1: sy *= ex
            else: sz *= ex

    # Aspect-ratio constraint
    if args.maintain_aspect_ratio:
        mx = max(sx, sy, sz)
        mn = min(sx, sy, sz)
        if mn > 0 and mx / mn > args.max_aspect_ratio:
            floor_v = mx / args.max_aspect_ratio
            sx, sy, sz = max(sx, floor_v), max(sy, floor_v), max(sz, floor_v)

    # Global scale variation (balanced around 1.0 — 50/50 below/above)
    if args.global_scale_variation:
        lo, hi = args.global_scale_min, args.global_scale_max
        def _balanced_g() -> float:
            return (lo + rng.random() * (1.0 - lo)) if rng.random() < 0.5 else (1.0 + rng.random() * (hi - 1.0))
        if args.uniform_global_scaling:
            g = _balanced_g()
            sx, sy, sz = sx*g, sy*g, sz*g
        else:
            sx, sy, sz = sx*_balanced_g(), sy*_balanced_g(), sz*_balanced_g()

    return (max(sx, 0.01), max(sy, 0.01), max(sz, 0.01))


# ---------------------------------------------------------------------------
# Boolean clearing zones  (tunnels / roads / cavities remove overlapping objects)
# ---------------------------------------------------------------------------

def generate_boolean_zones(rng: random.Random, args: argparse.Namespace) -> List[BooleanZone]:
    zones: List[BooleanZone] = []
    if args.boolean_count <= 0:
        return zones
    bx, by, bz = args.bounds_x, args.bounds_y, args.bounds_z
    diag = math.sqrt(bx*bx + by*by + bz*bz)
    shapes = ["sphere", "box", "cylinder", "capsule"]
    weights = args.boolean_shape_weights

    for _ in range(args.boolean_count):
        # Weighted shape selection
        total_w = sum(weights)
        r_w = rng.uniform(0, total_w)
        acc = 0.0
        shape = "box"
        for s, w in zip(shapes, weights):
            acc += w
            if r_w <= acc:
                shape = s
                break

        # Special overrides: tunnel / road — cumulative thresholds so road branch is reachable.
        # With defaults tunnel=0.50, road=0.30: [0,0.50)→tunnel, [0.50,0.80)→road, else keep shape.
        r_sp = rng.random()
        tunnel_p = args.tunnel_probability if args.enable_tunnels else 0.0
        road_p   = args.road_probability   if args.enable_roads   else 0.0
        if r_sp < tunnel_p:
            shape = "cylinder"
        elif r_sp < tunnel_p + road_p:
            shape = "box"

        # Position (inside bounds minus buffer)
        buf = args.boolean_boundary_buffer
        px = rng.uniform(-bx*(0.5 - buf), bx*(0.5 - buf))
        py = rng.uniform(by*buf, by*(1.0 - buf))
        pz = rng.uniform(-bz*(0.5 - buf), bz*(0.5 - buf))

        ratio = rng.uniform(args.boolean_size_ratio_min, args.boolean_size_ratio_max)
        base = diag * ratio
        ext = args.boolean_extension_factor if args.boolean_extend_beyond_bounds else 1.0

        if shape == "sphere":
            r = base * rng.uniform(0.5, 1.1) * 0.5
            he: Vec3 = (r, r, r)
        elif shape == "box":
            if args.enable_roads and rng.random() < 0.6:
                # Road: wide, long, flat
                he = (bx*rng.uniform(0.4, 0.8)*ext*0.5,
                      by*rng.uniform(0.05, 0.15)*0.5,
                      bz*rng.uniform(0.5, 1.0)*ext*0.5)
            else:
                he = (base*0.5, base*rng.uniform(0.5, 1.0)*0.5, base*0.5)
        elif shape == "cylinder":
            if args.enable_tunnels and rng.random() < 0.7:
                # Tunnel: narrow long cylinder
                radius = base*rng.uniform(0.15, 0.4)*0.5
                length = base*rng.uniform(1.5, 3.0)*ext*0.5
                he = (radius, length, radius)
            else:
                r = base*rng.uniform(0.25, 0.6)*0.5
                he = (r, base*rng.uniform(0.8, 1.5)*0.5, r)
        else:  # capsule
            r = base*rng.uniform(0.2, 0.45)*0.5
            he = (r, base*rng.uniform(1.2, 2.5)*0.5, r)

        rot_y = rng.uniform(0.0, 360.0) if args.boolean_random_rotation else 0.0
        zones.append(BooleanZone(shape=shape, position=(px, py, pz),
                                 half_extents=he, rotation_y=rot_y))
    return zones


def point_in_zone(p: Vec3, z: BooleanZone, expand: float = 0.0) -> bool:
    """Check if p is inside the clearing zone (Y-rotated AABB / sphere test).

    expand: conservative extra radius added to each half-extent axis before the
    test.  Pass the object's bounding radius so that the full object volume is
    checked, not just its centre point.
    """
    cos_r = math.cos(math.radians(-z.rotation_y))
    sin_r = math.sin(math.radians(-z.rotation_y))
    lx = cos_r*(p[0]-z.position[0]) - sin_r*(p[2]-z.position[2])
    ly = p[1] - z.position[1]
    lz = sin_r*(p[0]-z.position[0]) + cos_r*(p[2]-z.position[2])
    he = (z.half_extents[0] + expand,
          z.half_extents[1] + expand,
          z.half_extents[2] + expand)
    if z.shape == "sphere":
        return (lx/max(he[0], 1e-6))**2 + (ly/max(he[1], 1e-6))**2 + (lz/max(he[2], 1e-6))**2 <= 1.0
    return abs(lx) <= he[0] and abs(ly) <= he[1] and abs(lz) <= he[2]


# ---------------------------------------------------------------------------
# Environmental planes (random thin slabs around the scene)
# ---------------------------------------------------------------------------

def make_env_planes(rng: random.Random, args: argparse.Namespace,
                    color_count: int) -> List[Instance]:
    if args.max_plane_count <= 0:
        return []
    bx, by, bz = args.bounds_x, args.bounds_y, args.bounds_z
    diag = math.sqrt(bx*bx + bz*bz)
    planes: List[Instance] = []
    n = rng.randint(1, args.max_plane_count)
    for i in range(n):
        theta = rng.uniform(0, 2*math.pi)
        dist = diag * 0.5 * rng.uniform(0.3, 1.0)
        px, py, pz = math.cos(theta)*dist, rng.uniform(0.1*by, 0.9*by), math.sin(theta)*dist
        size = diag * rng.uniform(0.3, 0.9)
        thick = diag * rng.uniform(0.01, 0.05)
        if rng.random() < 0.4:
            sx_p = size * rng.uniform(0.25, 0.7) if rng.random() < 0.5 else size
            sz_p = size if sx_p != size else size * rng.uniform(0.25, 0.7)
        else:
            sx_p = sz_p = size
        planes.append(Instance(
            name=f"env_plane_{i:03d}",
            mesh="cube",
            color_idx=rng.randint(0, color_count - 1),
            translation=(px, py, pz),
            scaling=(sx_p, thick, sz_p),
            rotation=(rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360)),
        ))
    return planes


# ---------------------------------------------------------------------------
# Interior wall segments with doorway gaps
# ---------------------------------------------------------------------------

def make_wall_segments(rng: random.Random, args: argparse.Namespace,
                       color_count: int, instances: List[Instance]) -> None:
    bx, by, bz = args.bounds_x, args.bounds_y, args.bounds_z
    wall_cidx = 1  # reserved wall colour index

    for i in range(args.wall_count):
        along_x = rng.random() < 0.5
        height = rng.uniform(2.0, min(5.0, by))
        thickness = rng.uniform(0.12, 0.35)
        span = bx if along_x else bz
        total_len = rng.uniform(0.45, 0.85) * span
        gap_w = rng.uniform(0.15, 0.35) * total_len
        gap_c = rng.uniform(-0.2, 0.2) * total_len
        fixed = rng.uniform(-0.35, 0.35) * (bz if along_x else bx)

        left_len = max(0.25, (total_len - gap_w) / 2.0 + gap_c)
        right_len = max(0.25, (total_len - gap_w) / 2.0 - gap_c)
        left_c = -0.5 * total_len + 0.5 * left_len
        right_c = 0.5 * total_len - 0.5 * right_len

        pieces = [("left", left_c, left_len, height),
                  ("right", right_c, right_len, height)]
        if rng.random() < 0.55:
            lintel_h = rng.uniform(0.35, 0.9)
            pieces.append(("top", gap_c, gap_w, lintel_h))

        for label, cpos, plen, ph in pieces:
            y = (height - ph * 0.5) if label == "top" else ph * 0.5
            if along_x:
                tr: Vec3 = (cpos, y, fixed)
                sc: Vec3 = (plen, ph, thickness)
            else:
                tr = (fixed, y, cpos)
                sc = (thickness, ph, plen)
            instances.append(Instance(
                name=f"wall_{i:02d}_{label}",
                mesh="cube",
                color_idx=wall_cidx,
                translation=tr,
                scaling=sc,
                rotation=(0.0, 0.0, 0.0),
            ))


# ---------------------------------------------------------------------------
# GLB model support  (mirrors Unity's GLBModelCollection)
# ---------------------------------------------------------------------------

# Default category weights — from Unity's GLBModelCollection.categoryWeights
_GLB_CATEGORY_WEIGHTS: Dict[str, float] = {
    "Props":      2.0,   # Geometric shapes — platonic solids, pyramid, cone
    "Containers": 1.5,   # Boxes, cylinders, prisms
    "Buildings":  0.3,   # Arches, windows (rare)
}

# Default size distribution — from Unity's GLBModelCollection.sizeDistribution
_GLB_SIZE_DISTRIBUTION: Dict[str, float] = {
    "Small":  0.50,   # Platonic solids, pyramid, cone
    "Medium": 0.30,   # Box, cylinder, monkey, prism
    "Large":  0.20,   # Arches, windows
}


def _auto_categorize_glb(name: str) -> Tuple[str, str, float]:
    """Return (category, size, spawn_weight) — mirrors Unity's ModelDefinition.AutoCategorize."""
    n = name.lower()
    if any(k in n for k in ("arch", "wall", "window")):
        return "Buildings", "Large", 0.5
    if any(k in n for k in ("box", "cylinder", "prism")):
        return "Containers", "Medium", 1.5
    if any(k in n for k in ("pyramid", "cone", "dodecahedron", "icosahedron")):
        return "Props", "Small", 2.0
    if "monkey" in n:
        return "Props", "Medium", 0.8
    return "Props", "Medium", 1.0


def _glb_key(name: str) -> str:
    """Convert any filename stem to a valid Python identifier prefixed with 'glb_'.

    Handles dots, parentheses, leading digits, unicode, and consecutive separators
    that simple str.replace misses.  Examples:
        '12_dodecahedron' → 'glb_m12_dodecahedron'  (leading digit guarded)
        'My Arch (v2).glb' → 'glb_my_arch__v2_'
        'Thin-Rect.Arch'  → 'glb_thin_rect_arch'
    """
    safe = re.sub(r'[^a-z0-9]+', '_', name.lower())  # collapse non-alnum runs → _
    safe = safe.strip('_')                             # strip leading/trailing _
    if not safe:
        safe = 'model'
    if safe[0].isdigit():                              # identifiers can't start with digit
        safe = 'm' + safe
    return f"glb_{safe}"


def _parse_glb(path: Path) -> Tuple[List, List, List]:
    """Pure-Python glTF 2.0 binary (.glb) parser — no external dependencies.

    Returns raw (vertices, normals, face_indices) lists with all primitives from
    all meshes concatenated.  Vertices are NOT yet normalised here.
    """
    import struct
    import json as _json

    data = path.read_bytes()

    # --- 12-byte GLB header ---
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a valid GLB file (bad magic): {path}")
    if version != 2:
        raise ValueError(f"Only glTF 2.0 is supported (got version {version}): {path}")

    # --- JSON chunk (always first) ---
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:   # "JSON"
        raise ValueError(f"Expected JSON chunk first in {path}")
    gltf = _json.loads(data[20:20 + json_len])

    # --- BIN chunk (optional) ---
    bin_start = 20 + json_len
    bin_data  = b""
    if bin_start + 8 <= len(data):
        bin_len, bin_type = struct.unpack_from("<II", data, bin_start)
        if bin_type == 0x004E4942:   # "BIN\0"
            bin_data = data[bin_start + 8: bin_start + 8 + bin_len]

    # --- Accessor reader ---
    _COMP_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
    _TYPE_ELEMS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                   "MAT2": 4, "MAT3": 9, "MAT4": 16}

    def read_accessor(acc_idx: Optional[int]) -> Optional[list]:
        if acc_idx is None:
            return None
        acc = gltf["accessors"][acc_idx]
        bv_idx = acc.get("bufferView")
        if bv_idx is None:
            return None                    # sparse / zero accessor
        bv       = gltf["bufferViews"][bv_idx]
        count    = acc["count"]
        n_elem   = _TYPE_ELEMS.get(acc["type"], 1)
        fmt_char = _COMP_FMT.get(acc["componentType"], "f")
        comp_sz  = struct.calcsize(fmt_char)
        stride   = bv.get("byteStride") or (n_elem * comp_sz)
        base     = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        result   = []
        fmt      = f"<{n_elem}{fmt_char}"
        for i in range(count):
            vals = struct.unpack_from(fmt, bin_data, base + i * stride)
            result.append(vals if n_elem > 1 else vals[0])
        return result

    # --- Minimal transform support for GLB scene nodes ---
    # glTF matrices are column-major. Internally we use row-major 4x4 matrices
    # with column-vector math: p' = M * p.
    def mat_identity() -> Tuple[float, ...]:
        return (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )

    def mat_mul(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
        out = []
        for r in range(4):
            for c in range(4):
                out.append(sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)))
        return tuple(out)

    def mat_from_node(node: dict) -> Tuple[float, ...]:
        if "matrix" in node:
            col_major = node["matrix"]
            return tuple(float(col_major[c * 4 + r]) for r in range(4) for c in range(4))

        tx, ty, tz = node.get("translation", [0.0, 0.0, 0.0])
        sx, sy, sz = node.get("scale", [1.0, 1.0, 1.0])
        qx, qy, qz, qw = node.get("rotation", [0.0, 0.0, 0.0, 1.0])

        q_len = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if q_len > 1e-8:
            qx, qy, qz, qw = qx / q_len, qy / q_len, qz / q_len, qw / q_len

        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        return (
            r00 * sx, r01 * sy, r02 * sz, tx,
            r10 * sx, r11 * sy, r12 * sz, ty,
            r20 * sx, r21 * sy, r22 * sz, tz,
            0.0,      0.0,      0.0,      1.0,
        )

    def transform_point(m: Tuple[float, ...], p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            m[0] * p[0] + m[1] * p[1] + m[2]  * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6]  * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
        )

    def transform_dir(m: Tuple[float, ...], n: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return v3_normalize((
            m[0] * n[0] + m[1] * n[1] + m[2]  * n[2],
            m[4] * n[0] + m[5] * n[1] + m[6]  * n[2],
            m[8] * n[0] + m[9] * n[1] + m[10] * n[2],
        ), (0.0, 1.0, 0.0))

    # --- Collect all mesh primitives ---
    all_verts: List = []
    all_norms: List = []
    all_faces: List = []

    def append_mesh(mesh_index: int, transform: Tuple[float, ...]) -> None:
        meshes = gltf.get("meshes", [])
        if mesh_index < 0 or mesh_index >= len(meshes):
            return

        for prim in meshes[mesh_index].get("primitives", []):
            # Only triangle lists are used by the generated Falcor mesh builders.
            if prim.get("mode", 4) != 4:
                continue

            attrs     = prim.get("attributes", {})
            positions = read_accessor(attrs.get("POSITION"))
            normals   = read_accessor(attrs.get("NORMAL"))
            indices   = read_accessor(prim.get("indices"))

            if not positions:
                continue

            base = len(all_verts)
            all_verts.extend(transform_point(transform, tuple(p)) for p in positions)

            if normals:
                all_norms.extend(transform_dir(transform, tuple(n)) for n in normals)
            else:
                # Fallback: simple up-vector (orientation irrelevant for GV raster)
                all_norms.extend([(0.0, 1.0, 0.0)] * len(positions))

            if indices is not None:
                for t in range(0, len(indices) - 2, 3):
                    all_faces.append((base + int(indices[t]),
                                      base + int(indices[t + 1]),
                                      base + int(indices[t + 2])))
            else:
                for t in range(0, len(positions) - 2, 3):
                    all_faces.append((base + t, base + t + 1, base + t + 2))

    nodes = gltf.get("nodes", [])
    visited_nodes = set()

    def visit_node(node_index: int, parent_transform: Tuple[float, ...]) -> None:
        if node_index in visited_nodes or node_index < 0 or node_index >= len(nodes):
            return
        visited_nodes.add(node_index)

        node = nodes[node_index]
        world = mat_mul(parent_transform, mat_from_node(node))
        if "mesh" in node:
            append_mesh(int(node["mesh"]), world)
        for child in node.get("children", []):
            visit_node(int(child), world)

    scenes = gltf.get("scenes", [])
    scene_index = int(gltf.get("scene", 0))
    root_nodes = []
    if scenes and 0 <= scene_index < len(scenes):
        root_nodes = scenes[scene_index].get("nodes", [])

    if root_nodes and nodes:
        identity = mat_identity()
        for root in root_nodes:
            visit_node(int(root), identity)

    # Some simple GLBs omit scenes/nodes and store only meshes.
    if not all_verts:
        for mesh_index in range(len(gltf.get("meshes", []))):
            append_mesh(mesh_index, mat_identity())

    return all_verts, all_norms, all_faces


def extract_glb_mesh(path: Path) -> Tuple[List, List, List]:
    """Load a 3D model file and return (vertices, normals, faces) normalised to unit scale.

    For .glb files the built-in pure-Python parser is used (no external deps).
    For other formats (.obj, .fbx, etc.) trimesh is required.

    Vertices are centred at the origin; the longest axis spans [-0.5, 0.5],
    matching the scale convention of the built-in cube/sphere primitives.
    """
    suffix = path.suffix.lower()

    if suffix == ".glb":
        verts, norms, faces = _parse_glb(path)
    elif _HAS_TRIMESH:
        loaded = _trimesh.load(str(path), force="mesh", process=True)
        if isinstance(loaded, _trimesh.Scene):
            meshes = [g for g in loaded.dump() if isinstance(g, _trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"No triangle meshes found in {path}")
            loaded = _trimesh.util.concatenate(meshes)
        loaded.vertices -= loaded.bounds.mean(axis=0)
        ext = loaded.extents.max()
        if ext > 1e-6:
            loaded.vertices /= ext
        verts = [tuple(float(c) for c in v) for v in loaded.vertices]
        norms = [tuple(float(c) for c in n) for n in loaded.vertex_normals]
        faces = [tuple(int(i) for i in f) for f in loaded.faces]
        return verts, norms, faces
    else:
        raise RuntimeError(
            f"trimesh is required to load {suffix} files: pip install trimesh"
        )

    if not verts:
        raise ValueError(f"No geometry found in {path}")

    # Centre and normalise to [-0.5, 0.5] on the largest axis
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if extent < 1e-6:
        extent = 1.0

    verts = [((v[0] - cx) / extent, (v[1] - cy) / extent, (v[2] - cz) / extent)
             for v in verts]
    # Normals are direction vectors — normalise length
    def _nlen(n: tuple) -> float:
        l = math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        return l if l > 1e-8 else 1.0
    norms = [(n[0]/_nlen(n), n[1]/_nlen(n), n[2]/_nlen(n)) for n in norms]

    return verts, norms, faces


def load_glb_models(glb_dir: Path) -> List[GlbModel]:
    """Scan glb_dir (recursively) for .glb files and load their geometry.

    Uses the built-in pure-Python GLB parser — no trimesh required for .glb.
    Returns an empty list with a warning if the directory is missing or empty.
    """
    if not glb_dir.is_dir():
        print(f"WARNING: --glb-dir '{glb_dir}' does not exist — no GLB models loaded.",
              file=sys.stderr)
        return []

    glb_files = sorted(glb_dir.glob("**/*.glb"))
    if not glb_files:
        print(f"WARNING: no .glb files found in '{glb_dir}'.", file=sys.stderr)
        return []

    models: List[GlbModel] = []
    for p in glb_files:
        name = p.stem
        key  = _glb_key(name)
        cat, size, weight = _auto_categorize_glb(name)
        try:
            verts, norms, faces = extract_glb_mesh(p)
            models.append(GlbModel(
                name=name, path=p, category=cat, size=size,
                spawn_weight=weight, key=key,
                vertices=verts, normals=norms, faces=faces,
            ))
            print(f"  GLB: {name:32s}  {len(verts):5d} verts  {len(faces):5d} faces"
                  f"  [{cat}/{size}  w={weight}]")
        except Exception as exc:
            print(f"WARNING: could not load '{p}': {exc}", file=sys.stderr)

    return models


def select_glb_model(rng: random.Random, models: List[GlbModel]) -> GlbModel:
    """Weighted GLB model selection — mirrors Unity's SelectFromGLBCollection.

    Priority: (1) size distribution → (2) category weights → (3) spawn-weight fallback.
    """
    # 1. Try size-based selection
    r = rng.random()
    cumulative = 0.0
    chosen_size: Optional[str] = None
    for size, pct in _GLB_SIZE_DISTRIBUTION.items():
        cumulative += pct
        if r <= cumulative:
            chosen_size = size
            break
    if chosen_size is None:
        chosen_size = list(_GLB_SIZE_DISTRIBUTION.keys())[-1]

    size_pool = [m for m in models if m.size == chosen_size]
    if size_pool:
        return rng.choice(size_pool)

    # 2. Fallback: category-based selection
    total_cat = sum(_GLB_CATEGORY_WEIGHTS.values())
    r2 = rng.random() * total_cat
    acc = 0.0
    chosen_cat = list(_GLB_CATEGORY_WEIGHTS.keys())[0]
    for cat, w in _GLB_CATEGORY_WEIGHTS.items():
        acc += w
        if r2 <= acc:
            chosen_cat = cat
            break

    cat_pool = [m for m in models if m.category == chosen_cat]
    if cat_pool:
        return rng.choice(cat_pool)

    # 3. Final fallback: weighted random across all models
    total_w = sum(m.spawn_weight for m in models)
    r3 = rng.random() * total_w
    acc2 = 0.0
    for m in models:
        acc2 += m.spawn_weight
        if r3 <= acc2:
            return m
    return models[-1]


# ---------------------------------------------------------------------------
# Main instance builder
# ---------------------------------------------------------------------------

def select_mesh(rng: random.Random, args: argparse.Namespace,
                glb_models: Optional[List[GlbModel]] = None) -> str:
    """Return a mesh key string.

    When GLB models are available and the GLB weight roll succeeds, a GLB model
    key (e.g. 'glb_dodecahedron') is returned.  Otherwise a built-in primitive
    key ('cube' | 'sphere' | 'cylinder' | 'capsule') is returned.
    """
    glb_weight = getattr(args, "glb_model_weight", 0.0)
    if glb_models and rng.random() < glb_weight:
        return select_glb_model(rng, glb_models).key

    r = rng.random()
    if r < args.cube_probability:
        return "cube"
    r -= args.cube_probability
    if r < args.sphere_probability:
        return "sphere"
    r -= args.sphere_probability
    if r < args.cylinder_probability:
        return "cylinder"
    return "capsule"


def make_instances(args: argparse.Namespace,
                   glb_models: Optional[List[GlbModel]] = None):
    """Returns (instances, clusters, zones, colors).

    If glb_models is non-empty and --glb-model-weight > 0, scatter objects are
    chosen from the GLB library (with the same weighted-selection logic as
    Unity's GLBModelCollection) at the given probability; primitives fill the rest.
    """
    rng = random.Random(args.seed)
    bx, by, bz = args.bounds_x, args.bounds_y, args.bounds_z

    colors = make_random_colors(rng, args.color_count)
    instances: List[Instance] = []

    # --- Optional floor ---
    if args.fixed_floor:
        instances.append(Instance(
            name="floor", mesh="cube", color_idx=0,
            translation=(0.0, -0.06, 0.0),
            scaling=(bx, 0.12, bz),
            rotation=(0.0, 0.0, 0.0),
        ))

    # --- Boundary walls ---
    if args.boundary_walls:
        edge_h = min(2.5, by)
        et = 0.18
        for label, tr, sc in [
            ("north", (0.0, edge_h*0.5,  0.5*bz), (bx, edge_h, et)),
            ("south", (0.0, edge_h*0.5, -0.5*bz), (bx, edge_h, et)),
            ("east",  (0.5*bx, edge_h*0.5, 0.0),  (et, edge_h, bz)),
            ("west",  (-0.5*bx, edge_h*0.5, 0.0), (et, edge_h, bz)),
        ]:
            instances.append(Instance(
                name=f"boundary_{label}", mesh="cube", color_idx=1,
                translation=tr, scaling=sc, rotation=(0.0, 0.0, 0.0),
            ))

    # --- Interior wall segments ---
    make_wall_segments(rng, args, len(colors), instances)

    # --- Clusters ---
    clusters: List[GaussianCluster] = make_clusters(rng, args) if args.cluster_count > 0 else []

    # --- Boolean clearing zones ---
    zones = generate_boolean_zones(rng, args)

    # --- Scatter objects ---
    placed: List[Vec3] = []
    min_d2 = args.min_distance_between_objects ** 2
    attempts = 0

    while len(placed) < args.object_count and attempts < args.object_count * 15:
        attempts += 1
        p = generate_position(rng, args, clusters)

        # Min-distance check (XZ plane)
        if any((p[0]-q[0])**2 + (p[2]-q[2])**2 < min_d2 for q in placed):
            continue

        # Boolean zone rejection — tested before scale is known so use a generous
        # pre-scale radius; the exact radius is computed and re-checked after scaling.
        if any(point_in_zone(p, z) for z in zones):
            continue

        mesh = select_mesh(rng, args, glb_models)
        sc = generate_scale(rng, args)

        if args.ground_objects:
            # Set Y so object bottom sits near the floor; 12% chance of floating.
            # Unity-parity mode disables this because Unity's RuntimeSceneGenerator
            # instantiates objects directly in the 3D spawn volume.
            base_y = sc[1] * 0.5
            if rng.random() < 0.12:
                base_y += rng.uniform(0.4, max(0.5, by * 0.45))
            p = (p[0], base_y, p[2])

        # Boolean zone rejection — expand zone by the object's bounding radius so
        # large objects that merely straddle the zone boundary are also excluded.
        obj_radius = math.sqrt(sc[0]**2 + sc[1]**2 + sc[2]**2) * 0.5
        if any(point_in_zone(p, z, expand=obj_radius) for z in zones):
            continue

        if args.rotation_mode == "full":
            rot = (rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360))
        elif args.rotation_mode == "y_only":
            rot = (0.0, rng.uniform(0, 360), 0.0)
        else:
            rot = (rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 20))

        reserved_colors = 2 if (args.fixed_floor or args.boundary_walls or args.wall_count > 0) else 0
        cidx = rng.randint(reserved_colors, len(colors) - 1)

        instances.append(Instance(
            name=f"object_{len(placed):04d}",
            mesh=mesh, color_idx=cidx,
            translation=p, scaling=sc, rotation=rot,
        ))
        placed.append(p)

    # --- Environmental planes ---
    instances.extend(make_env_planes(rng, args, len(colors)))

    return instances, clusters, zones, colors


def compute_scene_bounds(instances: List[Instance],
                         args: argparse.Namespace,
                         include_planes: bool = False) -> SceneBounds:
    """Approximate Unity Renderer.bounds for camera placement.

    Unity's RuntimeSceneGenerator positions the camera from generated object
    renderer bounds and excludes environmental planes. Falcor's generator has
    normalized unit meshes, so the transformed unit-box AABB is a close proxy.
    """
    relevant = [
        inst for inst in instances
        if include_planes or not inst.name.startswith("env_plane_")
    ]
    if not relevant:
        size = (args.bounds_x, args.bounds_y, args.bounds_z)
        center = (0.0, 0.0, 0.0)
        return SceneBounds(
            center=center,
            size=size,
            min_corner=(-0.5 * size[0], -0.5 * size[1], -0.5 * size[2]),
            max_corner=(0.5 * size[0], 0.5 * size[1], 0.5 * size[2]),
        )

    mn = [float("inf"), float("inf"), float("inf")]
    mx = [float("-inf"), float("-inf"), float("-inf")]
    for inst in relevant:
        half = (inst.scaling[0] * 0.5, inst.scaling[1] * 0.5, inst.scaling[2] * 0.5)
        ext = euler_abs_extents(inst.rotation, half)
        for axis in range(3):
            mn[axis] = min(mn[axis], inst.translation[axis] - ext[axis])
            mx[axis] = max(mx[axis], inst.translation[axis] + ext[axis])

    # Unity calls Bounds.Expand(Vector3.one * 2f .magnitude). Bounds.Expand(float)
    # adds the value to the total size, so each extent grows by half that amount.
    safety_extent = math.sqrt(12.0) * 0.5
    for axis in range(3):
        mn[axis] -= safety_extent
        mx[axis] += safety_extent

    max_allowed_min = [-0.6 * args.bounds_x, -0.6 * args.bounds_y, -0.6 * args.bounds_z]
    max_allowed_max = [0.6 * args.bounds_x, 0.6 * args.bounds_y, 0.6 * args.bounds_z]
    for axis in range(3):
        mn[axis] = max(mn[axis], max_allowed_min[axis])
        mx[axis] = min(mx[axis], max_allowed_max[axis])

    size = (max(mx[0] - mn[0], 1e-3), max(mx[1] - mn[1], 1e-3), max(mx[2] - mn[2], 1e-3))
    center = ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
    return SceneBounds(center=center, size=size, min_corner=tuple(mn), max_corner=tuple(mx))


# ---------------------------------------------------------------------------
# Camera sampling  (orbit-based, matching Unity's PositionCameraRandomly)
# ---------------------------------------------------------------------------

def make_camera_samples(args: argparse.Namespace,
                        scene_bounds: Optional[SceneBounds] = None) -> List[CameraSample]:
    rng = random.Random(args.seed + 9173)
    if args.camera_use_scene_bounds and scene_bounds is not None:
        bx, by, bz = scene_bounds.size
        center = scene_bounds.center
        default_target = scene_bounds.center
    else:
        bx, by, bz = args.bounds_x, args.bounds_y, args.bounds_z
        center = (0.0, 0.0, 0.0)
        default_target = (0.0, by * 0.3, 0.0)

    fov_rad = math.radians(args.fov)
    h_fov_rad = 2.0 * math.atan(math.tan(fov_rad * 0.5) * args.camera_aspect_ratio)
    base_dist = max(
        (by * 0.6) / math.tan(fov_rad * 0.5),
        (max(bx, bz) * 0.6) / math.tan(h_fov_rad * 0.5),
    )

    samples: List[CameraSample] = []
    for _ in range(args.camera_count):
        dist = max(base_dist * rng.uniform(args.camera_distance_min, args.camera_distance_max), args.camera_min_distance)
        angle = rng.uniform(math.radians(args.camera_angle_min), math.radians(args.camera_angle_max))
        tilt  = math.radians(rng.uniform(args.camera_tilt_min, args.camera_tilt_max))
        h_off = rng.uniform(args.camera_height_min, args.camera_height_max) * by * 0.5

        hd = dist * math.cos(tilt)
        vd = dist * math.sin(tilt)
        px = center[0] + math.sin(angle) * hd
        pz = center[2] + math.cos(angle) * hd
        # Clamp Y above the floor so negative tilt + negative height_offset can't
        # bury the camera below the scene (minimum 0.5 m above the floor plane).
        py = center[1] + h_off + vd
        if args.clamp_camera_y:
            py = max(0.5, py)

        if args.randomize_camera_target:
            tx = center[0] + rng.uniform(-args.camera_target_offset_x, args.camera_target_offset_x)
            ty = center[1] + rng.uniform(-min(by*0.5, 3.0), min(by*0.5, 3.0))
            tz = center[2] + rng.uniform(-args.camera_target_offset_z, args.camera_target_offset_z)
        else:
            tx, ty, tz = default_target

        fov_s = rng.uniform(args.fov_min, args.fov_max) if args.randomize_camera_fov else args.fov
        forward = v3_normalize(v3_sub((tx, ty, tz), (px, py, pz)))
        samples.append(CameraSample(position=(px, py, pz), forward=forward, fov=fov_s))
    return samples


# ---------------------------------------------------------------------------
# .pyscene writer
# ---------------------------------------------------------------------------

def _emit_glb_builders(models_used: List[GlbModel]) -> List[str]:
    """Return pyscene lines that procedurally build one TriangleMesh per GLB model.

    The pattern mirrors the cylinder/capsule builders: a def function using
    TriangleMesh() + addVertex() + addTriangle(), then a module-level call.
    Variable name: {model.key}Mesh  (e.g. glb_dodecahedronMesh).
    """
    lines: List[str] = []
    for model in models_used:
        fname = f"_buildGlbMesh_{model.key[4:]}"   # strip "glb_" prefix for fn name
        vname = f"{model.key}Mesh"
        lines.append(f"# GLB model: {model.name}  ({model.category}/{model.size}"
                     f"  {len(model.vertices)} verts  {len(model.faces)} faces)")
        lines.append(f"def {fname}():")
        lines.append(f"    m = TriangleMesh()")
        lines.append(f"    _v = []")
        for (vx, vy, vz), (nx, ny, nz) in zip(model.vertices, model.normals):
            lines.append(
                f"    _v.append(m.addVertex("
                f"float3({vx:.5f},{vy:.5f},{vz:.5f}), "
                f"float3({nx:.5f},{ny:.5f},{nz:.5f}), "
                f"float2(0,0)))"
            )
        for i0, i1, i2 in model.faces:
            lines.append(f"    m.addTriangle(_v[{i0}], _v[{i1}], _v[{i2}])")
        lines.append(f"    return m")
        lines.append(f"{vname} = {fname}()")
        lines.append("")
    return lines


def write_scene(path: Path, instances: List[Instance],
                colors: List[Tuple[float, float, float, float]],
                first_camera: CameraSample,
                glb_models: Optional[List[GlbModel]] = None) -> None:
    lines: List[str] = [
        "# Generated by generate_synthetic_neuralpvs_scene.py",
        "# Load this file in the NeuralPVSExporter sample.",
        "",
        "# --- Mesh primitives ---",
        "import math as _math",
        "",
        "cubeMesh   = TriangleMesh.createCube()",
        "sphereMesh = TriangleMesh.createSphere()",
        "",
        "# Procedural cylinder (radius=0.5, half-height=0.5, segs=16)",
        "def _createCylinder(segs=16):",
        "    m  = TriangleMesh()",
        "    r  = 0.5",
        "    hh = 0.5   # half-height",
        "    bot_center = m.addVertex(float3(0, -hh, 0), float3(0,-1,0), float2(0.5,0.5))",
        "    top_center = m.addVertex(float3(0,  hh, 0), float3(0, 1,0), float2(0.5,0.5))",
        "    bot_ring = []",
        "    top_ring = []",
        "    side_bot = []",
        "    side_top = []",
        "    for i in range(segs):",
        "        a = 2*_math.pi*i/segs",
        "        c, s = _math.cos(a), _math.sin(a)",
        "        u = 0.5 + 0.5*c",
        "        v = 0.5 + 0.5*s",
        "        bot_ring.append(m.addVertex(float3(r*c,-hh,r*s), float3(0,-1,0), float2(u,v)))",
        "        top_ring.append(m.addVertex(float3(r*c, hh,r*s), float3(0, 1,0), float2(u,v)))",
        "        n = float3(c, 0, s)",
        "        uv = float2(i/segs, 0)",
        "        side_bot.append(m.addVertex(float3(r*c,-hh,r*s), n, float2(i/segs, 0)))",
        "        side_top.append(m.addVertex(float3(r*c, hh,r*s), n, float2(i/segs, 1)))",
        "    for i in range(segs):",
        "        ni = (i+1)%segs",
        "        m.addTriangle(bot_center, bot_ring[ni], bot_ring[i])",
        "        m.addTriangle(top_center, top_ring[i],  top_ring[ni])",
        "        sb0,sb1 = side_bot[i], side_bot[ni]",
        "        st0,st1 = side_top[i], side_top[ni]",
        "        m.addTriangle(sb0, st0, sb1)",
        "        m.addTriangle(sb1, st0, st1)",
        "    return m",
        "cylinderMesh = _createCylinder()",
        "",
        "# Procedural capsule (radius=0.5, cylinder half-height=0.25, segs=16, rings=8)",
        "def _createCapsule(segs=16, rings=8):",
        "    m  = TriangleMesh()",
        "    r  = 0.5",
        "    cy = 0.25   # cylinder half-height",
        "    # --- bottom hemisphere (flipped, centered at -cy) ---",
        "    bot_layers = []",
        "    for ri in range(rings+1):",
        "        phi = _math.pi*0.5 * ri / rings   # 0 → pi/2",
        "        y   = -cy - r*_math.cos(phi)",
        "        cr  = r*_math.sin(phi)",
        "        ring = []",
        "        for si in range(segs):",
        "            a = 2*_math.pi*si/segs",
        "            c, s = _math.cos(a), _math.sin(a)",
        "            nx,ny,nz = c*_math.sin(phi), -_math.cos(phi), s*_math.sin(phi)",
        "            ring.append(m.addVertex(float3(cr*c, y, cr*s), float3(nx,ny,nz), float2(si/segs, ri/rings)))",
        "        bot_layers.append(ring)",
        "    # --- top hemisphere (centered at +cy) ---",
        "    top_layers = []",
        "    for ri in range(rings+1):",
        "        phi = _math.pi*0.5 * ri / rings   # 0 → pi/2",
        "        y   =  cy + r*_math.cos(_math.pi*0.5 - phi)",
        "        cr  = r*_math.sin(_math.pi*0.5 - phi)",
        "        ring = []",
        "        for si in range(segs):",
        "            a = 2*_math.pi*si/segs",
        "            c, s = _math.cos(a), _math.sin(a)",
        "            nx,ny,nz = c*_math.cos(phi), _math.sin(phi), s*_math.cos(phi)",
        "            ring.append(m.addVertex(float3(cr*c, y, cr*s), float3(nx,ny,nz), float2(si/segs, ri/rings)))",
        "        top_layers.append(ring)",
        "    def stitch(layA, layB):",
        "        for si in range(segs):",
        "            ni = (si+1)%segs",
        "            m.addTriangle(layA[si], layB[si],  layA[ni])",
        "            m.addTriangle(layA[ni], layB[si],  layB[ni])",
        "    for li in range(rings):",
        "        stitch(bot_layers[li], bot_layers[li+1])",
        "        stitch(top_layers[li], top_layers[li+1])",
        "    # stitch the two hemispheres together at the equator",
        "    stitch(bot_layers[-1], top_layers[0])",
        "    return m",
        "capsuleMesh = _createCapsule()",
        "",
    ]

    # Emit procedural builders for every GLB model that actually appears in instances
    if glb_models:
        glb_model_map: Dict[str, GlbModel] = {m.key: m for m in glb_models}
        used_glb_keys = sorted({inst.mesh for inst in instances if inst.mesh.startswith("glb_")})
        used_glb_models = [glb_model_map[k] for k in used_glb_keys if k in glb_model_map]
        if used_glb_models:
            lines += _emit_glb_builders(used_glb_models)

    lines.append("# --- Per-instance random colour materials ---")

    for idx, (r, g, b, a) in enumerate(colors):
        lines += [
            f"mat_{idx:04d} = StandardMaterial('color_{idx:04d}')",
            f"mat_{idx:04d}.baseColor = {f4((r, g, b, a))}",
            f"mat_{idx:04d}.roughness = 0.75",
            f"mat_{idx:04d}.metallic  = 0.0",
            "",
        ]

    # MeshID table — one entry per unique (mesh, color_idx) combination
    lines.append("# --- MeshID table ---")
    needed = sorted({(inst.mesh, inst.color_idx) for inst in instances})
    for mesh, cidx in needed:
        lines.append(
            f"meshID_{mesh}_{cidx:04d} = sceneBuilder.addTriangleMesh({mesh}Mesh, mat_{cidx:04d})"
        )
    lines.append("")

    # Instance nodes
    lines.append("# --- Scene instances ---")
    for inst in instances:
        node = inst.name.replace("-", "_")
        lines += [
            f"{node}_nID = sceneBuilder.addNode(",
            f"    '{node}',",
            f"    Transform(scaling={f3(inst.scaling)},",
            f"              translation={f3(inst.translation)},",
            f"              rotationEulerDeg={f3(inst.rotation)}))",
            f"sceneBuilder.addMeshInstance({node}_nID, meshID_{inst.mesh}_{inst.color_idx:04d})",
            "",
        ]

    # Camera
    pos = first_camera.position
    target = v3_add(pos, first_camera.forward)
    lines += [
        "# --- Camera ---",
        "camera = Camera('SyntheticCamera0')",
        f"camera.position = {f3(pos)}",
        f"camera.target   = {f3(target)}",
        "camera.up = float3(0.0, 1.0, 0.0)",
        "camera.focalLength = 35.0",
        "sceneBuilder.addCamera(camera)",
        "sceneBuilder.selectedCamera = camera",
        "",
        "# --- Lighting ---",
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
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Camera-path CSV
# ---------------------------------------------------------------------------

def write_camera_path(path: Path, samples: List[CameraSample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["x", "y", "z", "forward_x", "forward_y", "forward_z", "fov"])
        for s in samples:
            w.writerow([f"{s.position[0]:.6f}", f"{s.position[1]:.6f}", f"{s.position[2]:.6f}",
                        f"{s.forward[0]:.6f}",  f"{s.forward[1]:.6f}",  f"{s.forward[2]:.6f}",
                        f"{s.fov:.4f}"])


# ---------------------------------------------------------------------------
# Manifest JSON
# ---------------------------------------------------------------------------

def write_manifest(path: Path, args: argparse.Namespace, scene_path: Path, csv_path: Path,
                   instances: List[Instance], clusters: List[GaussianCluster],
                   zones: List[BooleanZone],
                   glb_models: Optional[List[GlbModel]] = None,
                   scene_bounds: Optional[SceneBounds] = None) -> None:
    mesh_counts: dict = {}
    for inst in instances:
        mesh_counts[inst.mesh] = mesh_counts.get(inst.mesh, 0) + 1

    manifest = {
        "name": args.name,
        "seed": args.seed,
        "scene_path": str(scene_path),
        "camera_path_csv": str(csv_path),
        "object_count": args.object_count,
        "wall_count": args.wall_count,
        "camera_count": args.camera_count,
        "bounds": [args.bounds_x, args.bounds_y, args.bounds_z],
        "unity_parity": args.unity_parity,
        "fixed_floor": args.fixed_floor,
        "boundary_walls": args.boundary_walls,
        "ground_objects": args.ground_objects,
        "centered_y_distribution": args.centered_y_distribution,
        "rotation_mode": args.rotation_mode,
        "generated_instance_count": len(instances),
        "mesh_counts": mesh_counts,
        "color_count": args.color_count,
        "cluster_count": len(clusters),
        "boolean_zone_count": len(zones),
        "boolean_zones": [
            {"shape": z.shape, "position": list(z.position),
             "half_extents": list(z.half_extents), "rotation_y_deg": z.rotation_y}
            for z in zones
        ],
        "scaling_mode": args.scaling_mode,
        "computed_scene_bounds": None if scene_bounds is None else {
            "center": list(scene_bounds.center),
            "size": list(scene_bounds.size),
            "min": list(scene_bounds.min_corner),
            "max": list(scene_bounds.max_corner),
        },
        "suggested_neuralpvs_exporter_settings": {
            "mode": "Generate GV + PVV",
            "scene_path": str(scene_path),
            "dataset_name": args.name,
            "sampling_mode": "Path CSV",
            "camera_path_csv": str(csv_path),
            "visibility_mode": "Ray-tested view cell",
            "volume_mapping": "World AABB volume",
            "volume_size": args.volume_size,
            "volume_depth": args.volume_depth,
            "camera_aspect_ratio": args.camera_aspect_ratio,
            "view_cell_radius": args.view_cell_radius,
            "view_cell_radius_m": args.view_cell_radius,
            "view_cell_radius_cm": args.view_cell_radius_cm,
            "view_cell_near": args.near,
            "view_cell_far": args.far,
            "pvv_sample_steps": args.pvv_sample_steps,
            "linear_z": args.linear_z,
            "log_depth_scale": args.log_depth_scale,
            "unity_fov_expansion_degrees": args.unity_fov_expansion_degrees,
        },
        "glb_models": [
            {"name": m.name, "key": m.key, "category": m.category,
             "size": m.size, "spawn_weight": m.spawn_weight,
             "vertex_count": len(m.vertices), "face_count": len(m.faces),
             "source": str(m.path)}
            for m in (glb_models or [])
        ],
        "glb_model_weight": getattr(args, "glb_model_weight", 0.0) if glb_models else 0.0,
        "notes": [
            "Scene contains cubes, spheres, cylinders, and capsules with per-object random colours.",
            "Boolean clearing zones (tunnels/roads/cavities) filtered object placement.",
            "Advanced non-spherical Gaussian clustering was used for object distribution.",
            "Use the CSV as the exporter path CSV so every row becomes a viewcell/camera sample.",
        ] + (
            [f"GLB model library: {len(glb_models)} model(s) loaded from '{args.glb_dir}'."]
            if glb_models else []
        ),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def provided_flags(argv: List[str]) -> set:
    return {a.split("=", 1)[0] for a in argv if a.startswith("--")}


def flag_provided(flags: set, *names: str) -> bool:
    return any(name in flags for name in names)


def set_unless_provided(args: argparse.Namespace, flags: set, attr: str, value, *names: str) -> None:
    if not flag_provided(flags, *names):
        setattr(args, attr, value)


def find_default_glb_dir(out_dir: Path) -> Optional[Path]:
    candidates = [
        out_dir / "models",
        out_dir.parent / "models",
        Path.cwd() / "media" / "SyntheticNeuralPVS" / "models",
        Path.cwd() / "media" / "UnitySyntheticModels",
        Path("T:/NeuralPVS_LiveDemo/Assets/models"),
        Path("H:/NeuralPVS_LiveDemo/Assets/models"),
        Path("T:/Falcor/media/SyntheticNeuralPVS/models"),
        Path("C:/dev/Falcor/media/SyntheticNeuralPVS/models"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("**/*.glb")):
            return candidate
    return None


def apply_unity_parity_preset(args: argparse.Namespace) -> None:
    if not args.unity_parity:
        return

    flags = args._provided_flags
    set_unless_provided(args, flags, "bounds_x", 30.0, "--bounds-x")
    set_unless_provided(args, flags, "bounds_y", 25.0, "--bounds-y")
    set_unless_provided(args, flags, "bounds_z", 30.0, "--bounds-z")
    set_unless_provided(args, flags, "min_distance_between_objects", 0.8, "--min-distance-between-objects")

    if not flag_provided(flags, "--object-count", "--objects"):
        lo = args.object_count_min if args.object_count_min is not None else 50
        hi = args.object_count_max if args.object_count_max is not None else 150
        if hi < lo:
            hi = lo
        args.object_count = random.Random(args.seed + 31).randint(lo, hi)

    set_unless_provided(args, flags, "wall_count", 0, "--wall-count")
    set_unless_provided(args, flags, "fixed_floor", False, "--fixed-floor", "--no-fixed-floor")
    set_unless_provided(args, flags, "boundary_walls", False, "--boundary-walls", "--no-boundary-walls")
    set_unless_provided(args, flags, "boolean_count", 0, "--boolean-count")
    set_unless_provided(args, flags, "max_plane_count", 3, "--max-plane-count")

    set_unless_provided(args, flags, "cluster_count", 3, "--cluster-count")
    set_unless_provided(args, flags, "clustering_intensity", 0.3, "--clustering-intensity")
    set_unless_provided(args, flags, "cluster_shape_variation", 0.4, "--cluster-shape-variation")
    set_unless_provided(args, flags, "cluster_size_range", [0.5, 2.5], "--cluster-size-range")
    set_unless_provided(args, flags, "cluster_density_range", [0.3, 2.0], "--cluster-density-range")
    set_unless_provided(args, flags, "cluster_position_noise", 0.2, "--cluster-position-noise")
    set_unless_provided(args, flags, "cluster_rotation_variation", 0.3, "--cluster-rotation-variation")

    set_unless_provided(args, flags, "scaling_mode", "uniform", "--scaling-mode")
    set_unless_provided(args, flags, "uniform_scale_min", 0.5, "--uniform-scale-min")
    set_unless_provided(args, flags, "uniform_scale_max", 2.0, "--uniform-scale-max")
    set_unless_provided(args, flags, "extreme_scaling_probability", 0.2, "--extreme-scaling-probability")
    set_unless_provided(args, flags, "max_aspect_ratio", 3.0, "--max-aspect-ratio")

    set_unless_provided(args, flags, "ground_objects", False, "--ground-objects", "--no-ground-objects")
    set_unless_provided(args, flags, "centered_y_distribution", True, "--centered-y-distribution", "--no-centered-y-distribution")
    set_unless_provided(args, flags, "rotation_mode", "full", "--rotation-mode")
    set_unless_provided(args, flags, "use_glb_models", True, "--use-glb-models", "--no-use-glb-models")
    set_unless_provided(args, flags, "glb_model_weight", 1.0, "--glb-model-weight")

    set_unless_provided(args, flags, "randomize_camera_target", False, "--randomize-camera-target", "--no-randomize-camera-target")
    set_unless_provided(args, flags, "camera_use_scene_bounds", True, "--camera-use-scene-bounds", "--no-camera-use-scene-bounds")
    set_unless_provided(args, flags, "clamp_camera_y", False, "--clamp-camera-y", "--no-clamp-camera-y")
    set_unless_provided(args, flags, "camera_min_distance", 5.0, "--camera-min-distance")

    if args.use_glb_models and args.glb_dir is None:
        args.glb_dir = find_default_glb_dir(args.out_dir)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    flags = provided_flags(argv)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Basics
    p.add_argument("--out-dir",  type=Path, required=True)
    p.add_argument("--name",     default="synthetic_neuralpvs_scene")
    p.add_argument("--seed",     type=int, default=1)
    p.add_argument("--unity-parity", action="store_true",
                   help="Apply defaults matching Unity RuntimeSceneGenerator: GLB models, "
                        "50-150 objects, centered 3D placement, scene-bounds camera, "
                        "no Falcor floor/walls/boolean clearing zones.")

    # Scene contents
    p.add_argument("--object-count", "--objects", dest="object_count", type=int, default=180)
    p.add_argument("--object-count-min", type=int, default=None,
                   help="Unity-parity object-count lower bound when --object-count is not provided.")
    p.add_argument("--object-count-max", type=int, default=None,
                   help="Unity-parity object-count upper bound when --object-count is not provided.")
    p.add_argument("--wall-count",   type=int,   default=14)
    p.add_argument("--bounds-x",     type=float, default=18.0)
    p.add_argument("--bounds-y",     type=float, default=7.0)
    p.add_argument("--bounds-z",     type=float, default=18.0)
    p.add_argument("--min-distance-between-objects", type=float, default=0.5)
    p.add_argument("--fixed-floor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--boundary-walls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ground-objects", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--centered-y-distribution", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rotation-mode", choices=["mostly_upright", "full", "y_only"], default="mostly_upright")

    # GLB model library (mirrors Unity's GLBModelCollection)
    p.add_argument("--use-glb-models",   action=argparse.BooleanOptionalAction, default=False,
                   help="Enable GLB model library alongside (or instead of) built-in primitives. "
                        "GLB files use the built-in parser; non-GLB formats require trimesh.")
    p.add_argument("--glb-dir",          type=Path, default=None,
                   help="Directory to scan recursively for .glb files (required when --use-glb-models)")
    p.add_argument("--glb-model-weight", type=float, default=1.0,
                   help="Probability [0,1] that each scatter object is chosen from the GLB library "
                        "rather than built-in primitives. Default 1.0 uses GLB models for all scatter "
                        "objects when a library is enabled. Set lower to mix GLB with primitives.")

    # Mesh-type probabilities (capsule = remainder up to 1.0)
    p.add_argument("--cube-probability",     type=float, default=0.40)
    p.add_argument("--sphere-probability",   type=float, default=0.30)
    p.add_argument("--cylinder-probability", type=float, default=0.20)
    # capsule probability = 1 - cube - sphere - cylinder = 0.10 by default

    # Colours
    p.add_argument("--color-count", type=int, default=64,
                   help="Number of unique random colours (>= 4)")

    # Clustering
    p.add_argument("--cluster-count",             type=int,   default=6)
    p.add_argument("--clustering-intensity",       type=float, default=0.55)
    p.add_argument("--cluster-shape-variation",    type=float, default=0.4,
                   help="0=spherical, 1=highly elliptical")
    p.add_argument("--cluster-size-range",         type=float, nargs=2, default=[0.5, 2.5],
                   metavar=("MIN", "MAX"))
    p.add_argument("--cluster-density-range",      type=float, nargs=2, default=[0.3, 2.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--cluster-position-noise",     type=float, default=0.2)
    p.add_argument("--cluster-rotation-variation", type=float, default=0.3)

    # Scaling
    p.add_argument("--scaling-mode",             choices=["uniform", "nonuniform"], default="nonuniform")
    p.add_argument("--uniform-scale-min",        type=float, default=0.25)
    p.add_argument("--uniform-scale-max",        type=float, default=1.25)
    p.add_argument("--scale-min-x",              type=float, default=0.20)
    p.add_argument("--scale-max-x",              type=float, default=1.50)
    p.add_argument("--scale-min-y",              type=float, default=0.20)
    p.add_argument("--scale-max-y",              type=float, default=2.00)
    p.add_argument("--scale-min-z",              type=float, default=0.20)
    p.add_argument("--scale-max-z",              type=float, default=1.50)
    p.add_argument("--scaling-bias",             type=float, default=0.5,
                   help="0.5=uniform; <0.5=smaller bias; >0.5=larger bias")
    p.add_argument("--extreme-scaling-probability", type=float, default=0.15)
    p.add_argument("--extreme-scale-min",        type=float, default=0.10)
    p.add_argument("--extreme-scale-max",        type=float, default=4.00)
    p.add_argument("--maintain-aspect-ratio",    action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-aspect-ratio",         type=float, default=4.0)
    p.add_argument("--global-scale-variation",   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--global-scale-min",         type=float, default=0.5)
    p.add_argument("--global-scale-max",         type=float, default=1.5)
    p.add_argument("--uniform-global-scaling",   action=argparse.BooleanOptionalAction, default=True)

    # Environmental planes
    p.add_argument("--max-plane-count", type=int, default=4)

    # Boolean clearing zones
    p.add_argument("--boolean-count",             type=int,   default=3)
    p.add_argument("--boolean-size-ratio-min",    type=float, default=0.10)
    p.add_argument("--boolean-size-ratio-max",    type=float, default=0.40)
    p.add_argument("--boolean-shape-weights",     type=float, nargs=4,
                   default=[0.3, 0.3, 0.3, 0.1],
                   metavar=("SPHERE", "BOX", "CYLINDER", "CAPSULE"))
    p.add_argument("--enable-tunnels",            action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tunnel-probability",        type=float, default=0.50)
    p.add_argument("--enable-roads",              action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--road-probability",          type=float, default=0.30)
    p.add_argument("--boolean-boundary-buffer",   type=float, default=0.10)
    p.add_argument("--boolean-extend-beyond-bounds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--boolean-extension-factor",  type=float, default=1.5)
    p.add_argument("--boolean-random-rotation",   action=argparse.BooleanOptionalAction, default=True)

    # Camera sampling
    p.add_argument("--camera-count", "--samples", dest="camera_count", type=int, default=128)
    p.add_argument("--fov",                   type=float, default=60.0)
    p.add_argument("--camera-aspect-ratio",   type=float, default=1.777778)
    p.add_argument("--camera-distance-min",   type=float, default=0.8)
    p.add_argument("--camera-distance-max",   type=float, default=1.5)
    p.add_argument("--camera-min-distance",   type=float, default=3.0)
    p.add_argument("--camera-height-min",     type=float, default=-0.5)
    p.add_argument("--camera-height-max",     type=float, default=0.8)
    p.add_argument("--camera-angle-min",      type=float, default=0.0,   help="Degrees")
    p.add_argument("--camera-angle-max",      type=float, default=360.0, help="Degrees")
    p.add_argument("--camera-tilt-min",       type=float, default=-30.0, help="Degrees")
    p.add_argument("--camera-tilt-max",       type=float, default=45.0,  help="Degrees")
    p.add_argument("--randomize-camera-target",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--camera-target-offset-x",    type=float, default=2.0)
    p.add_argument("--camera-target-offset-z",    type=float, default=2.0)
    p.add_argument("--camera-use-scene-bounds",   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--clamp-camera-y",            action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--randomize-camera-fov",      action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fov-min",               type=float, default=45.0)
    p.add_argument("--fov-max",               type=float, default=75.0)

    # NeuralPVS exporter defaults (written to manifest only). The paper's
    # r30/r60/r90 notation is centimeters, i.e. 0.3/0.6/0.9 scene units.
    p.add_argument("--view-cell-radius",            type=float, default=None,
                   help="View-cell radius in meters/scene units. Use 0.3 for r30.")
    p.add_argument("--view-cell-radius-cm", "--radius", dest="view_cell_radius_cm", type=float, default=None,
                   help="View-cell radius in centimeters. Use 30, 60, or 90 for paper-style r30/r60/r90.")
    p.add_argument("--near",                        type=float, default=0.3)
    p.add_argument("--far",                         type=float, default=30.0)
    p.add_argument("--pvv-sample-steps",            type=int,   default=10)
    p.add_argument("--linear-z", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-depth-scale",             type=float, default=0.01)
    p.add_argument("--unity-fov-expansion-degrees", type=float, default=30.0)
    p.add_argument("--volume-size",  type=int, default=256)
    p.add_argument("--volume-depth", type=int, default=256)

    args = p.parse_args(argv)
    args._provided_flags = flags
    apply_unity_parity_preset(args)
    normalize_view_cell_radius(args)
    return args


def infer_view_cell_radius_cm_from_name(name: str) -> Optional[float]:
    match = re.search(r"(?:^|[_-])r(30|60|90)(?:$|[_-])", name)
    return float(match.group(1)) if match else None


def normalize_view_cell_radius(args: argparse.Namespace) -> None:
    inferred_cm = infer_view_cell_radius_cm_from_name(args.name)
    if args.view_cell_radius is not None and args.view_cell_radius_cm is not None:
        raise ValueError("Use either --view-cell-radius or --view-cell-radius-cm, not both.")

    if args.view_cell_radius_cm is not None:
        args.view_cell_radius = args.view_cell_radius_cm / 100.0
    elif args.view_cell_radius is None:
        args.view_cell_radius = (inferred_cm / 100.0) if inferred_cm is not None else 0.3
    elif args.view_cell_radius > 10.0:
        print(
            "WARNING: --view-cell-radius looks like centimeters. "
            "Interpreting it as cm; prefer --view-cell-radius-cm.",
            file=sys.stderr,
        )
        args.view_cell_radius /= 100.0

    args.view_cell_radius_cm = args.view_cell_radius * 100.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.object_count < 0 or args.wall_count < 0 or args.camera_count < 1:
        raise ValueError("object-count/wall-count must be >= 0; camera-count must be >= 1.")
    if args.color_count < 4:
        raise ValueError("--color-count must be at least 4.")
    total_mesh_prob = args.cube_probability + args.sphere_probability + args.cylinder_probability
    if total_mesh_prob > 1.0 + 1e-6:
        raise ValueError("cube + sphere + cylinder probabilities must sum to <= 1.0")

    # --- Load GLB models (optional) ---
    glb_models: List[GlbModel] = []
    if args.use_glb_models:
        if args.glb_dir is None:
            raise ValueError("--glb-dir must be specified when --use-glb-models is set.")
        print(f"Loading GLB models from: {args.glb_dir}")
        glb_models = load_glb_models(args.glb_dir)
        if glb_models:
            print(f"  {len(glb_models)} model(s) loaded "
                  f"(glb-model-weight={args.glb_model_weight:.2f})")
        else:
            print("WARNING: no GLB models loaded — falling back to primitives only.",
                  file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene_path    = args.out_dir / f"{args.name}.pyscene"
    csv_path      = args.out_dir / f"{args.name}_camera_path.csv"
    manifest_path = args.out_dir / f"{args.name}_manifest.json"

    instances, clusters, zones, colors = make_instances(args, glb_models)
    scene_bounds = compute_scene_bounds(instances, args, include_planes=False)
    cameras = make_camera_samples(args, scene_bounds)

    write_scene(scene_path, instances, colors, cameras[0], glb_models)
    write_camera_path(csv_path, cameras)
    write_manifest(manifest_path, args, scene_path, csv_path, instances, clusters, zones,
                   glb_models, scene_bounds)

    mesh_counts: dict = {}
    for inst in instances:
        mesh_counts[inst.mesh] = mesh_counts.get(inst.mesh, 0) + 1

    print(f"Wrote scene   : {scene_path}")
    print(f"Wrote CSV     : {csv_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Instances     : {len(instances)}  {mesh_counts}")
    print(f"Colours       : {args.color_count} unique random colours")
    print(f"Clusters      : {len(clusters)} (advanced non-spherical Gaussian)")
    print(f"Boolean zones : {len(zones)}")
    print(f"Camera samples: {len(cameras)}")
    print(f"Scene bounds  : center={scene_bounds.center}, size={scene_bounds.size}")
    print(f"View-cell r   : {args.view_cell_radius:.3f} m ({args.view_cell_radius_cm:.0f} cm)")
    if glb_models:
        print(f"GLB models    : {len(glb_models)} loaded, weight={args.glb_model_weight:.2f}")


if __name__ == "__main__":
    main()
