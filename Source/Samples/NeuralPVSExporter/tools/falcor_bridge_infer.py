#!/usr/bin/env python3
"""Predict Falcor NeuralPVS PVV files from exported GV files.

The Falcor exporter writes Unity-style GV/PVV datasets. This helper mirrors the
Unity bridge at the process boundary: it imports the adapted NeuralPVS Python
backend, loads one checkpoint, runs inference for each exported GV sample, and
writes files into the dataset's predicted_pvv directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuralpvs-root", required=True)
    parser.add_argument("--dataset-root", required=True, help="Falcor dataset folder containing metadata.json and gv/.")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint path.")
    parser.add_argument("--out-dir", default=None, help="Prediction folder. Defaults to <dataset-root>/predicted_pvv.")
    parser.add_argument("--model", default="OACNNsInterleaved")
    parser.add_argument("--backend", default="fvdb")
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--model-depth", type=int, default=2)
    parser.add_argument("--interleaver-r", type=int, default=2)
    parser.add_argument("--z-size", type=int, default=None, help="Defaults to metadata volume depth.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-pool-size", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_metadata(dataset_root: Path) -> dict:
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_gv_path(dataset_root: Path, sample: dict, index: int) -> Path:
    gv_file = sample.get("gv_file")
    if gv_file:
        return dataset_root / gv_file
    return dataset_root / "gv" / f"{index:04d}_gv.bin.gz"


def main() -> int:
    args = parse_args()
    neuralpvs_root = Path(args.neuralpvs_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else dataset_root / "predicted_pvv"

    if not neuralpvs_root.exists():
        raise FileNotFoundError(f"NeuralPVS root not found: {neuralpvs_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    metadata = load_metadata(dataset_root)
    samples = metadata.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in {dataset_root / 'metadata.json'}")

    volume_size = metadata.get("volume_size", [256, 256, 256])
    z_size = args.z_size if args.z_size is not None else int(volume_size[2])

    sys.path.insert(0, str(neuralpvs_root))

    import torch
    import torch.nn.functional as F

    from modules.dataset import load_volume
    from utils.init import init_model
    from utils.tensor import to_dense, to_sparse
    from utils.train import restore_checkpoint, save_volume

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_args = argparse.Namespace(interleaver_r=args.interleaver_r, dice_alpha=0.1)
    model = init_model(
        args.model,
        args.backend,
        args.in_channels,
        args.classes,
        args.model_depth,
        model_args,
    ).to(device)
    restore_checkpoint(model, str(checkpoint_path))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_samples = samples[: args.limit] if args.limit else samples

    for sample in selected_samples:
        index = int(sample.get("index", len(selected_samples)))
        gv_path = resolve_gv_path(dataset_root, sample, index)
        out_path = out_dir / f"{index}_predicted_pvv.bin.gz"

        if out_path.exists() and not args.overwrite:
            print(f"skip_existing={out_path}")
            continue
        if not gv_path.exists():
            raise FileNotFoundError(f"GV file not found for sample {index}: {gv_path}")

        gv = load_volume(str(gv_path), amp=False, z_size=z_size, cupy=False)
        x = torch.from_numpy(gv).unsqueeze(0).to(device)
        model_input = x if args.model.startswith("OACNNs") else to_sparse(x, args.backend)

        with torch.no_grad():
            y = model(model_input)
            y_dense = to_dense(y, x.shape).sigmoid()

            if args.max_pool_size and args.max_pool_size > 0:
                kernel = args.max_pool_size
                y_dense = F.max_pool3d(y_dense, kernel_size=kernel, stride=1, padding=kernel // 2)

            pred = (y_dense > args.threshold).int()

        save_volume(pred.cpu(), str(out_path))
        print(f"wrote={out_path} active_voxels={int(pred.sum().item())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
