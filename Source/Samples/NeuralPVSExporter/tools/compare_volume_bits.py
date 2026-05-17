#!/usr/bin/env python3
"""Compare NeuralPVS packed GV/PVV volume files at byte and bit level."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def read_volume(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def bit_count(data: bytes) -> int:
    return sum(byte.bit_count() for byte in data)


def compare_pair(reference: Path, candidate: Path) -> dict[str, object]:
    ref = read_volume(reference)
    cand = read_volume(candidate)

    if len(ref) != len(cand):
        raise ValueError(f"Size mismatch: {reference} has {len(ref)} bytes, {candidate} has {len(cand)} bytes")

    tp = fp = fn = tn = differing_bytes = 0
    for rb, cb in zip(ref, cand):
        if rb != cb:
            differing_bytes += 1
        tp += (rb & cb).bit_count()
        fp += ((~rb) & cb & 0xFF).bit_count()
        fn += (rb & (~cb) & 0xFF).bit_count()
        tn += ((~rb) & (~cb) & 0xFF).bit_count()

    union = tp + fp + fn
    positives = tp + fn
    negatives = fp + tn

    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "bytes": len(ref),
        "exact": ref == cand,
        "differing_bytes": differing_bytes,
        "ref_bits": bit_count(ref),
        "candidate_bits": bit_count(cand),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_rate": fp / negatives if negatives else 0.0,
        "fn_rate": fn / positives if positives else 0.0,
        "iou": tp / union if union else 1.0,
    }


def collect_pairs(reference: Path, candidate: Path, pattern: str) -> list[tuple[Path, Path]]:
    if reference.is_file() and candidate.is_file():
        return [(reference, candidate)]

    if not reference.is_dir() or not candidate.is_dir():
        raise ValueError("Reference and candidate must both be files or both be directories.")

    ref_files = sorted(reference.glob(pattern))
    cand_files = sorted(candidate.glob(pattern))
    if len(ref_files) != len(cand_files):
        raise ValueError(
            f"File-count mismatch for pattern {pattern!r}: reference has {len(ref_files)}, candidate has {len(cand_files)}"
        )

    return list(zip(ref_files, cand_files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Unity/reference volume file or directory.")
    parser.add_argument("--candidate", required=True, type=Path, help="Falcor/candidate volume file or directory.")
    parser.add_argument("--glob", default="*.bin.gz", help="Glob used when comparing directories.")
    parser.add_argument("--csv", type=Path, help="Optional CSV summary output path.")
    args = parser.parse_args()

    rows = [compare_pair(ref, cand) for ref, cand in collect_pairs(args.reference, args.candidate, args.glob)]

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    total_tp = sum(int(row["tp"]) for row in rows)
    total_fp = sum(int(row["fp"]) for row in rows)
    total_fn = sum(int(row["fn"]) for row in rows)
    total_tn = sum(int(row["tn"]) for row in rows)
    exact_count = sum(1 for row in rows if row["exact"])
    union = total_tp + total_fp + total_fn
    positives = total_tp + total_fn
    negatives = total_fp + total_tn

    print(f"Compared {len(rows)} file pair(s).")
    print(f"Exact matches: {exact_count}/{len(rows)}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}")
    print(f"fp_rate={total_fp / negatives if negatives else 0.0:.10f}")
    print(f"fn_rate={total_fn / positives if positives else 0.0:.10f}")
    print(f"IoU={total_tp / union if union else 1.0:.10f}")

    if rows:
        worst_fn = max(rows, key=lambda row: float(row["fn_rate"]))
        worst_fp = max(rows, key=lambda row: float(row["fp_rate"]))
        print(f"Worst fn_rate: {worst_fn['fn_rate']:.10f} ({worst_fn['reference']} vs {worst_fn['candidate']})")
        print(f"Worst fp_rate: {worst_fp['fp_rate']:.10f} ({worst_fp['reference']} vs {worst_fp['candidate']})")


if __name__ == "__main__":
    main()
