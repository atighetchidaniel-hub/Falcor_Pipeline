# NeuralPVS Falcor Lab Validation Checklist

This checklist is for validating the Falcor NeuralPVS exporter against the Unity/DerThomy pipeline on the lab Linux machine.

## 1. Before Building

- Confirm this Falcor copy is the one with the Unity-parity changes.
- Confirm the following files exist:
  - `Source/Samples/NeuralPVSExporter/NeuralPVSDepthOnly.3d.slang`
  - `Source/Samples/NeuralPVSExporter/NeuralPVSPVV.cs.slang`
  - `Source/Samples/NeuralPVSExporter/NeuralPVSGV.3d.slang`
  - `Source/Samples/NeuralPVSExporter/tools/generate_synthetic_neuralpvs_scene.py`
  - `Source/Samples/NeuralPVSExporter/tools/falcor_bridge_infer.py`
- Confirm `NeuralPVSDepthOnly.3d.slang` is listed in `Source/Samples/NeuralPVSExporter/CMakeLists.txt`.

## 2. Build Smoke Test

Build the Falcor sample first, before running any long exports. The first goal is only to catch C++ or Slang compile errors.

Expected result:

- `NeuralPVSExporter` builds.
- No missing shader-file errors.
- No remaining DXR/raytracing requirement for PVV generation.

## 3. Tiny Export Test

Use a very small synthetic scene first.

Suggested generator command:

```bash
python3 Source/Samples/NeuralPVSExporter/tools/generate_synthetic_neuralpvs_scene.py \
  --out-dir data/neuralpvs_synthetic/smoke_001 \
  --name smoke_001 \
  --seed 1 \
  --object-count 20 \
  --wall-count 4 \
  --camera-count 4
```

In `NeuralPVSExporter`, set:

- Scene path: `data/neuralpvs_synthetic/smoke_001/smoke_001.pyscene`
- Camera path CSV: `data/neuralpvs_synthetic/smoke_001/smoke_001_camera_path.csv`
- Dataset name: `smoke_001`
- Mode: `Generate GV + PVV`
- Volume size/depth: keep Unity-compatible defaults unless testing faster debug settings
- Sampling factor: `2`
- PVV sample steps: start with `2` for smoke test, then use `10` for real parity test
- Linear Z: enabled
- Log depth scale: `0.01`
- Unity FOV expansion: `30`
- High-detail GV cameras: disabled first, then test enabled separately

Expected result:

- Dataset folder contains `metadata.json`, `gv/`, `pvv/`, and `predicted_pvv/`.
- GV and PVV bit counts are nonzero.
- PVV should generally contain fewer or equal relevant visible regions than the GV, but the exact count depends on the scene.

## 4. Unity-Parity Export Test

Run a real comparison only after the tiny export succeeds.

Use matching settings between Unity and Falcor:

- Same camera path/viewcell centers
- Same scene geometry and scale
- Same volume size and depth
- Same sampling factor
- Same PVV sample steps
- Same radius
- Same near/far plane
- Same FOV expansion
- Same linear/log depth setting
- Same high-detail setting

For bit-level comparisons, compare raw decompressed GV/PVV bytes from matching sample indices.

Useful metrics:

- Exact byte equality
- Set-bit count difference
- Intersection over union of occupied bits
- False-positive and false-negative bit counts between Unity and Falcor volumes

If exact equality fails but IoU is high, inspect:

- Coordinate-system or handedness mismatch
- Y-axis texture origin mismatch in PVV depth projection
- Scene import scale or transform differences
- Alpha-tested materials
- Back-face culling differences in the PVV depth pass
- Linear depth/projection convention differences

## 5. fVDB Bridge Test

After Falcor GV export works, run the bridge on the Linux machine:

```bash
python3 Source/Samples/NeuralPVSExporter/tools/falcor_bridge_infer.py \
  --neuralpvs-root /home/atighedl/Adapted_fvdb \
  --dataset-root /path/to/falcor_dataset/smoke_001 \
  --checkpoint /path/to/checkpoint.pt \
  --backend fvdb \
  --model OACNNsInterleaved \
  --model-depth 2 \
  --interleaver-r 2 \
  --overwrite
```

Expected result:

- `predicted_pvv/` contains predicted PVV files.
- `Render PVV` mode can load and display the predicted PVV.

## 6. What To Record

For the thesis/evaluation notes, record:

- Falcor commit/folder version used
- Unity commit/folder version used
- Scene name
- Volume size/depth
- Radius
- Sampling factor
- PVV sample steps
- Linear/log depth setting
- High-detail on/off
- GV bit count
- PVV bit count
- Predicted PVV bit count
- Build/runtime errors
- Screenshots of rendered PVV mode

## 7. Current Caveat

The Falcor exporter now follows Unity's algorithmic GV/PVV path: rasterized GV, optional high-detail orthographic GV cameras, sample-camera depth rendering, and depth-to-PVV projection. True byte-level equality still must be validated on the lab machine because renderer conventions, scene import, materials, culling, and projection details can still differ between Unity and Falcor.
