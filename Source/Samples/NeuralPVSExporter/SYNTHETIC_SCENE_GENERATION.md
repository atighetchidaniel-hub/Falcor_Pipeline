# Synthetic Scene Generation for NeuralPVSExporter

This folder contains a Falcor sample that exports NeuralPVS-style geometry volumes (GV) and potentially visible volumes (PVV). The exporter handles the Falcor rendering side: loading a scene, rasterizing GV data, computing PVV data, writing compressed volume files, and rendering PVV files for inspection. The helper scripts in `tools/` provide the matching procedural scene-generation and neural-inference handoff used by the Unity workflow.

## What The Generator Adds

`tools/generate_synthetic_neuralpvs_scene.py` creates randomized scenes that the exporter can convert into GV inputs and PVV ground-truth targets. The default mode is Falcor-native and can include floors, boundary geometry, wall segments, primitive meshes, and boolean clearing zones. The `--unity-parity` preset instead mirrors Unity's `RuntimeSceneGenerator` defaults more closely: GLB models, 50-150 objects unless overridden, centered 3D object placement, uniform scale `0.5..2.0`, no extra Falcor floor/walls/boolean zones, and moving view-cell samples from generated scene bounds.

The script also writes a camera path CSV. Each row defines one viewcell/camera sample:

```text
x,y,z,forward_x,forward_y,forward_z,fov
```

The exporter uses this CSV when generating GV/PVV pairs. In practice, this means one synthetic scene can produce many supervised training samples.

## Generate A Synthetic Scene

From the Falcor repository root:

```bash
python3 Source/Samples/NeuralPVSExporter/tools/generate_synthetic_neuralpvs_scene.py \
  --out-dir data/neuralpvs_synthetic/synth_001 \
  --name synth_001 \
  --seed 1 \
  --object-count 180 \
  --wall-count 14 \
  --camera-count 128
```

This creates:

```text
data/neuralpvs_synthetic/synth_001/synth_001.pyscene
data/neuralpvs_synthetic/synth_001/synth_001_camera_path.csv
data/neuralpvs_synthetic/synth_001/synth_001_manifest.json
```

For Unity-comparable synthetic training data, prefer the parity preset:

```bash
python3 Source/Samples/NeuralPVSExporter/tools/generate_synthetic_neuralpvs_scene.py \
  --unity-parity \
  --out-dir data/neuralpvs_synthetic/synth_train_1000_r30_unityparity \
  --name synth_train_1000_r30_unityparity \
  --seed 30030 \
  --samples 1000 \
  --radius 30 \
  --glb-dir /path/to/Unity/Assets/models
```

Use `--object-count 200` if you intentionally want a fixed 200-object stress test; otherwise `--unity-parity` samples the object count from Unity's default `50..150` range.

In Unity-parity mode the CSV sample position is now a moving ViewCell center inside/near the generated scene bounds. This matches Unity's runtime controller more closely: the generator can initially aim at the scene center, but the active ViewCell is refreshed from the camera whenever the camera leaves the current cell. Keeping every CSV row fixed at `sceneBounds.center` made the GV nearly identical for all samples, which is not useful for NeuralPVS training.

For closer Unity synthetic-training parity, generate multiple fresh scene parts instead of one scene with all samples:

```bash
python3 Source/Samples/NeuralPVSExporter/tools/generate_synthetic_neuralpvs_batch.py \
  --out-dir data/neuralpvs_synthetic \
  --name synth_train_1000_r30_batch \
  --seed 30030 \
  --scene-count 10 \
  --samples-per-scene 100 \
  --radius 30 \
  --glb-dir /path/to/Unity/Assets/models
```

Load the generated `_export_plan.csv` in NeuralPVSExporter under **Batch Export Plan** and click **Start batch export**. Falcor will load each listed scene/path pair and export every part automatically. Then merge those exported part datasets:

```bash
python3 Source/Samples/NeuralPVSExporter/tools/merge_neuralpvs_datasets.py \
  --plan data/neuralpvs_synthetic/synth_train_1000_r30_batch/synth_train_1000_r30_batch_export_plan.csv \
  --source-root neuralpvs_export_test/datasets \
  --output-root /var/tmp/atighedl_runs/neuralpvs_falcor_local/datasets \
  --dataset-name falcor_synth_train_1000_r30_batch \
  --link \
  --overwrite
```

## Export GV/PVV In Falcor

Open the `NeuralPVSExporter` sample and set:

| Exporter field | Value |
| --- | --- |
| Scene path | Generated `.pyscene` file |
| Path CSV | Generated `_camera_path.csv` file |
| Dataset name | Same as the generated scene name, for example `synth_001` |
| Output root | Folder where datasets should be written |
| Mode | `Generate GV + PVV` |
| View cell radius | Use `0.3`, `0.6`, or `0.9` for paper-style `r30`, `r60`, or `r90` (centimeters converted to meters/scene units) |
| View cell near/far | Usually `0.3` and `30.0`, unless the scene scale changes |
| PVV sample steps | `10`, matching Unity's default `ViewCellSettings.pvvSampleSteps` |
| Sampling factor | `2`, matching Unity's default supersampling factor for GV/PVV generation cameras |
| Linear Z / Log depth scale | `Linear Z = true` and `0.01`, matching Unity's default volume settings |
| Unity FOV expansion | `30`, matching the extra FOV applied by Unity's generated viewcell cameras |
| High-detail GV cameras | Optional; when enabled, Falcor also renders the Unity-style front, side, and top orthographic GV cameras |

The exporter writes a dataset folder containing `gv/`, `pvv/`, `predicted_pvv/`, and `metadata.json`. The `predicted_pvv/` folder is only a convenience copy of the ground-truth PVV at export time. It is not a neural network prediction. True predicted PVVs should be produced later by the adapted fVDB backend.

## Run fVDB Inference For A Falcor Dataset

After GV export, run the bridge script on the Linux machine that has the fVDB backend and checkpoint:

```bash
python3 Source/Samples/NeuralPVSExporter/tools/falcor_bridge_infer.py \
  --neuralpvs-root /home/atighedl/Adapted_fvdb \
  --dataset-root /path/to/falcor_dataset/synth_001 \
  --checkpoint /path/to/checkpoint.pt \
  --backend fvdb \
  --model OACNNsInterleaved \
  --model-depth 2 \
  --interleaver-r 2 \
  --overwrite
```

This writes `<index>_predicted_pvv.bin.gz` files to `predicted_pvv/`, which can then be loaded with `Render PVV` mode.

## Training And Evaluation Flow

The intended pipeline is:

1. Generate one or more synthetic `.pyscene` files and path CSVs.
2. Use `NeuralPVSExporter` to export GV/PVV pairs from those generated scenes.
3. Train the fVDB NeuralPVS backend using GV as input and PVV as the supervised target.
4. Run inference on held-out synthetic samples or real scenes such as Sponza/Viking.
5. Copy or generate predicted PVV files into the Falcor dataset folder.
6. Use `Render PVV` mode to inspect the rendered result.

## Current Scope

The exporter now follows the Unity data path at the algorithm level: GV is rasterized into the packed volume texture, optional high-detail orthographic GV cameras are available, and PVV is generated by rendering each sample camera to a depth buffer and projecting the visible depth pixels back into the PVV volume. The remaining architectural difference is that Falcor calls the fVDB model through an external Python batch script rather than embedding Python inference inside the renderer process.

For strict byte-level comparisons, use the same scene geometry, camera CSV, screen/raster size, sampling factor, volume settings, and GPU/API conventions as the Unity run. Small differences can still appear if material culling, alpha testing, projection conventions, or scene import transforms differ between Unity and Falcor.
