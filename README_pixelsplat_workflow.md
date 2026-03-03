# PixelSplat Sim2Real — Workflow Guide

---

## Overview

All experiments using datasets built on the basis of **SEED4D** (SecoGAN, FLUX2_9B) and Nuscenes share the same dataloader, config files, and pipeline structure. Only the specific config values need to be adjusted per experiment.

---

## Configuration Files

| File | Purpose |
|---|---|
| `dataset_seed4d.py` | Set the active dataset |
| `epipolar.yaml` | Set model parameters |
| `main.yaml` | Set paths and checkpoints |
| `seed4d.yaml` | Set training parameters |
| `seed4d_test.yaml` | Set testing parameters |
| `eval_carla_view.yaml` | Set input/output resolution for evaluation |
| `train_carla_view.yaml` | Set input/output resolution for training |

---

## Available Experiments

### Ego–Ego
- Standard setup with ego-only views from the SEED4D-based dataset.

### Ego–Exo
- Uses both ego and exo camera views from the SEED4D-based dataset.

### Ego–Ego NuScenes *(val/test only)*
- Ego-only views evaluated against real-world NuScenes data.

### Ego–Exo NuScenes *(val/test only)*
- Ego and exo views evaluated against real-world NuScenes data.

### Ego–Exo Mixed *(training only)*
- Mixed training using ego and exo views from the SEED4D-based dataset.

### Ego–Exo Mixed Domain *(training only)*
- Mixed training combining SEED4D-based ego/exo views with NuScenes ego views for sim-to-real domain adaptation.

### Ego–Exo NuScenes Scene *(test only)*
- Tests on consecutive NuScenes frames to generate a GIF or video of a scene.
- Set `nuscene_scene_index` in `seed4d_test.yaml`
- Set `target_views` in `eval_carla_view.yaml`

---

## Running Experiments

### Training

```bash
python3 -m src.main +experiment=seed4d.yaml mode=train
```

### Testing

```bash
python3 -m src.main +experiment=seed4d_test.yaml mode=test
```

---

## Generating a GIF / Video from a Scene

After running the `ego-exo-nuscenes-scene` experiment, use the scene generator utility:

```bash
python3 src/utils/gif_video_scene_generator.py
```