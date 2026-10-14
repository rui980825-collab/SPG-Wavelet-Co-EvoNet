# SPG-Wavelet Co-EvoNet

## Overview

SPG-Wavelet Co-EvoNet is a detection-oriented framework for offline paired RGB-T
training-sample construction. Its core implementation combines prototype-guided
paired-object selection, context-constrained RGB-T co-generation, RGB-guided wavelet
refinement of thermal patches, density-aware scene construction, and automatic
annotation generation.

The framework operates at the training-data construction stage and supplies paired
samples for downstream RGB-T object detection. The ATSS tools provide a detector-based
training and evaluation workflow rather than a new detector architecture.

## Repository Structure

| Directory | Purpose |
| --- | --- |
| `coevonet/` | Paired-data loading, auxiliary networks, training losses, candidate constraints, and thermal refinement. |
| `scripts/` | Data preparation, prototype and prior construction, sample synthesis, pool merging, and statistics. |
| `configs/` | Sample-construction and auxiliary-model training configurations. |
| `third_party/mmdet3_rgbt_atss/` | Dual-stream ATSS runtime, data-composition configurations, and training and evaluation tools. |
| `docs/` | Implementation details, environment information, source provenance, and validation records. |

The `coevonet/` package is shared by the preparation, training, and construction entrypoints.

## Requirements

The reference Linux environment uses:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| MMDetection | 3.3.0 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |

Sample construction uses PyTorch, torchvision, NumPy, OpenCV, Pillow, PyYAML,
tqdm, and Matplotlib. ATSS training and evaluation additionally use the OpenMMLab
components and pycocotools. Platform-specific environment information is available
in `docs/observed_environments.json`.

Install a matching PyTorch/torchvision pair, then install the construction dependencies:

```bash
python -m pip install -r requirements.txt
```

For ATSS training and evaluation, install the additional dependencies with an MMCV
build compatible with your PyTorch and CUDA versions:

```bash
python -m pip install -r requirements-evaluation.txt
```

Run commands from the repository root. `scripts/run.py` sets the working directory
and import paths for the selected entrypoint. On Windows, ASCII repository and data
paths are recommended for compatibility with the preparation script's OpenCV filename I/O.

## Dataset Preparation

Obtain RGBT-Tiny through its [official project](https://github.com/XinyiYing/RGBT-Tiny).
Arrange the images and COCO annotations as follows:

```text
data/raw/dataset/
  images/<sequence>/00/<frame>.jpg
  images/<sequence>/01/<frame>.jpg
  annotations_coco/instances_00_train2017.json
  annotations_coco/instances_01_train2017.json
  annotations_coco/instances_00_test2017.json
```

Use the training and held-out evaluation partitions specified in the manuscript.
The supplied training annotations must reflect the intended construction/training
partition. Build patch banks, backgrounds, prototypes, and priors from training
data, reserving held-out data for evaluation.

With the training annotations and paired images in place, run:

```bash
python scripts/run.py scripts/00_prepare_assets.py
```

This entrypoint has no CLI flags. It reads the training pairs, extracts paired object
patches and 256 x 256 background crops, and writes `data/patches/`. Background
exclusion uses the RGB-side bounding boxes.

The preparation example above targets RGBT-Tiny. For RGBTDronePerson, consult its
[official project](https://nnnnerd.github.io/RGBTDronePerson/) for dataset access and
benchmark preparation resources; the RGBT-Tiny commands above do not define a
RGBTDronePerson-specific preprocessing or registration protocol. Follow the terms
of the respective dataset providers when obtaining and using the data.

## Auxiliary Model and Prototype Preparation

Prepare the auxiliary checkpoint using the following training entrypoint, or
specify a compatible checkpoint for the configured network:

```bash
python scripts/run.py scripts/10_train_coevonet.py --config configs/coevonet.yaml
```

Set the shell variable `CHECKPOINT` to the resulting or compatible checkpoint path.
The selected checkpoint must match the encoder architecture and channel dimensions
used by the prototype builder and sample-construction configuration. Then build the
paired prototype bank and scene-density prior:

```bash
python scripts/run.py scripts/02_build_prototype_bank.py --patch_bank_root data/patches/patch_bank --generator_ckpt "$CHECKPOINT" --feature_mode generator_encoder --out_root runs/prototypes/local
python scripts/run.py scripts/05_build_scene_density_prior.py --ann-file data/raw/dataset/annotations_coco/instances_00_train2017.json --out-yaml configs/local/density_prior.yaml --dataset-name RGBT-Tiny
```

The prototype builder exports `prototypes.npy` and associated feature and metadata
records. Context rules and a scene-density prior are provided under `configs/`.
To estimate context rules from the supplied training annotations, run:

```bash
python scripts/run.py scripts/01_build_context_prior.py --coco data/raw/dataset/annotations_coco/instances_00_train2017.json --raw_root data/raw/dataset/images --out_yaml configs/local/context_rules.yaml
```

The local-context analyzer is implemented in `coevonet/synthesis/constraints.py`,
and the generator encoder is implemented in `coevonet/models/generator.py`.
`docs/IMPLEMENTATION_MAP.md` describes the entrypoints and module relationships.
Use the `generator_encoder` feature mode with the corresponding trained checkpoint
when building the prototype bank for the construction configurations below.

## Sample Construction

Construction configurations specify the checkpoint, prototype bank, input assets,
sampling controls, and output location. Configure your artifact paths in a new YAML
while retaining the selected configuration's sampling and refinement settings:

```bash
python scripts/configure_paths.py --source configs/synthesis_v25lite_phaseC_train_tau027.yaml --output configs/local/phasec.yaml --generator-ckpt "$CHECKPOINT" --proto-bank-path runs/prototypes/local/prototypes.npy
python scripts/run.py scripts/20_synthesize_dataset.py --config configs/local/phasec.yaml
```

Phase-B and density-aware configurations use the same `20_synthesize_dataset.py`
entrypoint. Select the appropriate YAML for the intended switches, thresholds,
seeds, and output budget. Parameter definitions and effective settings are specified
by the configuration and its consuming code, rather than inferred from filenames.

The semantic hard-prior confidence threshold is set to 0.30
(`hard_prior_tau=0.30`) in the corresponding synthesis configurations.

The prior-construction commands write new files under `configs/local/`. To use those
priors, set `scene_density.prior_path` and `context_rules_path` in the corresponding
local construction YAML. These files are not selected automatically by the commands above.

For multiple shards, inspect the launcher commands in dry-run mode:

```bash
python scripts/run.py scripts/21_run_synth_shards.py --template-config configs/local/phasec.yaml --shard-start 0 --shard-end 9 --dry-run
```

Merge generated outputs and align their category IDs with the real-data annotations:

```bash
python scripts/run.py scripts/03_merge_synth_coco.py --runs-root runs/synth/uav_rgbir/v25lite_phaseC_train_tau027 --out-root data/synth/local_phasec --copy-images
python scripts/run.py scripts/04_align_coco_category_ids.py --real-ann data/raw/dataset/annotations_coco/instances_00_train2017.json --synth-ann data/synth/local_phasec/annotations/instances_synth_merged.json
python scripts/run.py scripts/22_summarize_synth_runs.py --runs-root runs/synth/uav_rgbir/v25lite_phaseC_train_tau027
```

Check each script's `--help` for its supported options; asset preparation is the exception noted above.

## Output Format

Each construction run writes the following outputs under its run directory:

| Output | Location |
| --- | --- |
| Paired RGB images | `images/rgb/` |
| Paired TIR images | `images/tir/` |
| COCO-style bounding-box annotations | `annotations/instances_synth.json` |
| Candidate and scene statistics | `tables/` |
| Construction summary | `meta/synthesis_summary.json` |
| Configuration records | `configs/` |

The COCO annotation set records successfully inserted instances using their final
image-space bounding boxes and category IDs.

## Detection and Evaluation

`third_party/mmdet3_rgbt_atss/` provides the dual-stream ATSS runtime, paired-data
loading transforms, a source-aware sampler, and configurations for Real-only,
Real + Phase-C, and Real + Phase-C density training. Dataset and output paths are
relative to the project root.

The tools support single-run and multi-seed training, checkpoint-based tiny/extra-tiny
evaluation, and run-summary aggregation. Inspect their arguments before selecting
the configuration, checkpoint, seed, and work directory for a run:

```bash
python scripts/run.py third_party/mmdet3_rgbt_atss/tools/run_formal_matrix_dualstream_atss.py --help
python scripts/run.py third_party/mmdet3_rgbt_atss/tools/train_formal_dualstream_atss.py --help
python scripts/run.py third_party/mmdet3_rgbt_atss/tools/eval_tiny_et_formal_dualstream_atss.py --help
python scripts/run.py third_party/mmdet3_rgbt_atss/tools/summarize_formal_dualstream_atss.py --help
```

The YAML construction configurations and Python detector configurations control
separate stages. Detector settings are defined by the selected Python configuration
and any explicit CLI overrides. Specify the required datasets and checkpoint paths
before invoking training or evaluation.

## Citation

```bibtex
@unpublished{liu_spg_wavelet_coevonet,
  author = {Rui Liu and Jiayan Zhang and Chengzhao Shan and Yongkui Ma},
  title = {{SPG-Wavelet Co-EvoNet}: Detection-Oriented {RGB-T} Sample Construction for {UAV-Borne} Tiny Object Detection Under Varying Illumination Conditions},
  note = {Manuscript under revision}
}
```

## Implementation Scope

This repository provides the core implementation of SPG-Wavelet Co-EvoNet,
including paired RGB-T sample construction, configuration files, data-preparation
utilities, and ATSS-based training and evaluation tools. The steps above connect
prepared paired data and auxiliary artifacts to the selected construction and
detector configurations.

Implementation details and source provenance are documented in `docs/IMPLEMENTATION_MAP.md`
and `docs/source_map.csv`. Run `python -B scripts/check_release.py` for source checks;
add `--runtime` in an environment with the listed dependencies to check imports,
configuration loading, CPU unit fixtures, and CLI help. Validation records are
available in `docs/LOCAL_CHECKS.md`.
