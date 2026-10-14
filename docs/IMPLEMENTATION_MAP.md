# Implementation Map

This is a source-navigation map, not a manuscript-equation alignment claim.

| Functionality | Existing implementation |
| --- | --- |
| Pair preparation / patch and background extraction | `scripts/00_prepare_assets.py` |
| Auxiliary encoder training | `scripts/10_train_coevonet.py` -> `coevonet/engine/train_generator.py` |
| Encoder architecture / Haar blocks | `coevonet/models/generator.py`, `coevonet/models/wkan.py` |
| Prototype and paired feature bank | `scripts/02_build_prototype_bank.py` |
| Runtime prototype and weighted patch selection | `scripts/20_synthesize_dataset.py`: `_load_prototype_runtime`, `_choose_patch_with_prototype` |
| Scene budget resolution / sampling | `scripts/20_synthesize_dataset.py`: `_resolve_scene_density_config`, `_sample_scene_density` |
| Scene-density prior estimation | `scripts/05_build_scene_density_prior.py` |
| Context, semantic and photometric screening | `coevonet/synthesis/constraints.py` |
| Thermal refinement | `coevonet/synthesis/wavelet_refiner.py`: `wavelet_refine_tir` |
| Placement, overlap, accepted insertions and COCO export | `scripts/20_synthesize_dataset.py`: `main`, `bbox_iou`, `paste_bgr`, `paste_gray` |
| Pool sharding / merging / category alignment | `scripts/21_run_synth_shards.py`, `03_merge_synth_coco.py`, `04_align_coco_category_ids.py` |
| Existing construction summary | `scripts/22_summarize_synth_runs.py` |
| Paired detector loading and sampler | `third_party/mmdet3_rgbt_atss/rgbt_atss_runtime/` |
| ATSS configs / train / tiny-et evaluation | `third_party/mmdet3_rgbt_atss/configs/` and `tools/` |

Core Python ASTs are unchanged. Changes in copied detector scripts/configs are limited to absolute
project-path strings. Their numerical configuration values, algorithms, operator fallback and
training/evaluation behavior are not rewritten. Source hashes and per-file changes are recorded
in `source_map.csv`. The two path helpers only launch preserved entrypoints or edit path
fields in a new YAML; they do not implement replacement scientific modules.

At the author's request, the 10 construction presets that specified `hard_prior_tau=0.27`
were updated to `0.30` in all three configuration locations: the top-level key, `thresholds`,
and `constraints`. The gate's Python default was already `0.30`. This is an intentional
semantic-screening parameter change, not a path-only edit or a photometric threshold change.
Other presets and photometric settings remain unchanged. Earlier `tau027` names, comments,
and run references identify the original preset, not experimental results from this revision.
