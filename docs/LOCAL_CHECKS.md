# Local Packaging Checks

Checked on 2026-10-14. This report concerns source packaging, not manuscript-table reproduction.

| Check | Result |
| --- | --- |
| Python syntax and project import paths | PASS: 52 Python files |
| Source-derived file integrity | PASS: 72 release hashes checked against the extraction map |
| Core computation preservation | PASS: core Python ASTs unchanged |
| Construction YAML loading | PASS: 22 configs |
| Core module imports | PASS: 10 modules |
| Density/sampling helper checks | PASS: configured interval and normalization checks |
| Thermal-refinement unit fixture | PASS: shape and dtype checks on a mathematical CPU fixture |
| Launcher, path helper and core CLI help | PASS: 12 commands |
| ATSS runtime registration and config loading | PASS: 4 configs |
| ATSS CLI help | PASS: 4 commands |
| Sensitive-file and content screening | PASS: no data/model assets, credentials or private absolute paths |

Runtime checks used the existing Linux/WSL Python 3.11 environment with PyTorch 2.7.0+cu128,
torchvision 0.22.0+cu128, MMDetection 3.3.0, MMCV 2.1.0 and MMEngine 0.10.7.
No training, experimental sample construction, detector inference, checkpoint loading or
new AP/AR evaluation was performed. No fresh environment installation was tested.
The archived Windows auxiliary environment does not include all training dependencies;
its metadata is recorded separately instead of presenting it as a verified runtime.

## Recheck Locally

```bash
python -B scripts/check_release.py
python -B scripts/check_release.py --runtime
```

The first command checks text-source syntax, file hashes and package content. The second
requires the construction dependencies and optionally checks the existing ATSS environment.
Both commands avoid training, dataset generation and experimental evaluation.

## Preparation Status

| Status | Value | Scope |
| --- | --- | --- |
| SOURCE_AUDIT | PASS | Copied source provenance and actual dependency paths confirmed |
| MANUSCRIPT_ALIGNMENT | PARTIAL | Explicitly deferred at the author's request |
| DEPENDENCY_STATUS | PARTIAL | Imports checked in existing environment; datasets, weights and prototype assets intentionally external |
| LOCAL_CHECK | PASS | Packaging/import/config/CLI/unit checks listed above |
| SECURITY_CHECK | PASS | Content screened; no sensitive source assets copied |
| PUBLICATION_READINESS | PARTIAL | Local candidate complete; author must approve redistribution rights and select a license |

All original selected source files remained unchanged. Within this independent release,
absolute project prefixes were removed from copied path strings/comments. Subsequently, at the
author's request, all 30 active `hard_prior_tau=0.27` definitions in 10 construction presets
were changed to `0.30`. This intentional semantic parameter change is recorded in `source_map.csv`.
The Python gate/default, photometric configuration, seeds, schedules and remaining YAML values
are unchanged. Historical filenames and source-run comments are retained as provenance.

The threshold revision is checked by comparing parsed configurations with their pre-revision
copies and resolving the effective `ConstraintConfig` through the actual synthesis entrypoint.
The semantic gate is checked on both sides of the revised confidence boundary using a unit
fixture. Existing experimental pools and reported AP/AR values are not regenerated or reattributed
to this revision. New helpers only handle launching, copied-config path fields and release checks.
No remote repository, upload or Git push was performed.
