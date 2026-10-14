## MMDet3 Dual-Stream ATSS Runtime

This directory hosts the main external detector line for the Co-EvoNet paper.

Design principles:
- keep the detector modern and stable (`MMDetection 3.x`)
- keep the model simple and detector-agnostic (`dual-stream ATSS-R50-FPN`)
- keep the paper detector-side evidence focused on utility validation

The package is intentionally independent from `mmdet3_qfdet` so that QFDet
remains an engineering feasibility / diagnostic line rather than the paper's
main validation path.
