# Release Scope and Approval Boundary

Included: source-derived paired preparation, auxiliary training, prototype-bank construction,
candidate constraints, thermal refinement, scene budgets, insertion/export, pool utilities,
existing custom ATSS runtime/configs and tiny/extra-tiny evaluation code.

Excluded: datasets and per-instance annotations, trained/proprietary weights, prediction outputs,
raw training logs, virtual environments, Git history, empty placeholder modules, QFDet, unrelated
research code, manuscript sources and response letters.

Current-manuscript/Algorithms 1-2 alignment has been deferred by the author. This package is not
described as a verified end-to-end reproduction of manuscript tables. Dataset-specific additional
protocols are not inferred or reimplemented. External model weights and prototype assets must be
provided or prepared legitimately by the user. Only trusted checkpoints/prototype files should be
loaded: the preserved source uses `torch.load` and pickled NumPy prototype records.

Before publication, the author must approve source redistribution, dataset/model instructions,
third-party obligations and a repository-wide license. No license is selected automatically and
no repository creation, network upload, git commit or git push is performed.

The release includes an author-requested semantic threshold revision: presets originally setting
`hard_prior_tau=0.27` now set it to `0.30`. Photometric screening is unchanged. This revision
affects subsequent construction runs; existing frozen outputs and experimental results remain
associated with their original settings and have not been recomputed.
