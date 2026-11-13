# Frozen B residual study, 2026-11-11

- `cv/`: all 14 candidates, 5 scene-disjoint folds, predeclared selection and complete OOF rows.
- `learning_curve/`: nested 10/25/50/100 percent training-scene budgets, median and spline models, held-out results.
- `final/`: frozen selected model, real training deployment smoke and one official 654-image evaluation.
- `analysis/`: descriptive scene/case tables and training-only learning-curve PNG/PDF.
- `external_arrays.json`: large raw/prepared/final depth arrays, locations and SHA256.

The accepted candidate is `residual_median_ratio_l1_a0.5`: AbsRel 0.0996068934, RMSE 0.3831236302 m, delta1 0.9055456208. Historic-test-exposure and nonuniform per-image/depth-bin effects are disclosed in `../../docs/PART3B_FINAL_20261111.md`.

Run `python scripts/verify_results.py` from the repository root. The current aggregate manifest is `results/final_audit_20261111/manifest.json`; old manifests remain byte-identical. This directory contains experimental evidence, not a coursework report or submission archive.
