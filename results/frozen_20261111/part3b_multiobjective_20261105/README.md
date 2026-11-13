# B multiobjective calibration evidence

Primary: `spline_rmse`, selected from training scene CV before the frozen test evaluation.

- `prepared/`: preregistered training-only protocol and exact prepared array hashes.
- `cv/fold0` through `cv/fold4`: all 18 candidate fits, three controls, full-resolution held-out metrics and completion receipts.
- `selection/`: all-candidate table, OOF metrics, bootstrap comparisons, fixed selection, per-fold primary comparison.
- `final/`: all-training fit records, frozen models, official 654-image test metrics.
- `audit/`: independent original-NumPy metrics, scene bootstrap, failure/category/depth-bin diagnostics and measured calibration overhead.
- `provenance/`: tests, input hashes, fixed-selection replay, deployment replay and evaluator-precision correction evidence.
- `external_arrays.json`: durable archive location and SHA256 for exact prepared arrays and full primary predictions. Large arrays are external; all lightweight numerical evidence is included here.

Run from the repository root:

```bash
python scripts/verify_results.py --manifest results/part3b_multiobjective_20261105/manifest.json
```

The original result manifest is unchanged and must still pass separately. See [method and reproduction instructions](../../docs/PART3B_MULTIOBJECTIVE_20261105.md). Part 1/2 and strict median B remain unchanged. The primary is a six-parameter monotone calibration extension under the course method-flexibility rule, not literal one-scalar B. Test means improve all three metrics; the RMSE gain interval versus median B includes zero. Historical test exposure and fixed-model bootstrap limitations are disclosed.
