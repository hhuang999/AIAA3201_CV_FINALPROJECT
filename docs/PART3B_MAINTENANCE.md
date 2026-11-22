# Part 3B maintenance and interpretation notes

This maintenance follows the review of `609f36e834d00e6059348166e3172d4bb51aed1f` on `feat/part3b-sota-v2`. It fixes runtime receipt handling, cache validation, public help and test discovery, and the interpretation of diagnostic evidence. It does not introduce a candidate, refit a model, select on the test set, or change an existing score. H6 remains the single-pass primary; M6 remains the two-pass extension.

The existing `PART3B_OVERVIEW.md`, `PART3B_SOTA.md`, `PART3B_SOTA_V2.md`, experimental figures and videos are retained as snapshots. The corrections below govern their interpretation until a separately authorized report revision. Source/configuration entries in the two delivery manifests are refreshed for maintenance; their experimental-result entries remain unchanged.

## Runtime receipts and scope

The cache builder writes `mean_seconds_per_image` in **seconds** separately to `runtime_aux.json` and `runtime_flip.json`. The previous evaluator looked for different keys in the aux file only and substituted `43.04` and `44.13` milliseconds. Consequently the archived totals such as `47.67` and `93.86 ms` are not validated end-to-end measurements. The archived JSON receipts are preserved, not silently rewritten.

The reader now consumes both real receipts, converts seconds to milliseconds, and retains their paths and checksums. Missing timing inputs produce unknown component totals rather than constants. Malformed timings and incompatible hardware metadata are rejected or excluded from a combined estimate. Measured post-processing and the sum of separately measured components are labeled separately. Post-processing starts from cached arrays; it does not include the full RGB-to-feature pipeline, all transfers or file loading. A component sum is not a direct end-to-end benchmark. GPU peak memory, where newly measured, is scoped to the post-processing measurement, not to the complete backbone pipeline.

The recorded CPU/GPU post-processing measurements can still be read as measurements of their stated paths. They do not establish a like-for-like speedup between a NumPy baseline, CPU Torch and GPU Torch. No replacement hardware timing or official-test evaluation is performed by this maintenance.

## Cache identity and coordinate conventions

Both auxiliary and flip caches must have complete manifests bound to the actual NYU data, split, original depth, predicted intrinsics and image IDs. Required files are checksum-verified, and the stored IDs must agree with the original ordering. Feature/depth/confidence/intrinsics shapes are checked. Value validation is restricted to the requested split, preserving the training-only target boundary. Reusing an incomplete or mismatched directory fails with a concrete error; it is not reported as verified.

The frozen evaluator's cache hashes remain authoritative for replay of the historical experiment. A newly rebuilt cache may be valid but differ numerically across devices; it must not be represented as an exact reproduction of the original cache. The frozen model hashes are unchanged:

| Artifact | SHA-256 |
|---|---|
| `results/part3b_sota/frozen_sota_model.json` | `5f8326a63470416e077cec12afb41ed793ffb2a58abf8d94a0eb88b84f08e86c` |
| `results/part3b_sota_v2/frozen_sota_v2_model.json` | `a7633ffa74b21b70e196ba822c15fe454ac96b528443d4a70753f0aa4ff479e6` |

M1 intentionally returns the flip prediction's principal point to the network's native prior before averaging camera features. This is a learned-prior heuristic described in the original Round 2 notes, not physical averaging of two intrinsics expressed in the original image coordinate system. Its name and numerical behavior are retained. H6 and M6 use different branches and do not perform this M1 principal-point reflection. No M1 ablation is rerun here.

## Scientific interpretation

- The inherited B3 work had official-test exposure. Round 2 was additionally motivated by Round 1's official-test failure cases. Calibration parameter fitting and quantitative selection use the official 795 training images, with models committed before each round's recorded evaluation. The 654-image official test set is therefore a reused benchmark, not a fresh blind holdout. A receipt records the delivered evaluation, not the absence of any other historical execution.
- The fixed backbone may process RGB images from both splits when generating caches. This differs from reading test targets for fitting. The train-only readers restrict access by image ID; they do not erase historical knowledge of test cases.
- The secondary residual R-squared regressions reuse globally constructed first-stage OOF targets in another CV pass. Other folds' targets can indirectly depend on the current validation fold through the first-stage fit. These R-squared and related secondary spatial diagnostics are **exploratory, non-nested diagnostics**, not independent cross-fitted generalization estimates. Their source outputs now carry this interpretation explicitly. The primary calibration CV fits directly within each fold and is a separate path. Nested refitting would be a new experiment and is not needed to retain the frozen H6 result.
- In Round 2, all 26 candidates were evaluated on the canonical five folds. The four-seed records cover nine finalists and three reference methods, not all 26 candidates. The H7-to-H6 collapse was observed in all five canonical folds, not in every seed.
- H6 versus v1 has positive paired 95% intervals for AbsRel and RMSE; its delta1 interval crosses zero. M6 versus H6 has intervals crossing zero for all three metrics. This is insufficient evidence of an advantage, not an equivalence test or strict Pareto dominance: M6 has a slightly lower RMSE point estimate.
- Bootstrap tail values of `0` denote zero nonpositive draws in 10,000 resamples. They do not mean a true probability of zero. Effective degrees of freedom are estimates, distinct from raw parameter counts. A lower test score than OOF is not proof of zero overfitting; the fit budgets and cohorts differ. The reported method names do not establish external state of the art.
- The Round 1 per-image-scale oracle change `0.102441 -> 0.061383` removes approximately **40.1%** of AbsRel, not 60.6%. The oracle uses ground truth for diagnosis and is not a deployable result.
- H6 remains the model chosen before Round 2 scoring. S4's slightly better test AbsRel/delta1 is retained as an ablation, not a reason for post-test model switching. Failure cases and the 6-10 m delta1 regression remain relevant limitations.

## Verification and use

Run the project suite from the repository root with the pinned reproduction environment:

```bash
python -m depthlab.verify
python -m depthlab.verify --manifest results/part3b_sota/manifest.json
python -m depthlab.verify --manifest results/part3b_sota_v2/manifest.json
python -m pytest tests -q
python -m depthlab.sota_calibration --help
python -m depthlab.sota_calibration learning-curve --help
```

`pytest.ini` restricts default discovery to `tests/`, so a prepared `third_party/` checkout cannot accidentally add upstream GPU scripts to the course-project suite. The learning-curve help correctly escapes its literal percent sign. Tests use synthetic fixtures for corrupt-cache handling and timing metadata; they do not fit a new project model or score the official test set.

The [maintenance validation record](../results/maintenance_20261122/validation.json) and [delivery manifest](../results/maintenance_20261122/manifest.json) record the checks and protected-artifact preservation. Older baseline manifests remain historical snapshots tied to their own commits. See [REPRODUCTION.md](REPRODUCTION.md) for environment, input preparation and explicit frozen-evaluation commands.
