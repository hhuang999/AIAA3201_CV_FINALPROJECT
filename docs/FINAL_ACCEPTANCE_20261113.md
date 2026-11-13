# Final acceptance of Parts 1, 2 and 3B

The additional acceptance round found no reproducible defect requiring another change to the numerical implementation. Code remains at `f9693224f0c9d079215f2b4175eaff31c21a2b31`, within the teammate's `depthlab` architecture. This acceptance adds evidence only. The earlier fixes, their rationale and the migration lineage are documented in [the final code review](FINAL_CODE_REVIEW_20261113.md).

## Requirements and scope

The current supplied `Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf` matches the archived requirements file, SHA-256 `ea65076917e87febb0c3c67773b9fc6edf304e2821a74266a03f22ef33abc226`. No requirements change was found in this round.

| Course requirement | Implementation and evidence |
|---|---|
| Part 1: classical stereo on the 15 Middlebury scenes, calibrated metric depth and invalid-pixel handling | `depthlab/stereo.py`; all 15 scenes rerun from the committed source export, across three configurations. All 135 disparity/depth/mask arrays are byte-identical to the restored frozen outputs. |
| Part 1: image-wise DAV2 video-depth baseline | `depthlab/infer_video.py --method image`; prior fresh 70-frame inference parity retained; the clean source export now verifies and reuses the complete cache. |
| Part 2: RGB-only UniDepthV2, official NYU split/crop/depth bounds and image-averaged metrics | `depthlab/infer_nyu.py`, `data.py`, `metrics.py`, `evaluate.py`; preparation verifies 795 training images, 654 test images and no shared scenes. The complete 1,449-image cache passes the actual inference entrypoint's input and artifact checks. |
| Part 2: VDA Small offline FP16 on the same frames; consistent visualization and efficiency records | `depthlab/infer_video.py --method video`; the original 70-frame cache passes the same clean-source checks. Timing/warmup/device routing was reviewed against the pinned upstream implementations. No new rendered video was produced. |
| Part 3B: a train-fitted global scalar frozen on test | The strict scalar remains an explicit control. The primary's monotone curve and bounded prediction-conditioned residual are labeled extensions, with train-only fitting metadata and the original frozen parameters. |
| Part 3B: dependable evidence for the extension | All 795 training images, five scene-disjoint folds, 14 residual candidates plus three controls, and 13,515 per-image metric rows were replayed using the current modules and original saved fold parameters. No fitting or test evaluation was performed. |

The supplied clip contains 70 frames at 24 FPS. Processing all available frames is recorded explicitly; the input was not padded or represented as a 100-frame clip.

This is code, experiment-preservation and reproducibility acceptance. It does not complete the separately deferred coursework report, presentation/video production or final submission packaging.

## Fresh execution and numerical results

A Git archive of the reviewed commit was extracted into a clean source directory. It used the existing pinned Linux/Python 3.10 reproduction environment and external data/upstream model checkouts; this round did not reinstall dependencies or claim a new machine installation.

- All **174 tests** passed from the clean source export. Four upstream Triton deprecation warnings remained; no test failed.
- All **eight** recorded commands passed: manifest verification, the full test suite, input preparation, complete stereo execution, and NYU/decoded-frame/DAV2/VDA complete-cache reuse.
- Stereo: **135/135 arrays byte-identical** across 15 scenes and three configurations.
- Frozen-fold replay was first checked on two training images per fold: **10 images / 170 rows**, with zero metric difference.
- Full frozen-fold replay: **795 images / 249 scenes / 17 methods / 13,515 rows**.
- Maximum absolute differences from the original OOF CSV: AbsRel **2.498001805406602e-16**, RMSE **8.881784197001252e-16**, delta1 **0**; maximum residual-scale difference **2.220446049250313e-16**. These are float-rounding differences, below the declared **1e-12** acceptance tolerance.
- Reconstructing the original eligibility rule from replayed scores gives the same two eligible candidates and the same primary, `residual_median_ratio_l1_a0.5`. This is an audit of the original selection, not a new deployment decision.
- Each fold's fit/validation IDs match its saved receipt. The scene overlap is zero in all five folds. Base-fold models, selection records, dataset/split identity and the exact original training prediction cache are hash-bound.

The previous complete fresh-inference and fixed-test replay evidence remains in [restoration evidence](../results/restoration_20261113/validation.json) and [final review evidence](../results/final_review_20261113/validation.json). Official-test results are still AbsRel **0.09960689343231784**, RMSE **0.3831236302386158 m**, delta1 **0.9055456208197196**. No official-test evaluation was repeated in this acceptance round.

The frozen model digest remains `176a0b027af87d11fb14a44a843ed50b8e068e4284ee8096e4daf34009c55ea5`. Historical test exposure remains disclosed. Replaying saved parameters establishes preservation; it does not create independent generalization evidence or imply a guaranteed course grade.

## Inspecting the evidence

The committed receipts are in [results/acceptance_20261113](../results/acceptance_20261113/). All previously tracked source, tests, parameters, results and documentation are unchanged. The new manifest covers those files and these acceptance records; earlier manifests remain valid.

From the repository root:

```bash
python -m depthlab.verify
python -m depthlab.verify --manifest results/acceptance_20261113/manifest.json
python -m pytest -q -p no:cacheprovider
```

Full runtime artifacts remain outside Git under `outputs/acceptance_20261113/` on the server (NAS-backed). They include the source archive, stereo arrays, complete replay table, and the one-time acceptance replay script whose digest is recorded in `frozen_oof_replay.json`. The exported source directory is `outputs/acceptance_checkout_20261113/`; it is on local storage because the NAS filesystem does not support the required external-input symlinks.

To repeat only the saved-fold acceptance replay with the pinned server environment:

```bash
/home/ai4s_group/XZG/coursework/CV/Project3/.venv-reproduction/bin/python -B \
  outputs/acceptance_20261113/replay_frozen_oof.py \
  --root outputs/acceptance_checkout_20261113 \
  --cache outputs/restored_20261113/parity_nyu_train \
  --output outputs/acceptance_20261113/oof_recheck
```

No new numerical implementation change is recommended before teammate review within this checked requirement, dataset and environment scope. Future changes should address a concrete defect, requirement change or separately approved experiment.
