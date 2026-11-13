# Final code review and migration provenance

## Implementation lineage

The restoration commit `b3bbb148d0c1a8930e38a2159b6d55bfca90d1a1` descends directly from the teammate repository at `411687c4cce2a0c8be021f188c7ae9ef6b85de56`. All 13 original `depthlab` modules and the established stereo/inference/calibration entrypoints remain. Eight modules were added for cache identity, frozen deployment, reproducible studies, learning curves, analysis and verification. Production modules do not import the old project application modules. Existing datasets, upstream model checkouts, checkpoints and the validated environment are reused as external inputs.

This is a migration within the teammate architecture, not a claim that every formula was independently rewritten. Previously validated numerical operations were deliberately ported where required for exact behavior. The frozen model and selected historical CSV/JSON/Markdown evidence were copied intact, with hashes identifying their origin. New cache, identity and recovery checks surround those operations.

## Final repairs

- Training entrypoints verify the cache dataset/split hashes, official 795/654 image identities, 249 training scenes and train/test scene separation before sampling.
- Evaluation checks an explicitly requested parameter digest for both parameter formats. Completed new fits must identify the official training images and retain train-only data provenance.
- Learning-curve reference comparison rejects nonfinite scores, duplicates, missing images and inconsistent scene/fold assignments before a completion marker can be published.
- DAV2 can run explicitly on CPU even when CUDA or MPS is available. Only the non-CUDA input transfer is adapted; it uses the same pinned upstream transformations. The existing CUDA method remains untouched.
- A previously decoded frame cache is reusable after a downstream interruption only when source, decoding options, dimensions and RGB checksum agree. Verified reuse does not rewrite the frames or metadata.
- Baseline and evaluation scripts share `outputs/reproduction` and `results/reproduction` defaults, preserving historical root-level results.
- Historical summary/figure helpers validate their required inputs and README markers before writing. They remain outside the frozen numerical route; no current coursework presentation was generated.

## Acceptance evidence

The complete suite passed **174 tests**, including **67 new regression cases**. The recorded 15 CLI help checks, shell syntax checks and installed dependency checks passed. A real one-frame DAV2 CPU inference passed. The actual 70-frame cache was reused with unchanged contents and modification times. Official training identities and the original full-fraction learning reference also passed the new checks.

Fixed-parameter replay on all **654** official test images reproduced the original residual depth maps with **zero changed pixels**. The **2,616** per-image metric rows differed by at most `4.440892098500626e-16`, consistent with CSV float round trips. Residual metrics remain AbsRel **0.09960689343231784**, RMSE **0.3831236302386158 m**, delta1 **0.9055456208197196**. This is preservation evidence, not a new optimization or independent holdout result.

Nine numerical modules remain byte-identical to the restoration commit, and all 12 inspected calibration math/sampling helpers have identical Python syntax trees. The frozen parameter SHA remains `176a0b027af87d11fb14a44a843ed50b8e068e4284ee8096e4daf34009c55ea5`. The earlier complete Part 1/2 restoration evidence remains in `results/restoration_20261113/`.

Machine-readable records are in `results/final_review_20261113/`. Raw runtime arrays and complete replay tables remain under `outputs/final_review_20261113/` on NAS. Run `python -m depthlab.verify` for the current delivery manifest. The earlier restoration manifest describes its original commit and is retained unchanged as historical evidence.

Coursework report, existing figure files and final video artifacts were not modified. The code is ready to freeze within the checked course-data and pinned-environment scope; presentation integration remains a separate authorized task.
