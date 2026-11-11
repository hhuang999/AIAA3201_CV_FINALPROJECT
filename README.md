# Stereo and monocular depth

AIAA 3201 course project on stereo geometry, RGB-only metric depth, video depth consistency, and training-only depth calibration.

## Stereo geometry

Use the 15 Middlebury quarter-resolution training scenes, including `im0.png`, `im1.png`, `calib.txt`, and `disp0GT.pfm` under `MiddEval3/trainingQ/`.

```bash
python -m depthlab.stereo --data-root data --output results
python -m pytest
```

Three settings compare unpadded 3-way SGBM, reflected-padding 3-way SGBM, and reflected-padding HH4. Matching uses grayscale images, a five-pixel block, and a disparity range rounded up to a multiple of 16. Both directions use the same matcher and padding. Predicted disparities must be finite, positive, inside the original corresponding image, and agree within one pixel with the reverse prediction sampled at the nearest column. Padding supplies context without making out-of-image matches valid.

Disparity MAE is computed on the intersection of valid predictions and finite positive ground truth. Coverage divides that intersection by the ground-truth valid count. Scene scores are averaged equally. Depth uses quarter-resolution calibration, `focal * baseline_m / (disparity + doffs)`. Invalid depths are NaN.

## Sources

- [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/submit3/).
- Hirschmüller, *Stereo Processing by Semiglobal Matching and Mutual Information*, TPAMI 2008.
- [OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html).
