These JSON files are measured evidence from the Windows host. They contain no
model weights. The checkpoint directory is outside the repository and its
SHA-256 is recorded in `reports/model-manifest.json`.

Reports omit host labels, absolute local paths, unrelated workload
metadata and calibration coefficients. Exact calibration remains in the pinned
upstream checkpoint config. Measured timings and answers are unchanged.
