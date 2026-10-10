[The native ONNX comparison](native-onnx-comparison-20261010.json) records
pinned PyTorch and native ONNX CPU runs: raw latency samples, artifact size,
cold load, actual fresh-folder setup totals, sampled RSS and parity. CPU
names are generic. Historical Windows reports remain alongside it; their
workloads, runtime versions and dates differ.

These files contain no model weights. The checkpoint directory is outside
the repository; artifact hashes are recorded in the model and native ONNX
manifests in `reports/`.

Reports omit hostnames, absolute local paths, unrelated process metadata and
calibration coefficients. Exact calibration remains in the pinned upstream
checkpoint configuration. Keep the immutable measurement-suite and public
regression-suite results separate: their wording and hashes differ.
