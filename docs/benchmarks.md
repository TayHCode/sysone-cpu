# Benchmarks and limits

Historical measurements: 2026-09-21, Windows 11, Ryzen 9 7950X (16 cores /
32 threads), 64 GiB RAM; CPU only. Python 3.12.10, torch 2.14.0+cpu,
Transformers 5.17.0, NumPy 2.5.3, ONNX Runtime 1.24.3. Source revisions and
artifact SHA-256 values are in the [manifests](../reports/model-manifest.json).
The fixture is synthetic, with 3 warm-ups and 10 timed repeats per workload.

| Runtime / typed-decisions | Threads | Cold load | One question p50 | 3 questions p50 | 16 questions p50 |
|---|---:|---:|---:|---:|---:|
| PyTorch | 16 | 19.76 s | 149.6 ms | 323.6 ms | 1,456.8 ms |
| PyTorch | 4 | 19.12 s | 277.5 ms | 919.6 ms | 4,825 ms |
| ONNX FP32 | 4 | 1.10 s | 131.2 ms | 907.7 ms | 5,142 ms |

[Raw PyTorch 16-thread report](../reports/results/laya-typed-decisions-cpu.json),
[4-thread report](../reports/results/laya-typed-decisions-cpu-4threads.json),
[ONNX report](../reports/results/laya-typed-decisions-onnx-cpu-4threads.json).
PyTorch's 16-thread peak RSS rose from 1,989 to 2,113 MiB across these workloads;
ONNX's observed peak was about 1,956 MiB. RSS is sampled, not an exhaustive
high-water mark. Cold load depends on filesystem cache.

The [backend comparison](../reports/results/laya-typed-decisions-parity.json)
found maximum absolute logit error 5.96e-6 and action-probability error zero,
with equal rounded public answers after ignoring a presentation-only score
legend. This is limited case coverage, not proof of agreement on every input.
The published base-English ONNX run is retained and explicitly labelled as a
different checkpoint; its numbers must not be mixed with typed-decisions.

## Dynamic int8 experiment (2026-10-10)

Separate single-run shared-machine experiment, same CPU and ORT 1.24.3;
40 hand-written cases / 70 decisions, no ground-truth labels. These workloads
are different from the September suite, so compare variants within this table.

| Variant | Size | Cold session, 4 threads | One question p50, 4 threads | Top-answer agreement with FP32 |
|---|---:|---:|---:|---:|
| FP32 | 1,611 MB | 4.62 s | 122.5 ms | reference |
| Dynamic int8 per-tensor | 406 MB | 0.82 s | about 48 ms | 59/70 = 84.3% |
| Dynamic int8 per-channel | 407 MB | about 0.8 s | about 49 ms | 60/70 = 85.7% |

[Sanitized raw int8 report](../reports/results/int8-experiment.json).
Mean absolute probability drift was 0.0759 / about 0.079; maximum drift about
0.325 / 0.327. Roughly one top answer in seven changed. **84–86% agreement is
not parity or labelled accuracy.** This conversion is not the quickstart and
is not offered as an equivalent replacement. Other quantizers and runtimes
have not been assessed here.

## Fresh-folder setup tests

Executed 2026-10-10 in three fresh folders with local uv 0.13.0 and managed
Python 3.12.15, prebuilt wheels and the pinned, verified 846 MB checkpoint.
Tools, managed Python, caches and model downloads were isolated to the test
folder. Tests ran sequentially and explicitly selected CPU/four threads.

| Platform / CPU | Five-command setup | Cold model load | One question / three questions | Peak folder size |
|---|---:|---:|---:|---:|
| Linux x86-64 / Intel Core i7-10700KF | 108 s | 22.95 s | 280.1 / 643.2 ms | 2.0 GiB |
| macOS arm64 / Apple M1 Pro | 89 s | 40.75 s | 134.1 / 263.4 ms | 2.3 GiB |
| Windows 11 x86-64 / Ryzen 9 7950X | 56.5 s | 17.86 s | 202.3 / 622.9 ms | 2.14 GiB |

Setup timings cover the five README commands after installing uv; file transfer
and uv installation are excluded. Windows including its uv installer took
57.6 s. An earlier fresh Linux probe took 66 s with 259.1 / 641.9 ms answers;
the final run spent about 60 s downloading the checkpoint. Network time and
shared-machine load vary. These are single-repeat functionality probes, not
comparable performance rankings or throughput estimates.

Repeated probes varied from 20.6 to 41.6 s for Linux cold load and from about
132 to 243 ms for macOS one-question latency. Filesystem caches and competing
work can change these numbers even on the same hardware.

All runs verified the exact weight hash, used `device: cpu`, produced finite
fixture answers, and had `same_public_answers: true`. The Linux/Windows wheel
is torch 2.14.0+cpu; Apple silicon uses torch 2.14.0 from PyPI on CPU.
[Linux result](../reports/results/quickstart-linux.json),
[macOS result](../reports/results/quickstart-mac.json),
[Windows result](../reports/results/quickstart-windows.json).
Reports omit redundant calibration coefficients; get those from the pinned
checkpoint. Big downloads, isolated tools and environments were removed after
saving the reports; each folder stayed below 3 GiB.
