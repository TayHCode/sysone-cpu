# Measurement method and limits

These measurements compare deployment behavior of the pinned English
typed-decisions checkpoint. Synthetic tickets and decision questions measure
latency, repeatability and backend agreement. They do not measure labelled
accuracy or confidence calibration on real data.

## Keep the comparison controlled

Use the same tokenizer, checkpoint configuration, state and questions for every
backend. Record the checkpoint revision, artifact hashes, package versions,
CPU model and thread count. Select CPU explicitly. Run one measurement process
at a time, with a cooldown between points; leave other applications alone and
record their CPU activity. Shared-machine results include that background load.

Measure one, three and sixteen questions per request. After three warmups,
record at least ten timed repeats. Report whole-request p50 and p95 latency,
questions per second and sampled peak resident memory. Keep the raw samples;
one timed repeat is a functionality check, not a performance estimate.

Measure four threads and sweep the useful thread counts on the machine. Select
the best thread count using the lowest single-question median, then run all
three workloads again at that count. A thread count selected for one question
may not be best for a larger batch. Never mix a different checkpoint's timing
into this comparison.

## Load and setup time

Cold load means a new process and runtime session; filesystem caches are not
flushed. Separate imports, model/session load and the first answer. Constructor
work, random initialization, graph optimization and weight copying can make
load time differ substantially between backends.

A fresh-folder first-run total includes Python/environment setup, binary-wheel
installation, model download, loading and the first answer. State whether uv
installation is included. Measure these phases rather than inferring them from
steady-state latency. Network throughput, filesystem caches and competing CPU
work affect setup time. An estimated sum of independently measured phases must
be labelled as a decomposition, not a directly observed wall-clock run.

File sizes refer to the complete graph plus external weights, or the PyTorch
safetensors file, in decimal bytes. Resident memory uses MiB. Sampling can miss
short peaks, so sampled RSS is not an exhaustive memory high-water mark.

## Equivalence gate

Each suite contains at least forty cases and seventy decisions, including
choice, score and binary questions, long inputs, Unicode and varied option
counts. Save the suite hash before running either backend. The immutable measurement
suite and the separately worded public regression suite both contain sixty
cases and 106 decisions; their hashes differ. Keep their results separate. PyTorch FP32 with
`laya==0.3.4` and the verified reference checkpoint remains the reference.

A candidate passes only if all of the following hold:

* At least 99% of top answers match the reference.
* Mean KL divergence is at most 0.005.
* No changed top answer has reference confidence of 0.7 or higher.

Also report mean absolute probability drift and every flip. The public API
rounds probabilities to four decimals; align option labels and normalize
rounded distributions before KL. Preserve a choice's reported label when a
rounded distribution ties. Reject missing cases, questions, options or invalid
probabilities rather than comparing only the remaining intersection.

Use `scripts/parity.py compare --reference reference.json --candidate candidate.json`
to apply this gate to saved answers. The command requires matching suite hashes,
pinned reference provenance and adequate suite coverage; a failed gate returns
a nonzero exit status. A passing synthetic suite is limited evidence of
agreement, not a proof for all inputs or a labelled accuracy score.

## Precision and runtime changes

Evaluate FP32, FP16, dynamic per-tensor int8, per-channel int8 and mixed int8
separately. In a mixed conversion, leave the decision path in FP32 and identify
which encoder weights were quantized. Strip stale ONNX `value_info` before
quantization, then verify the resulting graph and repeat the same parity suite.
An unsupported operator or conversion failure is a result to report.

Earlier int8 measurements found only 84–86% agreement with FP32. Faster or
smaller alone is insufficient for an equivalent replacement. Runtime upgrades
can also change tokenization, truncation, temperature scaling and confidence;
measure their unchanged behavior separately from compatibility adapters or
calibration experiments. Do not silently change the reference to make a
candidate pass.

## Reproduce the reference

After the default PyTorch quickstart, use a local verified checkpoint:

```sh
uv run --no-project --python .venv python benchmarks/bench_laya_cpu.py --model .cache/model/typed-decisions --threads 4 --warmup 3 --repeats 10 --counts 1,3,16 --output .cache/reference-t4.json
```

Repeat in a fresh process for each selected thread count. Save results outside
model/download folders before cleanup. Historical sanitized results remain in
[reports/results](../reports/results/README.md); their workloads, runtime
versions and dates differ, so they must not be combined into a hardware ranking.

## Measured comparison

The 2026-10-10 comparison uses the pinned original safetensors checkpoint and
the pinned community graph with the unchanged published native ONNX runtime.
The immutable measurement suite has 60 cases and 106 decisions. The separately
worded public regression suite is checked by the end-to-end recipe and tracked
by its own hash. [The aggregate JSON](../reports/results/native-onnx-comparison-20261010.json)
contains every four-thread and selected-thread record, all ten latency samples,
throughput, imports, cold load, first-answer time, sampled RSS, background CPU
observations, actual fresh-folder totals and gate outcomes.

Core i7 and Apple thread counts come from one-question pilots; Ryzen uses
eight threads from the independent sweep. These choices
do not identify the optimal count for each larger batch. Cold load includes
the upstream agent constructor and its lazy imports after the initial module
imports. Fresh totals are Core i7-only, one installation per backend, with isolated
Python, wheel and model caches, ending at the same fixed two-question request.
They exclude uv binary preparation and source transfer and do not establish a
setup-time distribution. The earlier ONNX README quickstart included additional
benchmark/batch probes, so its observed total is reported separately.

### Core i7-10700KF

| Backend | Threads | Artifact MB | Cold load s | 1q p50 / p95 ms | Top agreement | Mean KL | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch FP32 | 6 | 842.6 | 22.26 | 244.4 / 251.0 | 100.00% | 7.57476e-11 | PASS |
| ONNX FP32 | 6 | 1689.6 | 5.56 | 144.8 / 153.6 | 100.00% | 6.60807e-06 | PASS |
| ONNX FP16 | 8 | 847.1 | 6.36 | 186.5 / 191.7 | 100.00% | 7.12914e-06 | PASS |
| Dynamic INT8 per tensor (default operators) | 12 | 444.6 | 5.65 | 103.7 / 131.0 | 72.64% | 0.254583 | FAIL |
| Dynamic INT8 per channel (default operators) | 12 | 446.1 | 5.68 | 93.7 / 105.1 | 87.74% | 0.0457322 | FAIL |
| Mixed INT8 encoder / FP32 decision path | 12 | 661.4 | 5.72 | 99.5 / 167.3 | 80.19% | 0.0488856 | FAIL |

Core i7 fresh-folder totals (same selected threads): PyTorch FP32 39.9 s, ONNX FP32 33.5 s, ONNX FP16 28.8 s, Dynamic INT8 per tensor (default operators) 41.6 s, Dynamic INT8 per channel (default operators) 57.4 s, Mixed INT8 encoder / FP32 decision path 55.8 s.

The earlier five-command ONNX README route took 39.1 s. CPU provider, finite answers and batch checks passed; public-suite top agreement was 100.00%, mean KL 6.60807e-06, with 0 confident flips.

### Apple M1 Pro

| Backend | Threads | Artifact MB | Cold load s | 1q p50 / p95 ms | Top agreement | Mean KL | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch FP32 | 2 | 842.6 | 36.74 | 117.5 / 123.5 | 100.00% | 0 | PASS |
| ONNX FP32 | 8 | 1689.6 | 5.15 | 118.8 / 126.5 | 100.00% | 6.60814e-06 | PASS |
### Ryzen 9 7950X

| Backend | Threads | Artifact MB | Cold load s | 1q p50 / p95 ms | Top agreement | Mean KL | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch FP32 | 8 | 842.6 | 16.05 | 132.6 / 138.9 | 100.00% | 7.57476e-11 | PASS |
| ONNX FP32 | 8 | 1689.6 | 5.81 | 83.8 / 87.2 | 100.00% | 6.60814e-06 | PASS |

PyTorch FP32 is the default quickstart; ONNX FP32 is optional. ONNX startup
and steady inference have different
tradeoffs: Apple M1 Pro favors PyTorch for warm batch throughput. Completed
Core i7 FP16 passed parity, but its selected latency was higher than FP32. INT8 storage alone does not establish
equivalence. Per-tensor quantization uses the default ORT dynamic operator set;
per-channel uses the same operator set. Mixed quantization uses an explicit
encoder MatMul/Gemm whitelist, leaving embeddings and all decision operations
in FP32. All conversions strip stale value_info and preserve their weight-range
policy in the report. The Core i7-10700KF per-channel/mixed runs use reduced
range (7-bit weights in INT8 storage), following ORT’s documented concern about
AVX2 saturation. Additional quantization work on Apple M1 Pro and Ryzen 9 7950X was stopped
after the scope was narrowed to FP32. These rows establish the Core i7 outcome
and do not claim cross-platform INT8 equivalence. See
[ONNX Runtime’s quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

Additional full-range MatMul/Gemm-only diagnostics on Core i7-10700KF failed
parity too: per-tensor 73.58% top agreement with eight confident flips, and
per-channel 39.62% with fourteen. These use a different operator policy from
the primary default-operator rows and are not interchangeable measurements.

An independent Ryzen 9 7950X thread sweep selected eight threads for one
question (121.7 ms p50), twelve for three (281.4 ms) and sixteen questions
(1,102.7 ms), and twenty-four for sixty-four (4,890.3 ms). Those are separate
PyTorch measurements with different background activity; they motivate the
eight-thread selection here and do not replace this comparison.

## Workload-specific FP32 comparison

These completed 2026-10-10 points use three warmups, ten repeats, CPU-only
intra-op counts shown below and inter-op one. Latency is the whole request's
p50 in milliseconds. One run per point, synthetic inputs and ambient load
limit generalization; no confidence intervals or universal optimum are claimed.
Windows trials failing the predeclared background-load screen were excluded.
Apple and Windows grids below extend the earlier selected-thread rows; Core
i7 retains the four-thread comparison. Ratios are PyTorch / ONNX (>1 favors
ONNX), using independently best observed counts for each workload in the grid.

| CPU / OS | Questions | PyTorch threads / ms | ONNX threads / ms | ONNX speedup |
| --- | ---: | ---: | ---: | ---: |
| Ryzen 9 7950X / Windows | 1 | 8 / 132.623 | 12 / 82.027 | 1.62× |
| Ryzen 9 7950X / Windows | 3 | 12 / 274.822 | 12 / 240.594 | 1.14× |
| Ryzen 9 7950X / Windows | 16 | 12 / 1150.250 | 12 / 1364.207 | 0.84× |
| Apple M1 Pro / macOS | 1 | 4 / 116.696 | 8 / 118.850 | 0.98× |
| Apple M1 Pro / macOS | 3 | 4 / 223.136 | 8 / 380.063 | 0.59× |
| Apple M1 Pro / macOS | 16 | 8 / 773.398 | 8 / 2314.818 | 0.33× |
| Core i7-10700KF / Linux | 1 | 4 / 256.033 | 4 / 200.119 | 1.28× |
| Core i7-10700KF / Linux | 3 | 4 / 679.946 | 4 / 592.199 | 1.15× |
| Core i7-10700KF / Linux | 16 | 4 / 3488.053 | 4 / 3742.533 | 0.93× |

The Ryzen grid is 4/8/12 threads and the Apple grid is 4/8/10. PyTorch wins
all three Apple best-thread comparisons (one question is near a tie), and
sixteen questions on Ryzen is 1.19× faster with PyTorch. Core i7 six-thread
ONNX was faster even for sixteen questions in the earlier selected-thread run;
this reinforces that backend choice depends on both threads and workload.
Public-suite FP32 parity on Windows and Apple was 100% top agreement, mean
KL 6.60814e-6, zero confident flips, 60 cases / 106 decisions. Core i7 public
quickstart parity was also 100%, mean KL 6.60807e-6, zero confident flips.
The public suite hash is `c58ab537492a9314f192c5ce1140ea0d53683fa1177fc913ebb73a00e7116d5b`.
FP16 gave no latency gain over ONNX FP32 in the measured Core i7 comparison.
Every primary INT8 variant failed the gate (72.64–87.74% same top and confident
flips); FP16 and INT8 builds are not recommended.

## PyTorch thread sweep

This independent synthetic sweep used a fresh Python process per thread count,
three warmups and ten repeats per workload, inter-op one and ambient load.
The winning count is the observed median minimum, not a confidence bound.
Different load conditions explain differences from the FP32 comparison above;
do not combine separate runs into speedups.

| CPU | Questions | Best observed threads | p50 ms | Speedup vs 4 threads |
| --- | ---: | ---: | ---: | ---: |
| Ryzen 9 7950X, 16 cores | 1 | 8 | 121.651 | 1.63× |
| Ryzen 9 7950X, 16 cores | 3 | 12 | 281.429 | 2.21× |
| Ryzen 9 7950X, 16 cores | 16 | 12 | 1102.712 | 2.81× |
| Ryzen 9 7950X, 16 cores | 64 | 24 | 4890.318 | 2.49× |
| Core i7-10700KF | 1 | 4 | 244.210 | 1.00× |
| Core i7-10700KF | 3 | 4 | 620.075 | 1.00× |
| Core i7-10700KF | 16 | 4 | 3228.384 | 1.00× |
| Core i7-10700KF | 64 | 2 | 21482.446 | Not measured |

Ryzen measured 1/2/4/6/8/12/16/24/32 threads. Core i7 completed only 1/2/4
for 1/3/16 questions and 1/2 for 64; its four-thread 64-question point and
higher counts did not complete. No Apple thread-sweep points were measured;
its README guidance comes from the separate FP32 4/8/10 comparison. The
Core i7 four-thread interruption occurred after ten samples for each smaller
workload but before the final batched/single-answer check. Output digests
matched across completed thread counts. Start at four threads, try 8–12 on
many-core desktops, and use the benchmark to choose for your own CPU and
request size. More threads are not always faster.
