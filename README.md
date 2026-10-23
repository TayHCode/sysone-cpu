# sysone-cpu

Run an existing System-One-style decision model on CPU with upstream
[Laya](https://github.com/NandhaKishorM/laya). This recipe and measurement kit
selects a pinned English typed-decisions checkpoint. **PyTorch FP32 is the
default:** it works on all three measured platforms, is fastest on Apple
silicon in the best-thread comparison, and favors larger batches on the
measured Ryzen desktop. ONNX Runtime FP32 is an optional route for lower
latency with one or a few questions on x86 Windows/Linux.

sysone-cpu is an independent project, not affiliated with or endorsed by the
Laya authors. No server, background service or model weights ship here.

## Quickstart: PyTorch FP32

Prerequisites: a supported 64-bit CPU OS, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and this repository's files in the current folder. Use Python 3.12, at least
8 GB available RAM and 6 GB free disk for setup; allow 10 GB disk headroom.
Initial access to PyPI, the PyTorch CPU index and Hugging Face is required.
The public checkpoint needs no account; unauthenticated-download warnings are
normal. uv can obtain Python. All packages use binary wheels; no compiler or
model export is required.

### Prepare a stock Windows PC

Use 64-bit Windows and Windows PowerShell 5.1 or later. Start Windows PowerShell
from the Start menu. Python, Git and a compiler are not required beforehand.

1. Download this repository with **Code → Download ZIP**, then extract it and
   open PowerShell in the extracted `sysone-cpu-main` folder. If the repository
   files are already present, use that folder. Git is optional.
2. Install the latest [Microsoft Visual C++ v14 Redistributable for x64](https://aka.ms/vc14/vc_redist.x64.exe).
   Run the downloaded installer, approve its administrator prompt and restart
   Windows if requested. PyTorch needs this runtime even when installed from a
   binary wheel. Finish this installation before continuing.
3. Install uv in this folder. Paste the following commands into PowerShell.
   The full PowerShell path works even when PowerShell is missing from `PATH`.
   The execution-policy override applies only to the installer process.

```powershell
New-Item -ItemType Directory -Force .tools, .scratch | Out-Null
$env:UV_INSTALL_DIR = "$PWD\.tools"
$env:UV_NO_MODIFY_PATH = "1"
Invoke-WebRequest -UseBasicParsing https://astral.sh/uv/install.ps1 -OutFile .scratch/install-uv.ps1
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File .scratch\install-uv.ps1
$env:PATH = "$PWD\.tools;$env:PATH"
uv --version
```

Keep this shell open for the quickstart. If an organization enforces a Group
Policy that blocks scripts, ask its administrator to install uv using the
[official installation options](https://docs.astral.sh/uv/getting-started/installation/).
See [isolated setup](docs/setup.md#keep-everything-local) to keep Python, package
caches and model downloads inside the repository too.

### Run the model

Linux x86-64 and Windows x86-64, **six commands** (PowerShell on Windows):

```sh
uv venv --python 3.12
uv pip install --python .venv --only-binary :all: torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv --only-binary :all: laya==0.3.4 transformers==5.17.0 tokenizers==0.23.3 safetensors==0.8.0 huggingface_hub==1.32.0 numpy==2.5.3 psutil==7.2.2
uv run --no-project --python .venv python -c "import torch; print(torch.ones(1).item())"
```

Continue only if the import check prints `1.0`. If it fails on Windows, install
or repair the Visual C++ runtime above and retry before downloading weights.

```sh
uv run --no-project --python .venv python scripts/download_model.py
uv run --no-project --python .venv python benchmarks/bench_laya_cpu.py --model .cache/model/typed-decisions --threads 4 --warmup 1 --repeats 1 --counts 1,3 --output .cache/first-run.json
```

On Windows, paste these commands into the same PowerShell session used above.
They run uv and Python directly, so virtual-environment activation and an
execution-policy override are not needed for these six commands.
See [Windows execution policies](docs/setup.md#windows-execution-policies)
if you save them as a PowerShell script.

macOS on Apple silicon, **six commands** (native arm64 wheels from PyPI):

```sh
uv venv --python 3.12
uv pip install --python .venv --only-binary :all: torch==2.14.0
uv pip install --python .venv --only-binary :all: laya==0.3.4 transformers==5.17.0 tokenizers==0.23.3 safetensors==0.8.0 huggingface_hub==1.32.0 numpy==2.5.3 psutil==7.2.2
uv run --no-project --python .venv python -c "import torch; print(torch.ones(1).item())"
```

Continue only if the import check prints `1.0`.

```sh
uv run --no-project --python .venv python scripts/download_model.py
uv run --no-project --python .venv python benchmarks/bench_laya_cpu.py --model .cache/model/typed-decisions --threads 4 --warmup 1 --repeats 1 --counts 1,3 --output .cache/first-run.json
```

The pinned safetensors download is about **843 MB**; inference uses FP32 on
CPU, including Apple silicon (MPS and CUDA are not selected). The downloader
checks the pinned revision, size and SHA-256. The final command uses synthetic
ticket questions and saves answers, package versions, latency and sampled
memory to `.cache/first-run.json`. Expect `meta.device` to be `"cpu"`, finite
probabilities and passing batch checks. One timed repeat is a functionality
probe; use at least ten for measurements.

Read [setup](docs/setup.md) for isolation and cleanup,
[benchmarks](docs/benchmarks.md) for the comparison method and results,
and [how it works](docs/how-it-works.md) for the architecture.

## Choosing threads

The quickstart's `--threads 4` is a starting point, not a universal optimum.
Thread count depends on the CPU and number of questions in each request.
These are the best observed PyTorch counts in the indicated grids:

| CPU | Measured threads | 1 question | 3 questions | 16 questions | 64 questions |
| --- | --- | ---: | ---: | ---: | ---: |
| Ryzen 9 7950X, 16 cores | 1, 2, 4, 6, 8, 12, 16, 24, 32 | 8 | 12 | 12 | 24 |
| Apple M1 Pro | 4, 8, 10 | 4 | 4 | 8 | Not measured |
| Core i7-10700KF | 1, 2, 4 | 4 | 4 | 4 | Not measured at 4 |

On the Ryzen, those choices improved median latency over four threads by
1.63×, 2.21×, 2.81× and 2.49× respectively. More threads can help desktop chips
with many cores: try 8–12 for small and medium requests. On the eight-core
Intel, four was best among the completed 1/2/4-thread points; higher counts
were not covered in that sweep. Apple results above come from the separate
FP32 backend comparison, not a full thread sweep. These observations use a
synthetic workload and one run per point under ambient load.

Measure your own request sizes with the bench script, one process at a time:

```sh
uv run --no-project --python .venv python benchmarks/bench_laya_cpu.py --model .cache/model/typed-decisions --threads 4 --warmup 3 --repeats 10 --counts 1,3,16 --output .cache/pytorch-t4.json
```

Repeat in fresh processes with `--threads 8` and `--threads 12`, using different
output files. Compare whole-request medians. Do not transfer PyTorch thread
choices to ONNX without measuring that backend.

## Optional: ONNX Runtime FP32

ONNX FP32 had lower latency for one or a few questions on x86 Windows/Linux.
It was not faster on Apple silicon or for large Ryzen batches in the
best-thread comparison. At four threads on Core i7-10700KF it was 1.28×/1.15×
faster for one/three questions, but slower for sixteen. With each backend's
best observed count on Ryzen it was 1.62×/1.14× faster for one/three and
0.84× as fast for sixteen. On Apple M1 Pro the corresponding ratios were
0.98×/0.59×/0.33×. See the [measured tables](docs/benchmarks.md) for scope
and the separate Core i7 six-thread result.

Use a separate environment to retain the `laya==0.3.4` PyTorch reference.
Linux/Windows commands (on Apple silicon omit the CPU index in command two):

```sh
uv venv --python 3.12 .venv-onnx
uv pip install --python .venv-onnx --only-binary :all: torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-onnx --only-binary :all: "laya[onnx]==0.4.2" transformers==5.17.0 tokenizers==0.23.3 safetensors==0.8.0 huggingface_hub==1.32.0 numpy==2.5.3 psutil==7.2.2 onnxruntime==1.31.0 onnx==1.23.2 onnxscript==0.7.2 flatbuffers==25.12.19 protobuf==7.36.2
uv run --no-project --python .venv-onnx python scripts/download_onnx.py
uv run --no-project --python .venv-onnx python benchmarks/bench_laya_ort.py --threads 4 --warmup 1 --repeats 1 --counts 1,3 --output .cache/onnx-first-run.json
```

This route uses ONNX Runtime's CPU provider; expect `meta.providers` to be
`["CPUExecutionProvider"]`. Laya's ONNX extra still installs PyTorch for shared
helpers. The graph and external weights total **1.69 GB**, plus the original
checkpoint's tokenizer/configuration; both graph files have pinned revisions,
sizes and SHA-256 checks. This route downloads no PyTorch weights itself.
ONNX loaded faster on all three measured CPUs, but startup speed does not
establish lower warm request latency. Run the parity command below before
using a different graph or runtime.

## Check any ONNX build against the reference

The public regression suite has 60 cases and 106 decisions. Its neutral
wording and role-only metadata differ from the immutable measurement suite;
results are tracked by separate hashes. Run the two backends in separate
processes, then compare their saved answers:

```sh
uv run --no-project --python .venv python scripts/parity.py collect --backend pytorch --output .cache/parity-reference.json
uv run --no-project --python .venv-onnx python scripts/parity.py collect --backend onnx --output .cache/parity-candidate.json
uv run --no-project --python .venv python scripts/parity.py compare --reference .cache/parity-reference.json --candidate .cache/parity-candidate.json --output .cache/parity-result.json
```

To check a custom graph, add `--onnx path/to/model.onnx` to the ONNX collection
command; keep its external weights beside it and keep the reference unchanged.
The gate requires **at least 99% same top answer, mean KL at most 0.005, and no
flip with reference confidence at least 0.7**. Failed or incomplete comparisons
return a nonzero exit status. This checks agreement, not labelled accuracy.

## Limitations

English typed-decisions is the only checkpoint selected here. Its configured
context is 1,024 tokens; long input can be truncated. Confidence does not
establish correctness or calibration on your data. A passing synthetic suite
does not guarantee agreement on every future input.

Current dynamic int8 conversions on Core i7-10700KF matched only **72.6–87.7%
of top answers**, with confident flips; all failed the equivalence gate.
The earlier 84–86% results used different conversion/runtime settings. Smaller
files or lower latency alone do not meet the gate. FP16 passed agreement
checks but gave no latency gain over ONNX FP32 in the measured comparison. FP16 and int8 builds are **not recommended**; int8
fails the gate. The quickstart uses PyTorch FP32. Hardware, thread count,
input lengths and other applications affect performance. Larger batches can
favor a different backend or thread count.

## Local checks

These checks need only Python's standard library and do not download a model:

```sh
python3 -m unittest discover -s tests -q
python3 scripts/check_syntax.py
```

The release gate runs these local checks; it does not install the optional
inference environment or download weights. For bounded concurrency verification,
see [formal/README.md](formal/README.md).

Original recipes and harnesses: [MIT](LICENSE), copyright TayHCode.
Upstream Laya runtime/checkpoint: Apache-2.0; ONNX Runtime: MIT.
See [NOTICE](NOTICE) for attribution and the ONNX graph source. Download weights
from upstream; this repository never redistributes them.
