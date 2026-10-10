# sysone-cpu

Run an existing System-One-style decision model on a CPU, using upstream
[Laya](https://github.com/NandhaKishorM/laya). This small recipe and measurement
kit explains a previously used Windows CPU setup and lets you try its pinned
English typed-decisions checkpoint on Linux, macOS or Windows.

sysone-cpu is an independent project, not affiliated with or endorsed by the
Laya authors. The name identifies this recipe kit; the upstream runtime package
and its imports remain `laya`.

It is not a new inference engine, a chat model, or a claim of novel CPU support.
No server, background service or model weights ship here.

## Quickstart

Prerequisites: a supported 64-bit CPU OS, [uv](https://docs.astral.sh/uv/getting-started/installation/)
and this repository's files in your current folder. Use Python 3.12, at least
4 GB available RAM and about 6 GB free disk for setup (allow 10 GB headroom).
Network access to PyPI, the PyTorch CPU index and Hugging Face is needed initially;
the public checkpoint requires no login. Hugging Face may warn about
unauthenticated requests; downloads still work, and `HF_TOKEN` is optional for
higher rate limits. uv can obtain Python.

Linux x86-64 and Windows x86-64, **five commands** (PowerShell on Windows):

```sh
uv venv --python 3.12
uv pip install --python .venv --only-binary :all: torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv --only-binary :all: laya==0.3.4 transformers==5.17.0 safetensors==0.8.0 huggingface_hub==1.32.0 numpy==2.5.3 psutil==7.2.2
uv run --no-project --python .venv python scripts/download_model.py
uv run --no-project --python .venv python benchmarks/bench_laya_cpu.py --model .cache/model/typed-decisions --threads 4 --warmup 1 --repeats 1 --counts 1,3 --output .cache/first-run.json
```

macOS on Apple silicon, **five commands** (the native PyPI torch wheel; CPU is
still explicitly selected, so neither Metal nor CUDA is used):

```sh
uv venv --python 3.12
uv pip install --python .venv --only-binary :all: torch==2.14.0
uv pip install --python .venv --only-binary :all: laya==0.3.4 transformers==5.17.0 safetensors==0.8.0 huggingface_hub==1.32.0 numpy==2.5.3 psutil==7.2.2
uv run --no-project --python .venv python scripts/download_model.py
uv run --no-project --python .venv python benchmarks/bench_laya_cpu.py --model .cache/model/typed-decisions --threads 4 --warmup 1 --repeats 1 --counts 1,3 --output .cache/first-run.json
```

The download is about 846 MB. The helper pins its upstream revision and checks
the 842,609,220-byte safetensors file against its recorded SHA-256. The final
command runs only the existing upstream model on synthetic ticket questions,
prints timings and saves answers/versions to `.cache/first-run.json`. Expect
`meta.device` to be `"cpu"`, `batch_equivalence.same_public_answers` to be `true`,
and `fixture.answers` to contain finite numbers in that JSON file. One timed
repeat is a functionality probe, not a useful performance estimate.

Read [setup](docs/setup.md) for isolated tool installation, cleanup and snags;
[how it works](docs/how-it-works.md) for the architecture and other runtimes;
[benchmarks](docs/benchmarks.md) for the measured evidence; and
[the existing runtime](docs/existing-runtime.md) for the original CPU setup.

## With an existing Python 3.12 virtual environment

If a Python 3.12 virtual environment already exists at `.venv` in this folder,
skip `uv venv` and use these four commands for the same Linux/Windows first run.
On Apple silicon omit the CPU index from the first command.

```sh
uv pip install --python .venv --only-binary :all: torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv --only-binary :all: laya==0.3.4 transformers==5.17.0 safetensors==0.8.0 huggingface_hub==1.32.0 numpy==2.5.3 psutil==7.2.2
uv run --no-project --python .venv python scripts/download_model.py
uv run --no-project --python .venv python benchmarks/bench_laya_cpu.py --model .cache/model/typed-decisions --threads 4 --warmup 1 --repeats 1 --counts 1,3 --output .cache/first-run.json
```

## Limitations

English typed-decisions is the only checkpoint this quickstart selects. Its
configured context is 1,024 tokens; long input can be truncated. Confidence is
not a guarantee of correctness or calibration on your data. The synthetic
fixtures measure function, repeatability and backend agreement, not accuracy.
For the measured dynamic int8 conversion, only **84–86% of top answers agreed**
with FP32. It is not an equivalent drop-in replacement and is not installed here.
Hardware, thread count, input lengths and competing work affect timings.

## Local checks

Use the Python 3.12 `.venv` created above; activation and a system `python`
command are unnecessary. The checks need only Python's standard library. They
validate the helper's integrity behavior and Python syntax; the model smoke
test is separate.

```sh
uv run --no-project --python .venv python -m unittest discover -s tests -q
uv run --no-project --python .venv python scripts/check_syntax.py
```

Original recipes and harnesses: [MIT](LICENSE), copyright TayHCode.
Upstream runtime and checkpoint: Apache-2.0, see [NOTICE](NOTICE). Download
weights from their author; this repository never redistributes them.
