# Setup

Use the six commands in the README from the extracted repository folder.
On Windows open Windows PowerShell from the Start menu; an SSH session may initially be cmd.
Complete the [Windows prerequisites](../README.md#prepare-a-stock-windows-pc) first.
On Linux use x86-64; on macOS use native arm64 on Apple silicon. These are the
platforms targeted here, not a promise for every CPU or Python version.

## Keep everything local

uv normally stores tools and downloads outside the checkout. To isolate a run,
set the local cache and temporary directories before installing uv into `.tools`.
Run these commands in the same shell as the README quickstart.
See the [uv installer options](https://docs.astral.sh/uv/getting-started/installation/).
Review the installer before executing it. It is an upstream binary installer,
not a runtime build.

POSIX shell:

```sh
mkdir -p .tools .scratch
export TMPDIR="$PWD/.scratch" UV_PYTHON_PREFERENCE=only-managed
export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" HF_HOME="$PWD/.cache/huggingface"
curl -LsSf https://astral.sh/uv/install.sh -o .scratch/install-uv.sh
UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh .scratch/install-uv.sh
export PATH="$PWD/.tools:$PATH"
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force .tools, .scratch | Out-Null
$env:TEMP = "$PWD\.scratch"; $env:TMP = "$PWD\.scratch"; $env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_CACHE_DIR = "$PWD\.cache\uv"; $env:UV_PYTHON_INSTALL_DIR = "$PWD\.tools\python"; $env:HF_HOME = "$PWD\.cache\huggingface"
$env:UV_INSTALL_DIR = "$PWD\.tools"; $env:UV_NO_MODIFY_PATH = "1"
Invoke-WebRequest -UseBasicParsing https://astral.sh/uv/install.ps1 -OutFile .scratch/install-uv.ps1
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File .scratch\install-uv.ps1
$env:PATH = "$PWD\.tools;$env:PATH"
```

## Windows execution policies

The setup commands and the six-command PyTorch quickstart were tested in fresh
folders under process-scope `Restricted`, `AllSigned` and `Bypass`. The inline
directory, environment-variable and download commands work under all three.
The README's uv/Python commands run native programs and also work under all
three; this applies to the optional ONNX commands' invocation too.

| Process policy | Downloaded unsigned installer `.ps1` | Unsigned quickstart saved as `.ps1` | Commands pasted at the prompt |
| --- | --- | --- | --- |
| Restricted | blocked | blocked | work |
| AllSigned | blocked | blocked | work |
| Bypass | runs | runs | work |

For the downloaded installer under `Restricted` or `AllSigned`, use the exact
invocation in the isolation block above:

```powershell
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File .scratch\install-uv.ps1
```

After adding `.tools` to PATH, paste the README commands into the same shell.
Virtual-environment activation with `.venv\Scripts\Activate.ps1` is not needed:
every uv command selects its environment with `--python`. The activation hint
printed by `uv venv` does not add a required setup step.

If you save commands as an unsigned script, that script also needs Bypass under
`Restricted` or `AllSigned`. For example, after saving the README commands to
`.scratch\quickstart.ps1`, run:

```powershell
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File .scratch\quickstart.ps1
```

Each override applies only to that child PowerShell process. The calling shell
keeps its policy, and no saved execution policy changes. Group Policy can
override process policy; that case was not tested. A signed script trusted by
`AllSigned` was not part of these unsigned-script checks.

The ONNX downloader writes only `.cache/onnx` and `.cache/model`; it verifies
both the graph and external-weight hashes. The PyTorch downloader fetches
the separately verified safetensors file.
It never silently chooses base-English or multilingual Laya. Runtime installs
use prebuilt wheels. No compiler, GPU driver, ONNX export or service is needed.
If installation offers to compile a package, stop and check platform support.

## Troubleshooting

* On Windows, `WinError 126` while importing torch can mean a missing Visual C++
  runtime, including `VCRUNTIME140_THREADS.dll`. Install or repair the latest
  x64 runtime linked in the README, restart if requested, and retry.
* A missing `uv` means the local `.tools` folder is not on this shell's PATH.
* Use the CPU wheel index for torch on Linux and Windows. Use PyPI on Apple
  silicon; select `device="cpu"` (the harness does this) instead of MPS.
* Keep the named versions together. The ONNX quickstart uses the published `laya[onnx]==0.4.2` extra; keep
  its ORT, ONNX and onnxscript pins together. The default PyTorch route requires
  `laya==0.3.4` in `.venv`; keep the optional ONNX route in `.venv-onnx`.
* A checkpoint hash mismatch stops the helper. Remove only your failed download
  and retry; do not substitute another checkpoint or disable verification.
* Low-memory machines may be unable to build the in-memory model. Both runtimes
  need additional memory during construction/optimization beyond steady-state RSS.
* Upstream may normalize tokenizer configuration in the downloaded folder at
  load time. The pinned weight hash remains unchanged.
* Setup requires internet once. For subsequent inference set `HF_HUB_OFFLINE=1`
  and `TRANSFORMERS_OFFLINE=1`; the local checkpoint has the encoder config.
* Start at four threads and use the README thread guidance for your CPU. For a
  benchmark use `--warmup 3 --repeats 10 --counts 1,3,16` with
  `benchmarks/bench_laya_cpu.py` (or `bench_laya_ort.py` for ONNX), one process
  at a time.

## Cleanup and sharing

After saving the results you need, remove only the `.venv`, `.venv-onnx`, `.cache`, `.tools`
and `.scratch` folders created for your throwaway run. Installer scripts and
temporary archives (including Windows `uv.zip`) stay in `.scratch` with the
variables above. Do not delete an existing model cache or scratch folder.
Close this shell after cleanup so its variables no longer point to those folders.
Keep the run below 10 GB; delete it before testing the next machine.
Reports avoid automatic hostnames and absolute paths, but review any report
before sharing it, especially if you replace the synthetic inputs.
