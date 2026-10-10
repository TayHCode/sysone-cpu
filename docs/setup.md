# Setup

Use the five commands in the README from the extracted repository folder.
On Windows start `powershell -NoProfile`; an SSH session may initially be cmd.
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
Invoke-WebRequest https://astral.sh/uv/install.ps1 -OutFile .scratch/install-uv.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .scratch\install-uv.ps1
$env:PATH = "$PWD\.tools;$env:PATH"
```

The PowerShell execution-policy override applies only to the installer process;
it lets that downloaded script run when the default policy blocks `.ps1` files.
It does not change the user's saved execution policy. Group Policy restrictions
may still apply.

The downloader writes only `.cache/model` and verifies the checkpoint hash.
It never silently chooses base-English or multilingual Laya. Runtime installs
use prebuilt wheels. No compiler, GPU driver, ONNX export or service is needed.
If installation offers to compile a package, stop and check platform support.

## Troubleshooting

* A missing `uv` means the local `.tools` folder is not on this shell's PATH.
* Use the CPU wheel index for torch on Linux and Windows. Use PyPI on Apple
  silicon; select `device="cpu"` (the harness does this) instead of MPS.
* Keep the named versions together. The historical ONNX research tools are not needed by this quickstart.
* A checkpoint hash mismatch stops the helper. Remove only your failed download
  and retry; do not substitute another checkpoint or disable verification.
* Low-memory machines may be unable to build the in-memory model. The historical
  Windows run used roughly 2 GiB resident memory; allow additional setup overhead.
* Upstream may normalize tokenizer configuration in the downloaded folder at
  load time. The pinned weight hash remains unchanged.
* Setup requires internet once. For subsequent inference set `HF_HUB_OFFLINE=1`
  and `TRANSFORMERS_OFFLINE=1`; the local checkpoint has the encoder config.
* Start at four threads. For a benchmark use `--warmup 3 --repeats 10` and omit
  `--counts` to measure the full 1/3/4/8/16 suite, one process at a time.

## Cleanup and sharing

After saving the results you need, remove only the `.venv`, `.cache`, `.tools`
and `.scratch` folders created for your throwaway run. Installer scripts and
temporary archives (including Windows `uv.zip`) stay in `.scratch` with the
variables above. Do not delete an existing model cache or scratch folder.
Close this shell after cleanup so its variables no longer point to those folders.
Keep the run below 10 GB; delete it before testing the next machine.
Reports avoid automatic hostnames and absolute paths, but review any report
before sharing it, especially if you replace the synthetic inputs.
