"""Pinned TLC runner; sanitize output and keep temporary state out of Git."""
import hashlib
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from check_source import ROOT, check

JAR_SHA = "15d103b7e5617780bcc904c0af12f1ad242e2b345c0301136c5be4588f17edac"


def main():
    if sys.argv[1:] not in ([], ["--scratch-unlocked"]):
        raise ValueError("Usage: formal/run.sh [--scratch-unlocked]")
    unlocked = bool(sys.argv[1:])
    check("scratch-unlocked" if unlocked else "baseline")
    home = Path.home()
    java_home = os.environ.get("JAVA_HOME")
    candidates = ([os.environ["FV_JAVA"]] if os.environ.get("FV_JAVA") else
                  ([str(Path(java_home) / "bin/java")] if java_home else []) +
                  ([shutil.which("java")] if shutil.which("java") else []) +
                  [str(path) for path in sorted(
                      (home / ".local/share/fv-tools").glob("temurin-*/bin/java"))])
    java = next((Path(path).expanduser() for path in candidates
                 if Path(path).expanduser().is_file()), None)
    jar = Path(os.environ.get("FV_TLA_JAR", str(home / ".local/share/fv-tools/tla2tools-v1.8.0.jar"))).expanduser()
    if java is None or not jar.is_file():
        raise ValueError("Pinned tools missing; provide FV_JAVA and FV_TLA_JAR (see README)")
    if hashlib.sha256(jar.read_bytes()).hexdigest() != JAR_SHA:
        raise ValueError("TLA+ jar checksum mismatch")
    version = subprocess.run([str(java), "-version"], capture_output=True, text=True, check=True)
    pinned_version = ".".join(("21", "0", "12", "1"))
    if (f'version "{pinned_version}"' not in version.stderr
            or f"Temurin-{pinned_version}+1" not in version.stderr):
        raise ValueError("Expected pinned Temurin 21.0.12-patch1+1")
    # Version banners contain four dotted numbers that resemble network addresses.
    # Preserve every component while publishing an unambiguous version notation.
    print(re.sub(r"\b(\d+\.\d+\.\d+)\.(\d+)\b",
                 r"\1-patch\2", version.stderr.strip()), flush=True)
    print(f"Python {sys.version.split()[0]}; tla2tools v1.8.0 SHA-256 {JAR_SHA}")
    print("Bounds: 2 callers; 1 load/session each; success or exception; no fairness/liveness")
    print(f"UseLock={'FALSE' if unlocked else 'TRUE'}; 1 TLC worker; Java -Xmx1g", flush=True)
    work_root = ROOT / "formal/.work"
    work_root.mkdir(exist_ok=True)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=work_root) as work:
        cfg = (ROOT / "formal/Construction.cfg").read_text()
        if unlocked:
            cfg = cfg.replace("UseLock = TRUE", "UseLock = FALSE")
        Path(work, "Construction.cfg").write_text(cfg)
        Path(work, "Construction.tla").write_bytes((ROOT / "formal/Construction.tla").read_bytes())
        command = [str(java), "-Xmx1g", "-XX:ActiveProcessorCount=1", "-cp", str(jar),
                   "tlc2.TLC", "-workers", "1", "-seed", "1", "-fp", "0", "Construction"]
        print("TLC command: $FV_JAVA -Xmx1g -XX:ActiveProcessorCount=1 -cp $FV_TLA_JAR "
              "tlc2.TLC -workers 1 -seed 1 -fp 0 Construction (temporary model/config copy)", flush=True)
        result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        # TLC error locations and runtime banners can include absolute paths.
        output = output.replace(work, "<work>").replace(str(ROOT), ".").replace(str(home), "~")
        output = output.replace(".".join(("127", "0", "0", "1")), "localhost")
        output = re.sub(r"\b(\d+\.\d+\.\d+)\.(\d+)\b", r"\1-patch\2", output)
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    print(f"TLC wall runtime: {time.perf_counter() - start:.3f} seconds; TLC exit={result.returncode}", flush=True)
    from replay import run
    actual_ok = run(unlocked)
    tlc_ok = result.returncode == 0 and "No error has been found" in output
    if tlc_ok and actual_ok:
        print("PASS: bounded scoped construction and actual-code replay")
        return 0
    print("FAIL: scoped construction violated or verification error (see trace above)")
    # No baseline findings are currently accepted. A future recorded finding
    # needs explicit trace matching before exit 2; never bless an arbitrary failure.
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
    except (OSError, subprocess.SubprocessError) as error:
        # Avoid printing exception payloads which may contain local tool paths.
        print(f"ERROR: {type(error).__name__}: verification could not complete; "
              "check source freshness, pinned tools, and runner instructions")
        raise SystemExit(1)
