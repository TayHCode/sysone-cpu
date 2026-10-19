"""Single supported product mutation, model mirror, evidence, and exact restore.

Run only on an unchanged guarded source. Restores the captured bytes even if a
check fails. This task explicitly authorizes the temporary product-code edit.
"""
import difflib
import subprocess

from check_source import ROOT, LOCK, UNLOCKED, check


def capture(name, args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    log = "$ " + " ".join(args) + "\n" + result.stdout + result.stderr
    log += f"\nRunner exit: {result.returncode}\n"
    (ROOT / "formal/evidence" / name).write_text(log)
    print(f"{name}: exit {result.returncode}", flush=True)
    return result


def main():
    check()
    source = ROOT / "scripts/onnx_runtime.py"
    original = source.read_bytes()
    if original.count(LOCK) != 1:
        raise ValueError("Expected exactly one construction lock")
    mutated = original.replace(LOCK, UNLOCKED)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True),
                                      mutated.decode().splitlines(True),
                                      fromfile="a/scripts/onnx_runtime.py",
                                      tofile="b/scripts/onnx_runtime.py", n=0))
    (ROOT / "formal/evidence/seeded-change.patch").write_text(patch)
    try:
        source.write_bytes(mutated)
        guard = capture("source-drift.log", ["formal/run.sh"])
        if guard.returncode != 1 or "Source drift:" not in guard.stdout:
            raise ValueError("Freshness guard failed to reject mutation")
        broken = capture("seeded-failing.log", ["formal/run.sh", "--scratch-unlocked"])
        if (broken.returncode != 1 or "Invariant ScopedConstruction is violated" not in broken.stdout
                or "original_restored=False" not in broken.stdout
                or "B=0 (requested 4)" not in broken.stdout):
            raise ValueError("Seed did not produce both TLC and actual-product counterexamples")
    finally:
        # Never discard unrelated edits made while the scratch run is in progress.
        if source.read_bytes() != mutated:
            raise ValueError("Source changed during mutation; preserve it and restore manually from Git")
        source.write_bytes(original)
    restored = capture("reverted.log", ["formal/run.sh"])
    if restored.returncode != 0 or source.read_bytes() != original:
        raise ValueError("Restored source did not return to clean baseline")
    print("Seed confirmed; original product bytes restored; baseline PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
