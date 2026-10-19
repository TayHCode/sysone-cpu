"""Refuse source/model drift; recognize exactly one documented scratch mutation."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "formal/guarded.json"
LOCK = b"with _CONSTRUCTION_LOCK:"
UNLOCKED = b"with __import__('contextlib').nullcontext():"


def check(mode="baseline"):
    guards = json.loads(GUARD.read_text())
    if set(guards) != {"scripts/onnx_runtime.py"}:
        raise ValueError("Unexpected guarded-file set; review the model and skill")
    for name, expected in guards.items():
        data = (ROOT / name).read_bytes()
        if mode == "scratch-unlocked":
            if data.count(UNLOCKED) != 1 or LOCK in data:
                raise ValueError("Scratch mode requires exactly the documented product mutation")
            # Reconstruct only that single edit, then compare the ENTIRE source.
            data = data.replace(UNLOCKED, LOCK)
        elif mode != "baseline":
            raise ValueError("Unknown verification mode")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source drift: {name}; follow verify-onnx-construction skill")
    print(f"Source/model guard: matched ({mode})")


if __name__ == "__main__":
    try:
        check(sys.argv[1] if len(sys.argv) == 2 else "baseline")
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
