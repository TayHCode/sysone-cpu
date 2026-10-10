"""Fetch the recorded upstream checkpoint; never implement model inference."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPOSITORY = "convaiinnovations/laya"
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
SUBFOLDER = "typed-decisions"
WEIGHT_BYTES = 842_609_220
WEIGHT_SHA256 = "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e"
FILES = (
    "model.safetensors", "rl_agent_config.json", "encoder/config.json",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
)


def validate_model(folder: Path, expected_bytes: int = WEIGHT_BYTES,
                   expected_sha256: str = WEIGHT_SHA256) -> None:
    for name in FILES:
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe checkpoint file: {name}")
    weights = folder / FILES[0]
    if weights.stat().st_size != expected_bytes:
        raise ValueError("Checkpoint weight size mismatch")
    sha = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != expected_sha256:
        raise ValueError("Checkpoint weight SHA-256 mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    destination = Path(".cache/model")
    if not args.check_only:
        from huggingface_hub import snapshot_download
        print("Downloading about 846 MB of public upstream files; no account required.", flush=True)
        snapshot_download(
            repo_id=REPOSITORY, revision=REVISION, local_dir=str(destination),
            allow_patterns=[f"{SUBFOLDER}/{name}" for name in FILES],
        )
    validate_model(destination / SUBFOLDER)
    print("Pinned typed-decisions checkpoint verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
