"""Download a pinned ONNX FP32 graph and original tokenizer/config; no weights ship here."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPOSITORY = 'onnx-community/laya-typed-decisions-ONNX'
REVISION = '79c441b22da79eb3727bb01e8452e29a32cfa548'
ARTIFACTS = {
    'onnx/model.onnx': (4_442_939, '08a09f491df9efe9d5cbc96d8c5f22a3369951b58c9c64b568db588ab66c21d4'),
    'onnx/model.onnx_data': (1_685_175_296, '92778ebb6a278de5a7041b8a2f9449a05172264d99d8cf471a878ec50ae04741'),
}


def validate_artifacts(folder: Path, artifacts: dict = ARTIFACTS):
    for name, (size, expected) in artifacts.items():
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or unsafe ONNX artifact: {name}')
        if path.stat().st_size != size:
            raise ValueError(f'ONNX artifact size mismatch: {name}')
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): sha.update(block)
        if sha.hexdigest() != expected:
            raise ValueError(f'ONNX artifact SHA-256 mismatch: {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    # The metadata must match the original reference checkpoint, including calibration.
    from scripts.download_model import REPOSITORY as REF_REPO, REVISION as REF_REV, SUBFOLDER, FILES
    graph_dir = Path('.cache/onnx')
    config_dir = Path('.cache/model')
    config_files = [name for name in FILES if name != 'model.safetensors']
    if not args.check_only:
        from huggingface_hub import snapshot_download
        print('Downloading about 1.69 GB of pinned ONNX FP32 artifacts.', flush=True)
        snapshot_download(repo_id=REPOSITORY, revision=REVISION, local_dir=str(graph_dir),
                          allow_patterns=list(ARTIFACTS))
        snapshot_download(repo_id=REF_REPO, revision=REF_REV, local_dir=str(config_dir),
                          allow_patterns=[f'{SUBFOLDER}/{name}' for name in config_files])
    validate_artifacts(graph_dir)
    for name in config_files:
        path = config_dir / SUBFOLDER / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or unsafe reference metadata: {name}')
    print('Pinned ONNX graph, external weights and reference metadata verified.')
    return 0


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
