"""Collect backend answers separately and check equivalence to PyTorch FP32."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


GATE = {'same_top_pct_min': 99.0, 'mean_kl_max': 0.005,
        'reference_confidence_for_no_flips': 0.7}


def distribution(answer: dict, keys: list[str] | None = None):
    kind = answer.get('type')
    if kind == 'noul':
        yes = answer.get('noul')
        if not isinstance(yes, (float, int)) or not math.isfinite(yes) or not 0 <= yes <= 1:
            raise ValueError('Invalid binary probability')
        return ['false', 'true'], [1 - yes, yes]
    if kind not in ('choice', 'score') or not isinstance(answer.get('probabilities'), dict):
        raise ValueError('Missing typed probability distribution')
    probs = answer['probabilities']
    if not probs or (keys is not None and set(keys) != set(probs)):
        raise ValueError('Option keys differ or distribution is empty')
    labels = keys if keys is not None else list(probs)
    values = [probs[key] for key in labels]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1 for v in values):
        raise ValueError('Invalid probability')
    total = sum(values)
    # The public API rounds to 4 decimals; allow only rounding-scale mass error.
    if total <= 0 or abs(total - 1) > len(values) * .000051:
        raise ValueError('Invalid probability mass')
    return labels, [v / total for v in values]


def top(answer, labels, values):
    if answer['type'] == 'choice':
        label = answer.get('choice')
        if label not in labels or values[labels.index(label)] < max(values) - 1e-12:
            raise ValueError('Choice disagrees with probabilities')
        return label
    return labels[values.index(max(values))]


def compare_answers(reference: list[dict], candidate: list[dict]) -> dict:
    if not reference or len(reference) != len(candidate):
        raise ValueError('Case counts differ or suite is empty')
    agree = decisions = 0
    drifts, kls, flips = [], [], []
    for index, (ref, actual) in enumerate(zip(reference, candidate)):
        if not ref or set(ref) != set(actual):
            raise ValueError('Question keys differ or case is empty')
        for qid, answer in ref.items():
            other = actual[qid]
            if answer.get('type') != other.get('type'):
                raise ValueError('Answer types differ')
            labels, p = distribution(answer)
            _, q = distribution(other, labels)
            confidence = answer.get('confidence')
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid reference confidence')
            ref_top, actual_top = top(answer, labels, p), top(other, labels, q)
            same = ref_top == actual_top
            agree += int(same)
            decisions += 1
            drifts.extend(abs(a - b) for a, b in zip(p, q))
            kls.append(sum(a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(p, q)))
            if not same:
                flips.append({'case': index, 'question': qid, 'reference_top': ref_top,
                              'candidate_top': actual_top, 'reference_confidence': confidence})
    confident = [f for f in flips if f['reference_confidence'] >= .7]
    same_pct, mean_kl = 100 * agree / decisions, sum(kls) / decisions
    return {'cases': len(reference), 'decisions': decisions, 'same_top': agree,
            'same_top_pct': same_pct, 'mean_abs_probability_drift': sum(drifts) / len(drifts),
            'max_abs_probability_drift': max(drifts), 'mean_kl': mean_kl,
            'flips': flips, 'confident_flips': confident, 'thresholds': GATE,
            'passes_gate': same_pct >= 99 and mean_kl <= .005 and not confident,
            'precision': 'Public probabilities rounded to four decimals, normalized before KL; choice labels retain tie decisions.'}


def collect(args):
    import importlib.metadata
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['USE_TF'] = '0'
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cases_raw = Path(args.cases).read_bytes()
    cases = json.loads(cases_raw)
    if args.backend == 'pytorch':
        from scripts.download_model import validate_model, WEIGHT_SHA256
        import laya
        if importlib.metadata.version('laya') != '0.3.4':
            raise ValueError('Reference requires laya==0.3.4 in a separate environment')
        validate_model(Path(args.model))
        agent = laya.load(args.model, device='cpu')
        backend = 'pytorch-fp32'
        checkpoint = WEIGHT_SHA256
    else:
        from scripts.onnx_runtime import load
        agent = load(args.model, args.onnx, args.threads)
        backend = 'onnx'
        checkpoint = hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest()
    answers = [agent.predict(case['state'], case['questions'])['answers'] for case in cases]
    value = {'backend': backend, 'laya': importlib.metadata.version('laya'),
             'checkpoint_sha256': checkpoint, 'suite_sha256': hashlib.sha256(cases_raw).hexdigest(),
             'answers': answers}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Collected {len(answers)} cases with {backend}.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest='action', required=True)
    cmp = subs.add_parser('compare', help='Compare any build’s saved answers with a recorded reference')
    cmp.add_argument('--reference', required=True)
    cmp.add_argument('--candidate', required=True)
    cmp.add_argument('--output')
    cp = subs.add_parser('collect', help='Run exactly one backend; use separate processes/environments')
    cp.add_argument('--backend', choices=['pytorch', 'onnx'], required=True)
    cp.add_argument('--cases', default='reports/parity-cases.json')
    cp.add_argument('--model', default='.cache/model/typed-decisions')
    cp.add_argument('--onnx', default='.cache/onnx/onnx/model.onnx')
    cp.add_argument('--threads', type=int, default=4)
    cp.add_argument('--output', required=True)
    args = ap.parse_args()
    try:
        if args.action == 'collect':
            if args.threads < 1: raise ValueError('Threads must be positive')
            return collect(args)
        reference = json.loads(Path(args.reference).read_text(encoding='utf-8'))
        candidate = json.loads(Path(args.candidate).read_text(encoding='utf-8'))
        if reference.get('backend') != 'pytorch-fp32' or reference.get('laya') != '0.3.4':
            raise ValueError('Reference must be the pinned PyTorch FP32 runtime')
        from scripts.download_model import WEIGHT_SHA256
        if reference.get('checkpoint_sha256') != WEIGHT_SHA256:
            raise ValueError('Reference checkpoint differs from the pinned checkpoint')
        if not reference.get('suite_sha256') or reference['suite_sha256'] != candidate.get('suite_sha256'):
            raise ValueError('Suite hashes differ or are missing')
        result = compare_answers(reference['answers'], candidate['answers'])
        if result['cases'] < 40 or result['decisions'] < 70:
            raise ValueError('Gate requires at least 40 cases and 70 decisions')
        result['suite_sha256'] = reference['suite_sha256']
        text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
        if args.output:
            dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding='utf-8')
        print(text, end='')
        return 0 if result['passes_gate'] else 1
    except (ValueError, KeyError, TypeError) as error:
        ap.error(str(error))


if __name__ == '__main__':
    # Support both `python scripts/parity.py` and module execution.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
