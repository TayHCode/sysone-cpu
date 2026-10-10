"""Measure the pinned native Laya ONNX runtime on CPU; retain public answers."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from benchmarks.bench_laya_cpu import STATE, question_set, timed
from scripts.onnx_runtime import load
import psutil
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='.cache/model/typed-decisions')
    ap.add_argument('--onnx', default='.cache/onnx/onnx/model.onnx')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--counts', default='1,3,16')
    ap.add_argument('--output', default='.cache/onnx-results.json')
    args = ap.parse_args()
    counts = [int(n) for n in args.counts.split(',')]
    if args.threads < 1 or args.warmup < 0 or args.repeats < 1 or not counts or min(counts) < 1:
        ap.error('Use positive threads, counts and repeats; warmup may be zero')
    if args.onnx == '.cache/onnx/onnx/model.onnx':
        from scripts.download_onnx import validate_artifacts
        validate_artifacts(Path('.cache/onnx'))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(.1): peak[0] = max(peak[0], process.memory_info().rss)

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    start = time.perf_counter()
    agent = load(args.model, args.onnx, args.threads)
    cold = time.perf_counter() - start
    start = time.perf_counter()
    first = agent.predict(STATE, question_set(1))
    first_s = time.perf_counter() - start
    value = {'meta': {'device': 'cpu', 'providers': agent.session.get_providers(),
                     'python': platform.python_version(), 'platform': platform.platform(),
                     'versions': {name: importlib.metadata.version(name) for name in
                                  ['laya', 'torch', 'onnxruntime', 'onnx', 'onnxscript', 'numpy', 'transformers']},
                     'threads': args.threads, 'inter_op_threads': 1,
                     'cold_load_ms': round(cold * 1000, 3), 'first_answer_ms': round(first_s * 1000, 3),
                     'load_semantics': 'Fresh process/session; filesystem cache not flushed'},
             'first_answer': first['answers'], 'workloads': {}}
    try:
        for n in counts:
            metrics, _ = timed(agent, STATE, question_set(n), args.warmup, args.repeats, process)
            value['workloads'][str(n)] = metrics
            print(f'{n} questions: p50={metrics["p50_ms"]} ms p95={metrics["p95_ms"]} ms, {metrics["questions_per_second_p50"]} q/s', flush=True)
        questions = question_set(3)
        batched = agent.predict(STATE, questions)['answers']
        singles = {qid: agent.predict(STATE, {qid: q})['answers'][qid] for qid, q in questions.items()}
        value['batch_equivalence'] = {'same_public_answers': batched == singles}
        value['fixture'] = {'state': STATE, 'questions': questions, 'answers': batched}
    finally:
        stop.set(); observer.join()
    value['rss_peak_sampled_mib'] = round(max(peak[0], process.memory_info().rss) / 2**20, 2)
    value['rss_sampling_interval_s'] = .1
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {output}; provider={value["meta"]["providers"]}, batch equivalence={batched == singles}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
