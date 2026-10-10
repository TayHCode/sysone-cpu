"""Measure the upstream Laya PyTorch runtime on a CPU.

This is intentionally a small black-box harness around the published runtime:
it does not change model code, use CUDA, or claim accuracy from synthetic
inputs. It measures the deployment properties needed for an always-on local
classifier and checks repeatability/batch equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import psutil
import torch


STATE = {
    "ticket": {
        "subject": "Duplicate charge",
        "body": (
            "The customer was billed twice for invoice 4411 and asks for a refund "
            "today. They say the duplicate charge is blocking month-end close."
        ),
    }
}

BASE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "sales": "pricing, demos, new purchases",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["not urgent", "soon", "critical or blocking"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
    },
}


def question_set(n: int) -> dict:
    """Create n independent questions while keeping one mixed 3-question set."""
    if n == 1:
        return {"refund": BASE_QUESTIONS["refund"]}
    if n == 3:
        return dict(BASE_QUESTIONS)
    out = {}
    for i in range(n):
        q = BASE_QUESTIONS["department"] if i % 3 == 0 else BASE_QUESTIONS["refund"]
        out[f"q{i}"] = q
    return out


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def quantiles_ms(samples: list[float]) -> dict:
    return {
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
        "min_ms": round(float(min(samples)), 3),
        "max_ms": round(float(max(samples)), 3),
        "mean_ms": round(float(statistics.mean(samples)), 3),
    }


def timed(agent, state, questions, warmup: int, repeats: int, process: psutil.Process):
    for _ in range(warmup):
        agent.predict(state, questions)
    samples = []
    rss_peak = process.memory_info().rss
    cpu_samples = []
    outputs = []
    for _ in range(repeats):
        process.cpu_percent(None)
        t0 = time.perf_counter()
        result = agent.predict(state, questions)
        elapsed = (time.perf_counter() - t0) * 1000.0
        cpu = process.cpu_percent(None)
        samples.append(elapsed)
        cpu_samples.append(cpu)
        rss_peak = max(rss_peak, process.memory_info().rss)
        outputs.append(result)
    return {
        **quantiles_ms(samples),
        "questions": len(questions),
        "questions_per_second_p50": round(len(questions) / (np.percentile(samples, 50) / 1000.0), 3),
        "rss_peak_mib": round(rss_peak / 2**20, 2),
        "cpu_percent_mean_sampled": round(float(statistics.mean(cpu_samples)), 2),
        "output_sha256_first": digest(outputs[0]),
        "output_sha256_last": digest(outputs[-1]),
        "deterministic_repeated_output": len({digest(x) for x in outputs}) == 1,
        "sample_ms": [round(x, 3) for x in samples],
    }, outputs[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Laya checkpoint directory")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--counts", default="1,3,4,8,16", help="comma-separated question counts")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads; 0 keeps default")
    ap.add_argument("--output", default="reports/results/laya-cpu.json")
    args = ap.parse_args()

    counts = [int(value) for value in args.counts.split(",")]
    if args.repeats < 1 or args.warmup < 0 or args.threads < 0 or not counts or min(counts) < 1:
        ap.error("use positive counts/repeats and nonnegative warmup/threads")

    if args.threads:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    # Import the installed upstream wheel; no ignored source checkout is needed.
    import laya  # noqa: E402

    model_path = str(Path(args.model).expanduser().resolve())
    process = psutil.Process()
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    agent = laya.load(model_path, device="cpu")
    cold_load_ms = (time.perf_counter() - t0) * 1000.0
    rss_after_load = process.memory_info().rss

    results = {
        "meta": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "laya_version": getattr(laya, "__version__", "unknown"),
            "torch_threads": torch.get_num_threads(),
            "device": str(agent.device),
            "model_path": "<local typed-decisions checkpoint>",
            "model_config": {name: value for name, value in agent.cfg.items()
                             if name not in {"temperature", "temperature_by_options"}},
            "calibration_source": "pinned upstream rl_agent_config.json",
            "cold_load_ms": round(cold_load_ms, 3),
            "rss_before_mib": round(rss_before / 2**20, 2),
            "rss_after_load_mib": round(rss_after_load / 2**20, 2),
        },
        "workloads": {},
    }

    for n in counts:
        print(f"timing {n} questions ...", flush=True)
        metrics, result = timed(agent, STATE, question_set(n), args.warmup, args.repeats, process)
        results["workloads"][str(n)] = metrics
        print(
            f"  p50={metrics['p50_ms']:.1f} ms p95={metrics['p95_ms']:.1f} ms "
            f"{metrics['questions_per_second_p50']:.2f} q/s "
            f"RSS={metrics['rss_peak_mib']:.1f} MiB CPU={metrics['cpu_percent_mean_sampled']:.1f}%",
            flush=True,
        )

    # Compare one three-question request to three separate calls. This checks
    # batching semantics at the public answer level, not only timing.
    mixed = question_set(3)
    batched = agent.predict(STATE, mixed)["answers"]
    singles = {
        qid: agent.predict(STATE, {qid: q})["answers"][qid]
        for qid, q in mixed.items()
    }
    results["batch_equivalence"] = {
        "same_public_answers": batched == singles,
        "batched_digest": digest(batched),
        "singles_digest": digest(singles),
        "batched_answers": batched,
        "single_answers": singles,
    }

    # Capture a small, human-readable deterministic fixture for later port
    # comparisons. It contains only synthetic data.
    results["fixture"] = {"state": STATE, "questions": mixed, "answers": batched}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

