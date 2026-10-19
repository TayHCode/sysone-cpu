"""Exercise the real load() with deterministic scheduling and fake dependencies.

No ONNX, Laya, weights, downloads, or network calls. This is a model/code
conformance witness, not verification of upstream inference.
"""
import sys
import threading
import types
from unittest.mock import patch

from check_source import ROOT

sys.path.insert(0, str(ROOT))
from scripts import onnx_runtime as runtime


def replay(unlocked=False, fail_a=False, fail_b=False):
    entered = {name: threading.Event() for name in ("A", "B")}
    proceed = {name: threading.Event() for name in ("A", "B")}
    attempted_b = threading.Event()
    results, failures = {}, []
    ort = types.ModuleType("onnxruntime")

    def base(path, sess_options=None, providers=None, **kwargs):
        # Zero is a sentinel for an unconfigured session, not an ORT default claim.
        return getattr(sess_options, "intra_op_num_threads", 0)

    ort.InferenceSession = base
    ort.SessionOptions = type("SessionOptions", (), {})
    ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)
    agent_module = types.ModuleType("laya.onnx_agent")

    def wait(event):
        if not event.wait(5):
            raise RuntimeError("Replay scheduling timeout")

    def agent(model, onnx_path):
        entered[model].set()
        wait(proceed[model])
        results[model] = ort.InferenceSession(onnx_path)
        if {"A": fail_a, "B": fail_b}[model]:
            raise ValueError("Synthetic constructor failure after session creation")
        return results[model]

    agent_module.ONNXAgent = agent

    class ObservedLock:
        """Preserve real mutual exclusion, exposing B's attempt for scheduling."""
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "B":
                attempted_b.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    def caller(name, threads):
        try:
            runtime.load(name, "synthetic-graph", threads)
        except ValueError as error:
            if str(error) != "Synthetic constructor failure after session creation":
                failures.append(type(error).__name__)
        except Exception as error:
            failures.append(type(error).__name__)

    modules = {"onnxruntime": ort, "laya": types.ModuleType("laya"),
               "laya.onnx_agent": agent_module}
    with patch.dict(sys.modules, modules), \
            patch.object(runtime.importlib.metadata, "version", return_value="0.4.2"), \
            patch.object(runtime, "_CONSTRUCTION_LOCK", ObservedLock()):
        a = threading.Thread(target=caller, name="A", args=("A", 2), daemon=True)
        b = threading.Thread(target=caller, name="B", args=("B", 4), daemon=True)
        try:
            a.start()
            wait(entered["A"])
            b.start()
            wait(entered["B"] if unlocked else attempted_b)
            proceed["A"].set()
            a.join(5)
            wait(entered["B"])
            proceed["B"].set()
            b.join(5)
            if a.is_alive() or b.is_alive() or failures:
                raise RuntimeError("Replay thread failed or did not finish")
            restored = ort.InferenceSession is base
        finally:
            proceed["A"].set()
            proceed["B"].set()
            if a.ident is not None:
                a.join(5)
            if b.ident is not None:
                b.join(5)
    print(f"Actual load replay: A={results.get('A')} (requested 2), "
          f"B={results.get('B')} (requested 4), original_restored={restored}; "
          f"raises=({fail_a},{fail_b})")
    return results == {"A": 2, "B": 4} and restored


def run(unlocked=False):
    if unlocked:
        return replay(unlocked=True)
    return all(replay(fail_a=a, fail_b=b) for a in (False, True) for b in (False, True))
