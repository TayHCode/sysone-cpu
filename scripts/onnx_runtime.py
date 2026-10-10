"""Configure the published Laya ONNXAgent for this single-process CPU recipe."""
from __future__ import annotations

import importlib.metadata
import threading

_CONSTRUCTION_LOCK = threading.Lock()


def load(model: str, onnx: str, threads: int = 4):
    if threads < 1:
        raise ValueError('Use a positive thread count')
    if importlib.metadata.version('laya') != '0.4.2':
        raise ValueError('This measured native ONNX recipe requires laya==0.4.2')
    import onnxruntime as ort
    from laya.onnx_agent import ONNXAgent

    # 0.4.2 constructs its session internally and exposes no thread/provider
    # options. Scope this override to construction, restore even on failure,
    # and serialize this helper's constructors. Prediction stays upstream code.
    with _CONSTRUCTION_LOCK:
        original = ort.InferenceSession

        def cpu_session(path, sess_options=None, providers=None, **kwargs):
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return original(path, sess_options=options, providers=['CPUExecutionProvider'], **kwargs)

        ort.InferenceSession = cpu_session
        try:
            return ONNXAgent(model, onnx_path=onnx)
        finally:
            ort.InferenceSession = original
