# Existing CPU runtime

This recipe documents an existing local deployment, rather than a new inference
engine. The original measurement machine runs Windows 11 on a Ryzen 9 7950X
(16 cores / 32 threads) with 64 GiB RAM. Inference explicitly selects the CPU.

The runtime is upstream [Laya v0.3.4](https://github.com/NandhaKishorM/laya/tree/42626c348753fbb17572a813127df2278a1ec527),
source revision `42626c348753fbb17572a813127df2278a1ec527`.
The model is the `typed-decisions` folder of
[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982/typed-decisions),
Hub revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.

Required files are `model.safetensors`, `rl_agent_config.json`,
`encoder/config.json`, `tokenizer/tokenizer.json` and
`tokenizer/tokenizer_config.json`. The weight file is 842,609,220 bytes;
SHA-256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.
Historical ONNX graph/data hashes are recorded in `reports/onnx-manifest.json`;
the optional community graph uses `reports/native-onnx-manifest.json`.

The original setup used Python 3.12.10, torch 2.14.0+cpu,
Transformers 5.17.0, safetensors 0.8.0, huggingface_hub 1.32.0,
NumPy 2.5.3, psutil 7.2.2, ONNX Runtime 1.24.3, onnx 1.20.0,
onnxscript 0.6.2. Laya loads with `laya.load(model_directory, device="cpu")`;
on CPU it uses float32 arithmetic even though the source configuration says bf16.

The original benchmarks use `benchmarks/bench_laya_cpu.py --model` with a
local checkpoint directory, 3 warm-ups and 10 repeats at either 16 or 4 threads.

The historical secondary path used an already-created FP32 ONNX export from
[receptron/laya 0.1.2](https://github.com/receptron/laya/tree/6478649e723122ca24bbf5fb69ed1010023c9750),
run with `CPUExecutionProvider` and four threads. No conversion is required by
the optional ONNX recipe, which downloads a different pinned community export
and uses the published native ONNX runtime. The default uses the original
PyTorch checkpoint. No runtime is rebuilt here. See the model manifests in
`reports/` for its hashes. The published base-English ONNX bundle is a different
checkpoint and must not be substituted for typed-decisions.
