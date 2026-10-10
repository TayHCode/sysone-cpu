# How it works

System One here means a model that answers typed questions about an input in
one forward pass. You provide a state (for example, a ticket) and questions.
Laya tokenizes each question and state together, reads them with a bidirectional
ModernBERT-large encoder and uses its trained decision head to score the answer
markers. Temperature scaling turns these scores into a distribution.

`choice` selects a named option, `score` returns an expected ordinal value and
`noul` returns a probability for true. It generates no prose. Questions can be
batched, but the encoder does not reuse one state representation across them.
The English typed-decisions model has about 421 million parameters. CPU loading
uses float32 arithmetic even though the checkpoint configuration says bf16.

The implementation belongs to [Convai Innovations / Laya](https://github.com/NandhaKishorM/laya/tree/42626c348753fbb17572a813127df2278a1ec527).
This repository adds setup instructions, a pinned download and measurement
harnesses around that runtime. Calling `laya.load(model_directory, device="cpu")`
already works upstream; nothing has been ported or rebuilt here.

## Other existing paths

[llama.cpp's `/v1/systemone`](https://github.com/ggml-org/llama.cpp/pull/29818)
is an existing server interface for supported decision-model GGUFs. It is an
alternative runtime, not this project's implementation. A base-English GGUF
is a different checkpoint from typed-decisions; a converted encoder without
its head is also insufficient. No GGUF parity claim is made here.

[ollaya](https://github.com/ollaya-dev/ollaya) already provides pull/run/serve
commands and a compatible decision API using existing model artifacts.
[EdgeJev](https://github.com/yzfly/edgejev) provides ONNX build, quantization and
serving tools. [ONNX community artifacts](https://huggingface.co/onnx-community/laya-typed-decisions-ONNX)
and [receptron's exporter](https://github.com/receptron/laya/tree/6478649e723122ca24bbf5fb69ed1010023c9750)
provide other routes. None is required for the direct PyTorch quickstart.
The historical ONNX comparison measured the same typed-decisions weights,
whereas its base-English comparison measured a different checkpoint.

Laya is a distinct open model. Similar typed APIs do not imply numerical
agreement with other typed-decision models. Test predictions and calibration
on labelled examples from your own application before relying on the output.
