# TinyLM-Infra

TinyLM-Infra is an end-to-end tiny language model infrastructure project.

## Goals

- Implement tokenizer, dataset, GPT-style model, optimizer, and training loop from scratch.
- Implement inference features such as sampling, KV cache, and batching.
- Implement CUDA / Triton kernels for softmax, layernorm, and matmul.
- Integrate custom CUDA kernels with PyTorch extensions.
- Export model to ONNX and benchmark ONNX Runtime inference.

## Structure

- `tiny_lm/`: core model package
- `training/`: training scripts
- `inference/`: inference and benchmark scripts
- `kernels/`: CUDA and Triton kernels
- `torch_extension/`: PyTorch C++/CUDA extensions
- `tests/`: unit tests
- `docs/`: project notes and interview preparation
