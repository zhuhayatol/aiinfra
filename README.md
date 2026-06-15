# AI Infra 面试准备仓库

本仓库用于记录和实现 AI Infra 相关面试准备内容，包括算法学习、语言模型训练、CUDA/Triton 底层算子、推理框架、TPU-MLIR 编译部署以及面试笔记。

## 仓库结构

- `tinylm-infra/`：主项目，从零实现 TinyLM 的训练、推理与底层算子优化。
- `tpu-mlir-deploy-lab/`：TPU-MLIR 部署项目，用于模型转换、量化与 bmodel 生成。
- `notes/`：学习笔记，包括 Hot100、Karpathy、CS336、CUDA、推理系统、TPU-MLIR 等。
- `datasets/`：数据集目录，不同步到 GitHub。
- `model_artifacts/`：模型权重、ONNX、bmodel、日志和 benchmark 结果，不同步到 GitHub。

## 学习主线

1. Karpathy Zero to Hero：理解神经网络、语言模型、Tokenizer 和 GPT 基础。
2. CS336：系统学习 LLM 训练、数据处理、优化器、profiling 和 systems。
3. CUDA / Triton：实现 softmax、layernorm、matmul 等核心算子。
4. PyTorch / ONNX Runtime：完成训练、推理和性能 benchmark。
5. TPU-MLIR：完成 ONNX 到 MLIR，再到 bmodel 的部署链路。

## 项目目标

最终形成两个可展示项目：

1. TinyLM-Infra：从零实现语言模型训练、推理与底层算子优化。
2. TPU-MLIR Deploy Lab：基于 TPU-MLIR 的模型转换、INT8 量化与部署实验。
