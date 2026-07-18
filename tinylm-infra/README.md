# TinyLM-Infra

TinyLM-Infra 是一个面向 AI Infra 面试准备的端到端小型语言模型基础设施项目。

本项目目标不是训练大模型，而是从零实现一个小型语言模型的关键链路，并围绕训练、推理、底层算子和性能优化进行系统化学习。

## 项目目标

本项目计划实现以下内容：

- 从零实现 Tokenizer、Dataset、GPT-style Transformer、AdamW 和训练循环。
- 实现 autoregressive generation、temperature sampling、top-k sampling、top-p sampling 和 KV Cache。
- 实现 CUDA / Triton 版本的 softmax、layernorm、matmul 等核心算子。
- 使用 PyTorch C++/CUDA Extension 将自定义算子接入 PyTorch。
- 导出 ONNX，并使用 ONNX Runtime 进行推理 benchmark。
- 分析训练和推理阶段的 latency、throughput、显存占用和性能瓶颈。

## 目录结构

- `tiny_lm/`：核心模型代码，包括 tokenizer、dataset、model、optimizer 和 generation。

- `tests/`：自动化测试。


