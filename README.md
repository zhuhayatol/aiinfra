# AI Infra 面试准备仓库

本仓库用于记录和实现 AI Infra 相关面试准备内容，包括算法学习、语言模型训练、CUDA/Triton 底层算子、推理框架、TPU-MLIR 编译部署以及面试笔记。

## 仓库结构

- `tinylm-infra/`：主项目，从零实现 TinyLM 的训练、推理与底层算子优化。
- `tpu-mlir-deploy-lab/`：TPU-MLIR 部署项目，用于模型转换、量化与 bmodel 生成。
- `notes/`：学习笔记，包括 Hot100、Karpathy、CS336、CUDA、推理系统、TPU-MLIR 等。


## 项目目标

最终形成两个可展示项目：

1. TinyLM-Infra：从零实现语言模型训练、推理与底层算子优化。
2. TPU-MLIR Deploy Lab：基于 TPU-MLIR 的模型转换、INT8 量化与部署实验。
