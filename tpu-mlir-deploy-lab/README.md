# TPU-MLIR Deploy Lab

本项目用于学习和实践 TPU-MLIR 模型转换、量化和部署流程。

该项目不负责模型训练，而是接收 `tinylm-infra` 导出的 ONNX 模型，并完成从 ONNX 到 MLIR，再到 bmodel 的部署链路。

## 项目目标

本项目计划实现以下内容：

- 将 ONNX 模型转换为 MLIR。
- 分析生成的 MLIR 中间表示。
- 生成 FP32 bmodel。
- 准备 calibration 数据。
- 完成 INT8 量化。
- 生成 INT8 bmodel。
- 对比 FP32 和 INT8 的输出误差。
- 记录 unsupported op、shape 不匹配、量化误差等常见问题。

## 输入来源

模型输入来自：

```text
tinylm-infra/exports/
├── tinylm.onnx
├── sample_inputs.npz
└── tokenizer.json
```
复制到本项目：
tpu-mlir-deploy-lab/
├── models/
└── sample_inputs/
目录结构
models/：ONNX 模型文件。
sample_inputs/：模型输入样例。
calibration_data/：INT8 量化校准数据。
scripts/：TPU-MLIR 转换、部署和量化脚本。
mlir_outputs/：生成的 MLIR 文件。
bmodels/：生成的 bmodel 文件。
reports/：实验报告和误差分析。
docs/：TPU-MLIR 学习笔记和面试问题整理。
