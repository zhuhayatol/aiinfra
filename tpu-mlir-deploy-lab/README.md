# TPU-MLIR Deploy Lab

This project focuses on compiling and deploying models with TPU-MLIR.

## Goals

- Convert ONNX models to MLIR.
- Generate FP32 bmodel.
- Run INT8 calibration.
- Generate INT8 bmodel.
- Compare FP32 and INT8 outputs.
- Analyze MLIR IR and unsupported operators.

## Input

The ONNX model and sample inputs are exported from `tinylm-infra`.

## Structure

- `models/`: ONNX models
- `sample_inputs/`: sample input tensors
- `calibration_data/`: calibration data for INT8 quantization
- `scripts/`: TPU-MLIR conversion scripts
- `mlir_outputs/`: generated MLIR files
- `bmodels/`: generated bmodel files
- `reports/`: experiment reports
- `docs/`: TPU-MLIR notes
