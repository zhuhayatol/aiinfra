# tinylm-infra
用于学习大模型训练与推理基础设施的轻量项目

# 如何使用

以下命令均在项目根目录执行。

## 安装

- 安装项目运行依赖

```bash
pip install -e .
```

- 安装项目运行依赖和测试依赖

```bash
pip install -e ".[dev]"
```

## 准备数据

### TinyStories

下载并切分 TinyStories 数据集：

```bash
bash scripts/prepare_tinystories.sh
```

处理后的数据默认保存在：

```text
data/tinystories/
```

### HellaSwag

需要进行 HellaSwag 评测时执行：

```bash
bash scripts/prepare_hellaswag.sh
```

如果不需要进行 HellaSwag 评测，可以在配置文件中设置：

```yaml
evaluation:
  run_hellaswag: false
```

## 配置文件

项目提供两份训练配置：

```text
configs/train_gpt2.yaml    正式训练配置
configs/train_debug.yaml   调试和断点恢复测试配置
```

配置文件主要包括：

```text
experiment    实验名称、随机种子和输出目录
model         模型结构
data          数据目录、batch size和序列长度
training      学习率、训练步数、精度和梯度裁剪
evaluation    验证间隔和HellaSwag设置
checkpoint    保存间隔和恢复路径
```

命令行中的 `--resume` 优先于配置文件中的 `checkpoint.resume_from`。

## 测试

### 全部单元测试

```bash
pytest -s tests/unit
```

### 配置和Checkpoint测试

```bash
pytest -s tests/unit/config/test_config.py
pytest -s tests/unit/test_checkpoint.py
```

### GPT模型测试

```bash
pytest -s tests/unit/test_gpt2_model.py
```

### Tokenizer测试

```bash
pytest -s tests/test_tokenizer
```

### test_pretrained.py

- 加载本地Hugging Face权重

```bash
GPT2_LOCAL_PATH=$(pwd)/tiny_lm/model/gpt2_huggingface \
pytest -s tests/integration/test_pretrained.py
```

- 也可以先设置环境变量

```bash
export GPT2_LOCAL_PATH=$(pwd)/tiny_lm/model/gpt2_huggingface
pytest -s tests/integration/test_pretrained.py
```

- 不指定本地路径时，使用Hugging Face缓存或在线下载

```bash
unset GPT2_LOCAL_PATH
pytest -s tests/integration/test_pretrained.py
```

**注意**：显式设置 `GPT2_LOCAL_PATH` 后，只会读取本地权重，不会自动访问网络。

### 全部测试

```bash
pytest -s tests
```

## 训练

### 单进程训练

使用默认正式配置：

```bash
python -m tiny_lm.train.train_gpt2
```

指定配置文件：

```bash
python -m tiny_lm.train.train_gpt2 \
  --config configs/train_debug.yaml
```

也可以使用脚本：

```bash
bash scripts/train_gpt.sh \
  --config configs/train_gpt2.yaml
```

### DDP训练

两张GPU：

```bash
NUM_GPUS=2 bash scripts/train_gpt.sh \
  --config configs/train_gpt2.yaml
```

四张GPU：

```bash
NUM_GPUS=4 bash scripts/train_gpt.sh \
  --config configs/train_gpt2.yaml
```

DDP参数由 `torchrun` 设置的 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE` 环境变量决定。

当前DDP断点恢复会恢复模型、优化器和训练步数，不恢复每个rank独立的DataLoader和随机数状态。

## Checkpoint

Checkpoint保存在配置文件指定的输出目录中：

```text
outputs/<experiment.name>/checkpoints/
```

例如：

```text
outputs/gpt2_tinystories/checkpoints/model_00049.pt
```

Checkpoint中包含：

```text
模型权重
模型结构配置
完整训练配置
优化器状态
训练step
验证损失
单卡DataLoader状态
单卡CPU和CUDA随机数状态
```

### 从Checkpoint恢复训练

```bash
python -m tiny_lm.train.train_gpt2 \
  --config configs/train_gpt2.yaml \
  --resume outputs/gpt2_tinystories/checkpoints/model_00049.pt
```

也可以使用脚本：

```bash
bash scripts/train_gpt.sh \
  --config configs/train_gpt2.yaml \
  --resume outputs/gpt2_tinystories/checkpoints/model_00049.pt
```

恢复后会从Checkpoint中已经完成的step的下一步继续训练。

## 输出目录

一次训练的输出目录如下：

```text
outputs/gpt2_tinystories/
├── config.yaml
├── gpt2_tinystories.log
└── checkpoints/
    └── model_00049.pt
```

其中：

```text
config.yaml                 本次训练实际使用的配置副本
gpt2_tinystories.log        训练、验证和HellaSwag日志
checkpoints/                模型Checkpoint
```

## 文本生成

默认从以下目录寻找最新Checkpoint：

```text
outputs/gpt2_tinystories/checkpoints/
```

运行：

```bash
bash scripts/generate.sh
```

输入prompt后按回车生成文本，输入以下任意内容退出：

```text
q
quit
exit
```

## 绘制训练曲线

```bash
bash scripts/plot.sh \
  --log outputs/gpt2_tinystories/gpt2_tinystories.log \
  --save outputs/gpt2_tinystories/training_curves.png
```
