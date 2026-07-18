# TinyLM-Infra：代码功能、训练参数与完整运行流程

> 本文面向当前 `tinylm-infra` 仓库快照，按实际目录和代码说明各文件职责、核心函数、训练参数、运行顺序和注意事项。  
> 它既可以作为项目说明，也可以作为后续重构、排错和面试复习时的工程手册。

---

# 1. 项目全局视图

## 1.1 当前项目链路

```text
Tokenizer 实现与测试
    ↓
TinyStories 下载与 tokenization
    ↓
保存 train/val NumPy shards
    ↓
DataLoaderLite 按 rank 读取 token
    ↓
GPT-2 模型 forward/loss
    ↓
AdamW + LR schedule + gradient accumulation
    ↓
单卡或 DDP 训练
    ↓
validation / generation / HellaSwag
    ↓
log / plot / checkpoint
    ↓
交互式加载 checkpoint 生成文本
```

## 1.2 当前主要目录

```text
tinylm-infra/
├── tiny_lm/
│   ├── tokenizer/
│   ├── data/
│   ├── dataloader/
│   ├── model/
│   ├── train/
│   ├── eval/
│   ├── generation/
│   └── utils/
├── tests/
├── scripts/
├── data/             # 运行后生成，压缩包中已排除
├── log/
├── note/
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

# 2. 各代码文件的功能

# 2.1 `tiny_lm/tokenizer/base.py`

这是 tokenizer 的公共基础模块。

## `get_stats(text)`

输入 token id 列表，统计相邻 pair 的频次：

```text
[1, 2, 1, 2]
→ (1,2): 2
→ (2,1): 1
```

用途：BPE 训练时寻找最高频 pair；encode 时寻找当前可合并 pair。

## `merge(tokens, pair, count)`

从左到右扫描 token 列表，将指定 pair 替换为新的 token id。

关键点：

- 不重叠合并；
- 命中 pair 后索引前进 2；
- 未命中时前进 1。

## `replace_control_characters()` / `render_token()`

将换行、控制字符等转义，用于生成便于人阅读的词表文件，避免控制字符破坏终端或文本布局。

## `Tokenizer`

公共状态：

```text
merges
pattern
special_tokens
vocab
```

公共能力：

- `encode()`：基础 byte encode；
- `decode()`：根据 vocab 拼接 bytes，再 UTF-8 decode；
- `_build_vocab_()`：从基础 byte token、merge 和 special token 重建词表；
- `save()`：保存 `.model` 与 `.vocab`；
- `load()`：恢复 merge 规则和 special token；
- `save_vocab()`：生成人类可读词表。

注意：

- `train()` 是抽象接口；
- 当前 `decode()` 默认严格 UTF-8 解码，非法 token 组合可能报错；
- `.model` 只保存 pair，merge id 按 256 起的顺序重建。

---

# 2.2 `tiny_lm/tokenizer/basic.py`

## `BPETokenizer.train()`

流程：

```text
文本 → UTF-8 bytes
→ 统计相邻 pair
→ 合并最高频 pair
→ 分配新 id（从 256 开始）
→ 重复直到达到 vocab_size
```

`vocab_size - 256` 是最大 merge 次数。

## `BPETokenizer.encode()`

不是简单按照词表做最长匹配，而是严格按训练阶段的 merge 优先级执行：

1. 统计当前相邻 pair；
2. 找到在 `merges` 中 id 最小的 pair；
3. 执行合并；
4. 直到没有可合并 pair。

merge id 越小，表示越早学习到，优先级越高。

## `decode()`

复用基类，根据每个 id 对应的 bytes 拼回原文本。

---

# 2.3 `tiny_lm/tokenizer/regex.py`

在 byte-level BPE 前先使用 GPT 风格正则表达式切分文本。

## `RegexTokenizer.__init__()`

- 保存 regex pattern；
- 编译 pattern；
- 初始化 special token 正反向映射。

## `register_special_tokens()`

注册人工定义 token，例如：

```text
<|endoftext|>
<|pad|>
```

special token 不由 BPE merge 学习得到。

## `train()`

流程：

```text
原始文本
→ regex 分块
→ 每块转 bytes
→ 汇总所有块内部 pair 频次
→ 选择全局最高频 pair
→ 在所有块中执行 merge
```

重要约束：BPE 不跨 regex chunk 合并。

## `encode_chunk()`

对单个 byte chunk 按 merge 优先级反复合并。

## `encode_ordinary()`

regex 分块后逐块执行 `encode_chunk()`，不处理 special token。

## `encode()`

根据 `allowed_special` 决定：

- special token 是否允许；
- 是否把 special token 当普通文本；
- 是否在遇到禁止 special token 时抛错。

实现时需要优先匹配更长 special token，避免前缀重叠导致错误切分。

## `decode()`

- 普通 token 从 vocab 取 bytes；
- special id 从 inverse map 恢复字符串；
- 最终拼接。

---

# 2.4 `tiny_lm/tokenizer/gpt4.py`

目标是复现 `cl100k_base` 的 tokenization 行为。

主要职责：

- 使用 GPT-4 的 regex pattern；
- 加载/构造与 tiktoken 对齐的 mergeable ranks；
- 注册 GPT-4 special tokens；
- 保持 encode 结果与官方 `tiktoken` 一致。

它更偏向“兼容实现与验证”，而 Basic/Regex tokenizer 更偏向理解和训练 BPE。

---

# 2.5 `tiny_lm/data/tinystories.py`

负责下载 TinyStories，并转换为训练可直接读取的 token shards。

## 全局配置

```text
dataset_path = roneneldan/TinyStories
local_dir = tinystories
shard_size = 1,000,000 tokens
```

## `tokenize(doc)`

1. 在每篇文档前添加 `<|endoftext|>`；
2. 用 GPT-2 tokenizer 的 `encode_ordinary()` 编码正文；
3. 检查 token id 小于 `2^16`；
4. 保存为 `np.uint16`。

## `write_datafile(filename, tokens_np)`

通过 `np.save()` 写 shard。即使文件名不手动带 `.npy`，NumPy 也会追加扩展名。

## `process_spilt(dataset, split_name)`

> 代码函数名当前写作 `process_spilt`，语义上应为 `process_split`，后续可重命名。

功能：

- 创建 CPU multiprocessing pool；
- 并行 tokenize 文档；
- 将 token 填入固定大小 buffer；
- 满 1M token 后写 shard；
- 文档跨 shard 时切分 remainder；
- 最后写不足 1M token 的尾 shard。

## `main()`

分别下载：

```text
train
validation
```

并保存为：

```text
tinystories_train_*.npy
tinystories_val_*.npy
```

---

# 2.6 `tiny_lm/dataloader/dataloader.py`

## `load_tokens(filename)`

```text
.npy
→ NumPy array
→ int32
→ torch.long
```

说明：

- 磁盘保存为 uint16；
- 加载时转 long 以作为 Embedding index；
- 中间 int32 转换不是绝对必要，但不会改变 id。

## `DataLoaderLite.__init__()`

输入：

| 参数 | 含义 |
|---|---|
| `B` | 每进程 micro batch size |
| `T` | sequence length |
| `process_rank` | 当前 DDP rank |
| `num_processes` | world size |
| `split` | `train` 或 `val` |
| `file_name` | 单文本文件模式 |
| `local_dir` | shard 数据集目录 |

两种模式：

### 文本文件模式

- 读取完整文本；
- GPT-2 tokenize；
- 全部 token 放入内存。

### Shard 模式

- 定位 `data/{local_dir}`；
- 按文件名筛选 train/val；
- 排序；
- 加载第一个 shard。

## `reset()`

```text
current_shard = 0
加载第一个 shard
current_position = B×T×rank
```

validation 每次评估前调用 reset，使评估数据保持一致。

## `next_batch()`

读取：

```text
B×T+1 tokens
```

构造：

```text
x: [B,T]
y: [B,T]
```

然后位置前进：

```text
B×T×world_size
```

若当前 shard 即将耗尽，切换下一 shard。

## `num_tokens()`

shard 模式逐个加载并统计 token 数。

当前问题：

- 文本文件分支是 `pass`；
- 逐个完整加载 shard 统计较慢；
- 未返回值，只打印；
- 后续可通过文件 metadata 或保存 manifest 优化。

---

# 2.7 `tiny_lm/model/gpt2.py`

## `GPTConfig`

保存模型超参数。

## `CausalSelfAttention`

功能：

```text
[B,T,C]
→ QKV projection
→ 拆多头
→ causal SDPA
→ 合并多头
→ output projection
```

`c_proj.NANOGPT_SCALE_INIT` 标记该层执行残差缩放初始化。

应补充：

```python
assert config.n_embd % config.n_head == 0
```

当前 Linear 没有传入 `bias=config.bias`，因此配置中的 `bias` 尚未真正生效。

## `MLP`

```text
C → 4C → GELU → C
```

输出 `c_proj` 同样进入残差流，需要缩放初始化。

## `Block`

Pre-LN：

```python
x = x + attn(ln_1(x))
x = x + mlp(ln_2(x))
```

## `GPT.__init__()`

创建：

- token embedding；
- position embedding；
- n_layer 个 Block；
- final LayerNorm；
- LM Head；
- weight tying；
- 自定义初始化。

## `_init_weight()`

- Linear weight：normal；
- Linear bias：zero；
- Embedding：normal；
- residual output projection：`std / sqrt(2L)`。

## `forward(idx, target=None)`

1. 检查 `T <= block_size`；
2. token + position embedding；
3. 经过 Blocks；
4. final LayerNorm；
5. LM Head；
6. 可选 cross entropy。

输入输出：

```text
idx:     [B,T]
target:  [B,T]
logits:  [B,T,V]
loss:    scalar or None
```

## `generate()`

自回归循环，支持：

- context crop；
- temperature；
- top-k；
- multinomial；
- 外部 generator。

当前没有：

- top-p；
- repetition penalty；
- EOS early stop；
- KV Cache；
- batch 内不同完成长度。

## `from_pretrained()`

支持：

```text
gpt2
gpt2-medium
gpt2-large
gpt2-xl
```

关键：

- 过滤 attention mask buffer；
- 对 Conv1D 风格权重转置；
- shape assert；
- `copy_()` 到本地参数。

## `configure_optimizers()`

1. 收集可训练参数；
2. `dim>=2` 参数执行 decay；
3. 一维参数不 decay；
4. 检查 fused AdamW；
5. 创建 AdamW。

---

# 2.8 `tiny_lm/eval/prepare_hellaswag.py`

负责联网下载一次 HellaSwag validation，并显式保存为可离线加载的 Dataset 目录。

目录：

```text
data/hellaswag/
├── hf_cache/
└── validation/
```

关键区别：

```text
cache_dir
≠
save_to_disk
```

`load_from_disk()` 只能读取 `save_to_disk()` 生成的目录，不能直接读取 HuggingFace cache 根目录。

---

# 2.9 `tiny_lm/eval/hellaswag.py`

## `load_hellaswag()`

从：

```text
data/hellaswag/validation
```

使用 `load_from_disk()` 读取。

## `render_example(example, device)`

将一条样本转换为：

```text
tokens: [4,T]
mask:   [4,T]
label
```

- 四个 ending 分别与 context 拼接；
- ending 前加空格；
- padding 到相同长度；
- mask 仅标记 ending。

## `get_most_likely_row()`

1. forward 得到 `[4,T,V]`；
2. shift logits/tokens/mask；
3. 逐 token 计算 cross entropy；
4. mask context/padding；
5. 每个 ending 求平均 loss；
6. `argmin` 作为预测。

## `evaluate_hellaswag()`

- 加载数据；
- 遍历样本；
- 跳过超过 block_size 的样本；
- 支持 autocast；
- 支持 `max_examples`；
- 返回 acc、correct、total、skipped、预测列表。

工程注意：

- 训练时建议传 `raw_model`；
- 当前评估在 rank 0 独占执行，需要 DDP barrier；
- 完整 HellaSwag 应报告完整 validation 结果，而 200 条只用于训练监控。

---

# 2.10 `tiny_lm/train/train_gpt2.py`

这是训练主控制器。

## 主要阶段

```text
识别 DDP
→ 初始化设备和 rank
→ 计算 global batch / grad_accum
→ 创建 train/val loader
→ 创建模型
→ 可选 compile
→ 可选 DDP wrap
→ 创建 optimizer
→ 创建日志
→ step 循环
```

每个 step：

```text
validation（按间隔）
→ checkpoint（按间隔）
→ sample（按间隔）
→ HellaSwag（按间隔）
→ model.train()
→ zero_grad
→ grad accumulation
→ DDP loss reduce
→ gradient clip
→ 更新 LR
→ optimizer.step
→ 同步计时
→ 打印和记录 train loss
```

## 当前 DDP 识别

```python
ddp = int(os.environ.get("RANK", -1)) != -1
```

由 `torchrun` 设置环境变量。

## `device` 与 `device_type`

```text
device      = cuda:0 / cuda:1 / cpu
device_type = cuda / cpu
```

- `.to(device)` 使用具体设备；
- `autocast(device_type=...)` 使用设备类型。

## 模型引用

```python
model = DDP(model)
raw_model = model.module if ddp else model
```

- 训练 forward 用 `model`；
- checkpoint、generate、config、optimizer helper 用 `raw_model`。

## 日志与 checkpoint

只有 master process 写文件，避免多个进程并发覆盖。

当前启动时：

```python
open(log_file, "w")
```


会清空旧日志。正式实验应使用独立 run 目录或 append/resume 逻辑。


---

# 2.11 `tiny_lm/generation/generate.py`

## `find_latest_checkpoint()`

按文件名排序寻找最新：

```text
log/model_*.pt
```

依赖文件名中的 step 使用固定宽度，例如 `model_00500.pt`。

## `load_checkpoint_model()`

1. `torch.load()`；
2. config dict/object 兼容；
3. 创建 GPT；
4. 清理 `_orig_mod.` / `module.` 前缀；
5. load state dict；
6. 移动设备并 eval。

## `generate_once()`

- GPT-2 tokenizer encode；
- `[T] → [1,T]`；
- repeat 多条序列；
- 调用 model.generate；
- decode 并打印。

## `main()`

- 自动选择 CUDA/CPU；
- 模型只加载一次；
- 进入 prompt 输入循环；
- `q/quit/exit` 退出。

---

# 2.12 `tiny_lm/utils/plot.py`

## `plot_log()`

读取：

```text
step stream value
```

将每个 stream 整理为按 step 排序的数据。

绘制两个子图：

1. train/val loss；
2. HellaSwag accuracy。

输出：

```text
log/training_curves.png
```

当前注意点：

- 如果没有某类 stream，`legend()` 可能出现提示；
- `subplot` 适合当前简单图，后续可增加 lr、norm、tokens/sec；
- 日志中同一 stream 同 step 会被后写值覆盖，因为内部使用 dict。

---

# 2.13 `scripts/`

## `train_gpt.sh`

- 定位项目根目录；
- 输出 Python 和 GPU 信息；
- `NUM_GPUS=1` 时使用 Python module；
- `NUM_GPUS>1` 时使用 torchrun。

运行：

```bash
bash scripts/train_gpt.sh
NUM_GPUS=2 bash scripts/train_gpt.sh
```

## `prepare_tinystories.sh`

```bash
python -m tiny_lm.data.tinystories
```

## `prepare_hellaswag.sh`

```bash
python -m tiny_lm.eval.prepare_hellaswag
```

## `generate.sh`

```bash
python -m tiny_lm.generation.generate
```

## `plot.sh`

```bash
python -m tiny_lm.utils.plot
```

所有脚本先切换项目根目录，避免 package import 和相对路径错误。

---

# 2.14 `tests/`

## GPT 测试

已覆盖：

- forward shape；
- target loss 为 scalar；
- generate shape；
- 输入超过 block size 时 generate 可裁剪；
- HuggingFace 权重加载与生成。

需要改进：

- `from_pretrained` 测试硬编码 CUDA；
- 本地模型路径是相对路径；
- 应通过 `pytest.mark.integration` 和 `skipif` 隔离；
- 增加 block_size forward 报错测试；
- 增加 weight tying 和 optimizer group 测试。

## Tokenizer 测试

覆盖：

- encode/decode identity；
- Basic/Regex/GPT4；
- Wikipedia BPE 示例；
- special token；
- save/load；
- 与 tiktoken 对齐。

---

# 3. 训练参数总表

# 3.1 模型参数

| 参数 | 当前值 | 调大后的主要代价 |
|---|---:|---|
| `block_size` | 1024 | attention 计算/显存快速增加 |
| `vocab_size` | 50304（训练） | Embedding/LM Head 参数增加 |
| `n_layer` | 12 | 计算、显存、深度增加 |
| `n_head` | 12 | 需与 `n_embd` 整除 |
| `n_embd` | 768 | 参数量和 GEMM 规模显著增加 |

大致规律：

- Attention 主要随 `T²` 增长；
- MLP 主要随 `T×C²` 增长；
- 增大 `n_embd` 的代价通常非常高；
- `block_size` 是显存调节的重要旋钮。

---

# 3.2 Batch 与 token 参数

当前代码：

```text
total_batch_size = 16,384 tokens/update
B = 4 sequences/process/micro-step
T = 512 tokens/sequence
```

公式：

```text
micro_tokens_per_rank = B × T
global_tokens_per_micro_step = B × T × world_size
grad_accum_steps = total_batch_size / global_tokens_per_micro_step
```

当前单卡：

```text
micro_tokens = 4×512 = 2048
grad_accum_steps = 8
```

显存不足时优先调：

1. 降低 `B`；
2. 降低 `T`；
3. 增加 `grad_accum_steps` 以保持 total batch；
4. 开启 BF16；
5. 再考虑 gradient checkpointing（当前未实现）。

`T` 同时影响单序列上下文和 attention 开销，通常比调整 `B` 更敏感。

---

# 3.3 训练 token 预算

当前：

```text
max_steps = 51
total tokens = 835,584
```

约 4.74 亿 token 数据的一轮：

| `total_batch_size` | 约一轮 steps |
|---:|---:|
| 16,384 | 28,930 |
| 524,288 | 904 |

这些只是按 token 数计算，不考虑 shard 边界丢弃、跳转和重复。

---

# 3.4 学习率参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `max_lr` | `6e-4` | warmup 后峰值 |
| `min_lr` | `6e-5` | 峰值的 10% |
| `warmup_steps` | 10 | 短 smoke test |
| optimizer constructor LR | `3e-4` | 每 step 会被 scheduler 覆盖 |

重要细节：

`configure_optimizers(learning_rate=3e-4)` 中的 LR 只是创建时初始值。训练循环在第一次 `optimizer.step()` 前就把 param group 的 LR 改为 `get_lr(step)`。

因此当前实际 step 0 LR：

```text
6e-4 × 1/10 = 6e-5
```

---

# 3.5 AdamW 参数

```text
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
```

需要监控：

- loss 是否稳定；
- gradient norm；
- 更新早期是否震荡；
- validation 是否恶化。

不要孤立复制别人的学习率。学习率与：

```text
模型规模
batch token 数
数据分布
训练时长
初始化
优化器
```

都有耦合。

---

# 3.6 精度与性能参数

| 选项 | 当前状态 | 用途 |
|---|---|---|
| BF16 autocast | 开启 | 提升吞吐、降低激活显存 |
| TF32 | 注释关闭 | 加速 FP32 matmul |
| SDPA | 开启 | 统一 attention 高效后端 |
| torch.compile | `False` | 图编译与 fusion |
| fused AdamW | 自动检测 | 减少 optimizer kernel |
| gradient clipping | 1.0 | 限制异常梯度 |

---

# 3.7 评估参数

| 参数 | 当前值 |
|---|---:|
| validation interval | 50 steps |
| validation batches | 20 |
| HellaSwag interval | 50 steps |
| HellaSwag examples | 200 |
| generation interval | 50 steps |
| generation sequences | 4 |
| generation tokens | 32 |
| generation temperature | 1.0 |
| generation top-k | 50 |

正式训练时评估频率需要平衡：

```text
评估可信度
vs
训练吞吐损失
```

训练 step 很快而 HellaSwag 很慢时，50 step 一次可能占用大量时间。

---

# 3.8 Checkpoint 参数

当前：

```text
step > 0
并且 step % 500 == 0 或 last_step
```

但当前 `max_steps=51`，因此只会在最后一步保存一次。

checkpoint 当前用于 inference，不足以精确 resume。

---

# 4. 一次完整训练的运行流程

# 4.1 环境准备

当前 `requirements.txt` 为空，应至少补充：

```text
torch
numpy
tiktoken
regex
datasets
tqdm
matplotlib
transformers
pytest
```

建议固定兼容版本或至少记录：

```text
Python
PyTorch
CUDA
GPU
transformers
datasets
```

---

# 4.2 准备 TinyStories

```bash
bash scripts/prepare_tinystories.sh
```

预期：

```text
data/tinystories/
├── tinystories_train_000000.npy
├── ...
└── tinystories_val_000000.npy
```

检查：

```bash
find data/tinystories -type f | head
du -sh data/tinystories
```

---

# 4.3 准备 HellaSwag

```bash
bash scripts/prepare_hellaswag.sh
```

预期：

```text
data/hellaswag/validation/
├── data-*.arrow
├── dataset_info.json
└── state.json
```

---

# 4.4 运行测试

```bash
pytest -q
```

建议把需要 GPU/本地 HuggingFace 权重的测试隔离后再运行默认测试。

---

# 4.5 单卡训练

```bash
bash scripts/train_gpt.sh
```

内部执行：

```bash
python -m tiny_lm.train.train_gpt2
```

---

# 4.6 多卡训练

```bash
NUM_GPUS=2 bash scripts/train_gpt.sh
```

内部执行：

```bash
torchrun   --standalone   --nproc_per_node=2   -m tiny_lm.train.train_gpt2
```

运行前检查：

```bash
nvidia-smi
```

并保证：

```text
total_batch_size % (B*T*NUM_GPUS) == 0
```

---

# 4.7 训练循环的精确顺序

```text
step 开始
│
├─ 记录 t0
│
├─ 是否到评估点？
│   ├─ val_loader.reset()
│   ├─ validation forward
│   ├─ all_reduce val loss
│   ├─ rank 0 写 val log
│   └─ rank 0 条件保存 checkpoint
│
├─ 是否到采样点？
│   ├─ barrier
│   ├─ rank 0 generate
│   └─ barrier
│
├─ 是否到 HellaSwag 点？
│   ├─ barrier
│   ├─ rank 0 evaluate
│   ├─ rank 0 写 hella log
│   └─ barrier
│
├─ model.train()
├─ optimizer.zero_grad()
├─ loss_accum = 0
│
├─ micro_step 循环
│   ├─ next_batch
│   ├─ to(device)
│   ├─ BF16 forward
│   ├─ loss /= grad_accum_steps
│   ├─ loss_accum += detach(loss)
│   ├─ 最后 micro step 才同步 DDP 梯度
│   └─ backward
│
├─ all_reduce loss_accum
├─ clip_grad_norm
├─ get_lr(step)
├─ 写入 optimizer param_groups
├─ optimizer.step()
├─ cuda synchronize
├─ 计算 step time / tokens/sec
└─ rank 0 打印并写 train log
```

---

# 5. 训练前检查清单

## 数据

```text
[ ] train shard 存在
[ ] val shard 存在
[ ] token dtype/范围正确
[ ] HellaSwag save_to_disk 目录存在
[ ] DataLoader 的 split 文件筛选正确
```

## 模型

```text
[ ] n_embd 能整除 n_head
[ ] vocab_size 大于所有 target token id
[ ] 初始 loss 接近 ln(vocab_size)
[ ] weight tying 生效
[ ] forward shape 正确
```

## Batch

```text
[ ] total_batch_size 单位是 token
[ ] total_batch_size 可被 B*T*world_size 整除
[ ] grad_accum_steps 不为 0
[ ] B/T 不超过显存
```

## DDP

```text
[ ] 每个进程绑定正确 GPU
[ ] forward 使用 DDP model
[ ] checkpoint 只由 rank 0 保存
[ ] 独占评估前后有 barrier
[ ] 异常时销毁 process group
```

## 实验管理

```text
[ ] 日志不会误覆盖重要实验
[ ] 保存完整配置
[ ] 记录 Git commit
[ ] checkpoint 目录空间足够
```

---

# 6. 训练中监控清单

## 正确性

```text
loss 是否有限
gradient norm 是否异常
validation 是否总体下降
生成是否从乱码逐步形成结构
```

## 性能

```text
tokens/sec
step time
GPU utilization
显存 allocated/reserved
DDP 是否在评估处长时间等待
```

## 参数调整顺序

显存不足：

```text
先降 B
→ 再评估是否降 T
→ 增加 grad_accum 保持 global batch
→ BF16
→ 后续实现 activation checkpointing
```

训练不稳定：

```text
检查数据/target
→ 检查初始 loss
→ 降 LR
→ 增加 warmup
→ 检查 grad norm
→ 暂时关闭 compile/低精度定位问题
```

吞吐低：

```text
检查 GPU utilization
→ 数据加载是否阻塞
→ B/T 是否过小
→ SDPA backend
→ fused optimizer
→ TF32/BF16
→ torch.compile
→ profiler
```

---

# 7. 训练完成后的流程

## 绘图

```bash
bash scripts/plot.sh
```

输出：

```text
log/training_curves.png
```

## 交互式生成

```bash
bash scripts/generate.sh
```

终端：

```text
Prompt> Once upon a time
```

## 检查 checkpoint

```bash
ls -lh log/model_*.pt
```

## 记录实验结论

至少记录：

```text
模型配置
数据 token 数
GPU 与 world size
总训练 token
最小 train/val loss
HellaSwag 样本数与准确率
峰值 tokens/sec
生成样本
异常与修改
```

---

# 8. 当前工程的可改进

## 实验管理

1. 引入 TrainConfig；
2. 命令行参数或 YAML；
3. 每个 run 独立目录；
4. 保存完整 config 和 Git hash；
5. log 增加 lr、norm、tokens/sec。

## 性能基础设施

1. profiler；
2. DataLoader 预取/mmap；
3. torch.compile 对照 benchmark；
4. activation checkpointing；
5. 多 rank HellaSwag；
6. 更系统的吞吐与显存测量。

## 推理基础设施

1. top-p；
2. EOS；
3. KV Cache；
4. prefill/decode 分离；
5. batch generation；
6. latency/throughput benchmark。

---

# 9. 阶段总结

当前代码已经形成一个完整的小型语言模型工程闭环：

```text
可以准备数据
可以训练
可以扩展到 DDP
可以验证
可以做下游评估
可以保存
可以画图
可以重新加载并交互生成
```

下一阶段最重要的不是继续堆功能，而是把以下三件事做扎实：

1. **训练配置可复现；**
2. **性能指标能被系统测量和解释。**

这三点完成后，项目会从“GPT-2 复现练习”进一步变成真正有 AI Infra 含量的训练系统。

---

# 我的工程记录与感想

## 本次运行配置

```text
日期：
GPU：
PyTorch/CUDA：
world_size：
模型配置：
B：
T：
total_batch_size：
grad_accum_steps：
max_steps：
总训练 tokens：
```

## 最终结果

```text
最低 train loss：
最低 val loss：
HellaSwag：
峰值 tokens/sec：
checkpoint：
```
