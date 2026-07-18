# TinyLM：GPT-2 复现与训练完整学习笔记

> 本文基于当前 `tinylm-infra` 代码、原笔记 `note/karpathy/reproduce_gpt2.md`、后续学习记录以及项目推进过程中的讨论整理而成。  
> 目标不是只记录“代码怎么写”，而是建立一套从 **模型结构、数值初始化、训练优化、分布式训练、数据管线、评估到推理** 的完整理解框架。

---

## 0. 先看最重要的内容

整个 TinyLM-GPT2 阶段可以压缩为下面这条主线：

```text
GPT-2 模型结构复现
    ↓
用 HuggingFace 预训练权重验证结构正确性
    ↓
按照 GPT-2 规则初始化随机模型
    ↓
TinyStories 下载、分词并保存为 token shards
    ↓
DataLoader 按 B×T 构造 next-token prediction 数据
    ↓
AdamW + 学习率预热/余弦衰减 + 梯度裁剪
    ↓
梯度累计扩大有效 batch
    ↓
DDP 扩展到多 GPU
    ↓
validation loss + 文本采样 + HellaSwag
    ↓
日志、曲线、checkpoint 和交互式生成
```

这一阶段真正需要掌握的不是某个 API，而是六个核心关系：

1. **GPT 的输入和监督目标是错开一个 token 的序列。**
2. **模型的 shape 变化必须始终可解释。**
3. **有效 batch 是由 micro batch、序列长度、梯度累计和 GPU 数共同决定的。**
4. **DDP 同步的是梯度，不会自动替你同步所有日志、评估流程和自定义状态。**
5. **训练吞吐、显存占用和数值稳定性需要同时考虑。**
6. **validation loss、生成效果和下游评估分别观察模型的不同能力，不能互相替代。**

---

# 第一部分：GPT-2 模型结构

## 1. GPT-2 是什么结构

GPT-2 是 **decoder-only Transformer**。它不包含单独的 Encoder，也不执行双向注意力，而是根据当前位置之前的 token 预测下一个 token。

模型主体：

```text
token ids
    ↓
Token Embedding
    +
Position Embedding
    ↓
Transformer Block × n_layer
    ↓
Final LayerNorm
    ↓
LM Head
    ↓
每个位置上的词表 logits
```

每个 Transformer Block：

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

这是 **Pre-LayerNorm** 结构。LayerNorm 位于 Attention 和 MLP 之前。与 Post-LayerNorm 相比，Pre-LN 通常更利于深层网络中的梯度传播和训练稳定性。

---

## 2. `GPTConfig`：模型结构的统一入口

当前配置包含：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True
```

参数含义：

| 参数 | 含义 | 主要影响 |
|---|---|---|
| `block_size` | 最大上下文 token 数 | 注意力显存、最大输入长度 |
| `vocab_size` | 输出词表大小 | Embedding 和 LM Head 参数量 |
| `n_layer` | Transformer Block 数量 | 深度、参数量、计算量 |
| `n_head` | 多头注意力头数 | 每头维度与注意力结构 |
| `n_embd` | 隐藏维度 | 几乎所有矩阵乘法规模 |
| `bias` | 是否使用 bias | 与目标架构兼容性 |

必须满足：

```text
n_embd % n_head == 0
```

因为：

```text
head_dim = n_embd / n_head
```

每个注意力头必须获得整数维度。建议在 `CausalSelfAttention.__init__()` 中显式断言：

```python
assert config.n_embd % config.n_head == 0
```

---

## 3. Causal Self-Attention：最关键的 shape 变化

输入：

```text
x: [B, T, C]
```

其中：

- `B`：micro batch size
- `T`：sequence length
- `C`：embedding dimension，即 `n_embd`

### 3.1 一次线性映射得到 Q、K、V

GPT-2 将 Q、K、V 合并为一个线性层：

```python
qkv = self.c_attn(x)
```

shape：

```text
[B, T, C] → [B, T, 3C]
```

再切分：

```python
q, k, v = qkv.split(C, dim=2)
```

每一个都是：

```text
[B, T, C]
```

### 3.2 拆分多头

```python
q = q.view(B, T, n_head, head_dim).transpose(1, 2)
```

变化过程：

```text
[B, T, C]
→ [B, T, n_head, head_dim]
→ [B, n_head, T, head_dim]
```

`scaled_dot_product_attention()` 期望的常见输入就是：

```text
[B, n_head, T, head_dim]
```

### 3.3 因果注意力

```python
y = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
)
```

`is_causal=True` 表示位置 `t` 只能看到 `0...t`，不能看到未来 token。这正是自回归语言模型训练成立的前提。

需要严谨区分：

- `scaled_dot_product_attention` 是 PyTorch 的统一 SDPA 接口。
- 在满足硬件、dtype、shape 等条件时，PyTorch 可能选择 Flash Attention、memory-efficient attention 或普通数学实现。
- 因此，更准确的说法是：**使用 SDPA 接口，使后端有机会调用 Flash Attention 内核**，而不是无条件等同于 Flash Attention。

### 3.4 合并注意力头

attention 输出：

```text
[B, n_head, T, head_dim]
```

需要恢复为：

```text
[B, T, C]
```

代码：

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

为什么要 `.contiguous()`：

- `transpose()` 只改变张量的逻辑 stride，通常不会立刻重新排列底层数据。
- `view()` 要求内存布局可兼容。
- `.contiguous()` 会按照新的逻辑顺序生成连续内存，再安全地执行 `view()`。

最后经过输出投影：

```python
y = self.c_proj(y)
```

---

## 4. MLP：通道维度上的非线性变换

GPT-2 的 MLP：

```text
C → 4C → C
```

代码逻辑：

```python
x = self.c_fc(x)
x = GELU(x)
x = self.c_proj(x)
```

使用：

```python
nn.GELU(approximate="tanh")
```

是为了与 GPT-2/HuggingFace 的近似形式保持一致。

Attention 负责 token 之间的信息混合；MLP 主要负责每个 token 位置内部的通道变换。二者承担不同功能。

---

## 5. Block：Pre-LN 与残差连接

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

一个 Block 有两条残差分支：

1. Attention 残差分支
2. MLP 残差分支

如果有 `L` 个 Block，就有大约 `2L` 个残差分支持续向 residual stream 累加。这也是后面需要残差缩放初始化的原因。

---

## 6. GPT 主体：Embedding、Block、LM Head

### 6.1 Token Embedding

```python
tok_emb = wte(idx)
```

shape：

```text
idx:     [B, T]
tok_emb: [B, T, C]
```

### 6.2 Position Embedding

```python
pos = torch.arange(0, T, device=idx.device)
pos_emb = wpe(pos)
```

shape：

```text
pos:     [T]
pos_emb: [T, C]
```

相加时发生 broadcast：

```text
[B, T, C] + [T, C] → [B, T, C]
```

必须让 `pos` 与 `idx` 位于同一设备，否则会出现 CPU/CUDA device mismatch。

### 6.3 经过所有 Block

```python
for block in self.transformer.h:
    x = block(x)
```

shape 始终保持：

```text
[B, T, C]
```

### 6.4 输出 logits

```python
logits = self.lm_head(x)
```

shape：

```text
[B, T, vocab_size]
```

它表示：对 batch 中每个时间位置，模型都给出了整个词表上的未归一化分数。

---

## 7. Next-token prediction 与交叉熵

DataLoader 构造：

```text
buf = [t0, t1, t2, ..., tN]

x = [t0, t1, ..., tN-1]
y = [t1, t2, ..., tN]
```

也就是：

```text
x 的第 t 个位置 → 预测 y 的第 t 个 token
```

模型输出：

```text
logits:  [B, T, V]
targets: [B, T]
```

而 `F.cross_entropy` 需要：

```text
input:  [N, V]
target: [N]
```

所以展平：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
)
```

此处的 loss 默认已经对 `B×T` 个 token 求平均。

---

# 第二部分：初始化与数值正确性

## 8. 初始 loss 的 sanity check

随机初始化时，如果模型对每个 token 给出的概率近似相同：

```text
p(token) ≈ 1 / V
```

单 token 的交叉熵约为：

```text
-loss = -ln(1 / V) = ln(V)
```

因此：

- `V=50257` 时，理论初始 loss 约为 `10.8249`
- `V=50304` 时，理论初始 loss 约为 `10.8258`

你的日志中 step 0 的 loss 约为 `10.91`，与 `ln(50304)` 同量级，属于合理的初始检查结果。

这不是要求初始 loss 精确等于 `ln(V)`。初始化、LayerNorm、有限样本、logits 方差都会造成偏差。它的价值是快速发现明显错误，例如：

- loss 一开始就极大或出现 NaN；
- logits 对少量 token 存在异常强烈偏好；
- target 对齐错误；
- 词表大小、数据 token id 或输出维度不匹配。

---

## 9. Weight Tying

```python
self.transformer.wte.weight = self.lm_head.weight
```

输入端：

```text
token id → embedding vector
```

输出端：

```text
hidden state → 与每个 token embedding 的匹配分数
```

共享权重的好处：

1. 减少参数量；
2. 减少显存占用；
3. 输入和输出使用同一个 token 表示空间；
4. 与 GPT-2 结构保持一致。

注意：如果用 `state_dict`、checkpoint 或 HuggingFace 权重验证，必须确认共享关系仍然成立。

---

## 10. GPT-2 权重初始化

普通 Linear：

```text
weight ~ Normal(0, 0.02)
bias = 0
```

Embedding：

```text
weight ~ Normal(0, 0.02)
```

实现方式：

```python
self.apply(self._init_weight)
```

`apply()` 会递归遍历子模块并执行初始化函数。

### 10.1 为什么不能完全依赖 PyTorch 默认初始化

显式初始化有三个作用：

1. 与 GPT-2 的训练设定对齐；
2. 控制初始 logits 与中间激活的尺度；
3. 防止深层残差累积导致方差持续扩大。

---

## 11. 残差缩放初始化

每个 Block 有两个输出投影进入残差流：

- `Attention.c_proj`
- `MLP.c_proj`

给它们添加标记：

```python
self.c_proj.NANOGPT_SCALE_INIT = 1
```

初始化时：

```python
std = 0.02 * (2 * n_layer) ** -0.5
```

直觉推导：

假设有 `N` 个近似独立的残差分支，每个分支方差为 `σ²`：

```text
Var(r1 + r2 + ... + rN) ≈ Nσ²
```

为了让累积后的方差仍维持在相同量级，每个分支方差应缩小到原来的 `1/N`。标准差是方差的平方根，因此标准差缩放为：

```text
1 / sqrt(N)
```

GPT-2 中按 `N≈2L` 处理，所以：

```text
std = 0.02 / sqrt(2L)
```

这不是对所有 Linear 都缩放，只对进入 residual stream 的输出投影缩放。

---

# 第三部分：加载 HuggingFace GPT-2 权重

## 12. 为什么先复现模型，再加载官方权重

从头实现的 GPT 类如果能无误加载 HuggingFace GPT-2 权重，并生成合理文本，说明以下内容大概率正确：

- 模块层级；
- 参数名称；
- 参数 shape；
- Attention 的 QKV 拆分；
- Block 顺序；
- LayerNorm 位置；
- Embedding 和 LM Head；
- forward 的大部分逻辑。

这是比“随机模型能 forward”更强的结构正确性验证。

---

## 13. `from_pretrained()` 的关键步骤

### 13.1 根据模型规格建立本地模型

例如 GPT-2 small：

```text
n_layer = 12
n_head  = 12
n_embd  = 768
vocab_size = 50257
block_size = 1024
```

### 13.2 过滤不需要的 buffer

HuggingFace 旧式 attention 实现中可能包含：

```text
attn.bias
attn.masked_bias
```

这些用于保存 causal mask 等 buffer。当前模型使用：

```python
F.scaled_dot_product_attention(..., is_causal=True)
```

因此本地实现不需要这些 buffer，参数对齐时应过滤。

### 13.3 Conv1D 与 Linear 的权重方向

HuggingFace GPT-2 延续了 OpenAI `Conv1D` 风格，其部分权重布局与 `nn.Linear` 相反。需要转置的常见权重：

```text
attn.c_attn.weight
attn.c_proj.weight
mlp.c_fc.weight
mlp.c_proj.weight
```

复制前必须检查：

```python
assert sd_hf[k].shape[::-1] == sd[k].shape
```

然后：

```python
sd[k].copy_(sd_hf[k].t())
```

其余参数直接 `copy_()`。

为什么用 `copy_()`：

- 保留目标 `Parameter` 对象；
- 只修改底层数值；
- 不破坏模块注册、共享参数关系和优化器引用。

---

# 第四部分：自回归生成

## 14. `generate()` 的执行过程

每次只生成一个新 token：

```text
当前 token 序列
    ↓
截取最后 block_size 个 token
    ↓
forward
    ↓
取最后时间位置 logits
    ↓
temperature
    ↓
top-k 过滤
    ↓
softmax
    ↓
multinomial 采样
    ↓
拼接新 token
```

### 14.1 为什么只取最后一个位置

```python
logits = logits[:, -1, :]
```

当前只需要预测“下一个 token”，前面位置的预测已经不再使用。

### 14.2 上下文裁剪

```python
idx_cond = idx[:, -block_size:]
```

GPT-2 的 position embedding 只支持 `block_size` 范围。序列增长后，只保留最近的上下文。

当前实现没有 KV Cache，因此每生成一个 token 都会重新计算整个有效上下文，复杂度较高。这正是后续 inference infra 中需要优化的部分。

### 14.3 Temperature

```python
logits = logits / temperature
```

- `temperature < 1`：分布更尖锐，更保守；
- `temperature > 1`：分布更平，更随机；
- `temperature` 必须大于 0。

### 14.4 Top-k

保留最大的 k 个 logits，其余设为负无穷：

```python
v, _ = torch.topk(logits, k)
logits[logits < v[:, [-1]]] = -float("inf")
```

然后再 softmax。应当在 logits 层面过滤，而不是先 softmax 后过滤。

### 14.5 随机数生成器

训练中采样使用固定 seed：

```python
sample_rng.manual_seed(42)
```

这样不同训练 step 的样本更容易横向比较。交互式生成可根据需要使用固定或随机 seed。

---

# 第五部分：训练效率与数值优化

## 15. TF32

```python
torch.set_float32_matmul_precision("high")
```

在支持 TF32 的 NVIDIA GPU 上，允许 float32 矩阵乘法采用更高吞吐的内部路径。它主要影响 matmul，不等于把模型参数整体改成低精度。

是否启用应通过：

- tokens/sec；
- validation loss；
- 训练稳定性；

共同验证，而不是默认认为一定更好。

---

## 16. BF16 混合精度

```python
with torch.autocast(
    device_type=device_type,
    dtype=torch.bfloat16,
):
    logits, loss = model(x, y)
```

混合精度的核心不是“所有参数都变成 BF16”，而是 autocast 根据算子类型选择较合适的执行精度。模型参数通常仍保存为 FP32，部分矩阵运算使用 BF16。

BF16 相比 FP16：

- 指数范围接近 FP32；
- 更不容易 overflow；
- 通常不需要 GradScaler；
- 需要 GPU 支持。

必须记录硬件与 PyTorch 环境，因为低精度支持取决于设备。

---

## 17. SDPA 与 Flash Attention

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

它可能减少：

- 显式 `T×T` attention matrix 的中间显存；
- HBM 读写；
- kernel launch 与中间张量开销。

实际使用哪个后端由 PyTorch 调度。性能验证要看 profiler，而不能仅凭函数名判断。

---

## 18. `torch.compile`

```python
model = torch.compile(model)
```

潜在收益：

- 图捕获；
- kernel fusion；
- 降低 Python 调度开销；
- 对重复 shape 的训练循环进行优化。

注意：

1. 首次编译会有明显开销；
2. 动态 shape 和 Python 控制流可能触发 graph break；
3. checkpoint key 可能带 `_orig_mod.` 前缀；
4. DDP 与 compile 的组合需要按 PyTorch 版本测试；
5. 当前训练代码在 `use_compile=True` 时跳过采样和 HellaSwag，这属于当前工程限制，不是 compile 的理论要求。

---

## 19. 词表 padding：50257 → 50304

当前训练模型使用：

```python
GPTConfig(vocab_size=50304)
```

而 GPT-2 tokenizer 的实际 token id 范围是 `0...50256`。

多出的 47 个输出类别不会出现在训练 targets 中，但扩大到更适合硬件对齐的维度，可能让 GEMM 更高效。更严谨的解释是：

- 50304 具有更友好的对齐因子；
- Tensor Core/GEMM 内核通常偏好特定 tile 的倍数；
- 性能收益取决于 GPU、dtype、矩阵 shape 和内核实现；
- 不是“只要是 2 的整数次幂就一定更快”。

副作用：

- LM Head 和 Embedding 多出少量参数；
- 初始理论 loss 应按 50304 计算；
- 生成时理论上可能抽到额外 token id。因为这些 token 从未作为 target 出现，充分训练后概率通常会降低，但严格实现可在采样时限制到 tokenizer 的有效词表范围。

---

# 第六部分：优化器与学习率

## 20. AdamW

当前配置：

```python
AdamW(
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)
```

AdamW 将 weight decay 与自适应梯度更新解耦。它维护：

- 一阶动量；
- 二阶动量；
- bias correction；
- 参数更新；
- decoupled weight decay。

---

## 21. Weight Decay 参数分组

当前实现按维度分组：

```text
param.dim() >= 2  → decay
param.dim() < 2   → no decay
```

通常：

- Linear/Embedding 矩阵：执行 decay；
- bias：不 decay；
- LayerNorm scale/bias：不 decay。

原因：

- 大型矩阵决定主要的线性变换和表示空间，限制其无限增长具有正则意义；
- bias 主要执行平移；
- LayerNorm 参数负责归一化后的缩放和平移；
- 将归一化 scale 强行衰减到 0 可能削弱表示能力。

按维度分组是一种简洁工程规则。更严格的实现也可以按模块类型和参数名显式分类。

---

## 22. Fused AdamW

如果当前 PyTorch 的 AdamW 支持 `fused`，且设备为 CUDA：

```python
fused=True
```

可以将多个逐元素更新步骤融合，减少：

- GPU kernel launch；
- 中间显存访问；
- CPU/Python 调度。

必须做兼容检查，因为并非所有设备、dtype 和 PyTorch 版本都支持。

---

## 23. 学习率预热与余弦衰减

当前调度：

```text
0 → max_lr：线性 warmup
max_lr → min_lr：cosine decay
超过 max_steps：保持 min_lr
```

warmup：

```python
lr = max_lr * (step + 1) / warmup_steps
```

余弦衰减：

```python
ratio = (step - warmup_steps) / (max_steps - warmup_steps)
coeff = 0.5 * (1 + cos(pi * ratio))
lr = min_lr + coeff * (max_lr - min_lr)
```

为什么 warmup：

- 初始化阶段梯度统计尚未稳定；
- Adam 的动量估计刚开始建立；
- 直接使用峰值学习率更容易产生不稳定更新。

当前代码中：

```text
max_lr = 6e-4
min_lr = 6e-5
warmup_steps = 10
max_steps = 51
```

这是一个短流程 smoke test，不是完整 TinyStories 预训练计划。

---

## 24. 梯度裁剪

```python
norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0,
)
```

它先计算所有参数梯度的整体 L2 norm。若超过阈值，就统一按比例缩小。

作用：

- 避免某一次异常梯度让参数发生过大跳变；
- 为日志提供 `norm` 监控指标；
- 它不能修复长期错误的学习率、数据问题或 NaN 根因。

---

# 第七部分：梯度累计

## 25. 为什么需要梯度累计

单次显存只能容纳一个较小的 micro batch，但希望一次参数更新覆盖更多 token。

定义：

```text
每个进程每个 micro step 的 token 数 = B × T
全局每次参数更新的 token 数
= B × T × grad_accum_steps × world_size
```

因此：

```text
grad_accum_steps
= total_batch_size / (B × T × world_size)
```

必须断言整除：

```python
assert total_batch_size % (B*T*world_size) == 0
```

### 当前配置

```text
total_batch_size = 16384 tokens
B = 4
T = 512
```

单卡：

```text
grad_accum_steps = 16384 / (4×512) = 8
```

两卡：

```text
grad_accum_steps = 16384 / (4×512×2) = 4
```

四卡：

```text
grad_accum_steps = 2
```

八卡：

```text
grad_accum_steps = 1
```

---

## 26. 为什么 loss 要除以 `grad_accum_steps`

每个 micro batch 的 cross entropy 已经是该 micro batch 内 token loss 的均值。

若直接反向传播并累加 `K` 次，梯度会成为 `K` 个均值梯度之和，而不是等价大 batch 的均值梯度。

所以：

```python
loss = loss / grad_accum_steps
loss.backward()
```

如果每个 micro batch token 数相同，那么最终梯度等价于全局大 batch 的平均梯度。

---

## 27. `loss_accum` 为什么要 `detach()`

```python
loss_accum += loss.detach()
```

日志只需要数值，不需要把累加器连接到 autograd graph。`detach()` 可避免保留额外计算图和显存。

---

# 第八部分：DDP 多 GPU 训练

## 28. DDP 的基本进程模型

通常一张 GPU 对应一个进程。

环境变量：

| 变量 | 含义 |
|---|---|
| `RANK` | 全局进程编号 |
| `LOCAL_RANK` | 当前节点上的 GPU 编号 |
| `WORLD_SIZE` | 总进程数 |

初始化：

```python
init_process_group(backend="nccl")
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
```

`rank == 0` 通常作为 `master_process`，负责：

- 打印日志；
- 写文件；
- 保存 checkpoint；
- 执行只需一次的可视化或样本输出。

---

## 29. `model` 与 `raw_model`

```python
model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model
```

职责区分：

### `model`

用于：

```text
forward
backward
DDP gradient synchronization
```

### `raw_model`

用于：

```text
访问 config
configure_optimizers()
generate()
state_dict()
自定义模型方法
```

训练 forward 必须经过 DDP wrapper，否则 DDP hook 不会正确工作。

---

## 30. DDP 数据划分

每个 rank 的起始位置：

```text
current_position = B × T × process_rank
```

每次前进：

```text
B × T × num_processes
```

例如两个进程：

```text
rank 0: 第 0 块、第 2 块、第 4 块……
rank 1: 第 1 块、第 3 块、第 5 块……
```

这样在同一个 micro step 中，各进程读取不同 token 区间。

---

## 31. 梯度累计期间减少 DDP 通信

DDP 默认每次 `backward()` 都同步梯度。但前面的 micro step 只需要本地累积，最后一个 micro step 再同步即可。

当前方式：

```python
model.require_backward_grad_sync = (
    micro_step == grad_accum_steps - 1
)
```

它与使用 `model.no_sync()` 的目的相同：减少不必要的 all-reduce 通信。

---

## 32. 同步训练 loss

梯度由 DDP 自动同步，但每个进程的 `loss_accum` 是普通张量，不会自动同步。

```python
dist.all_reduce(
    loss_accum,
    op=dist.ReduceOp.AVG,
)
```

这样主进程打印的是各 rank 的平均训练 loss。

需要理解：同步 loss 只是为了日志一致，不会影响已经计算出的梯度。

---

## 33. Validation loss 的同步

validation 由所有 rank 共同执行，各自读取不同数据，再：

```python
dist.all_reduce(
    val_loss_accum,
    op=dist.ReduceOp.AVG,
)
```

`all_reduce` 本身要求所有进程参与，因此已经形成集体同步点，通常不需要在它旁边再加 barrier。

---

## 34. Rank 0 独占评估的同步问题

生成和当前 HellaSwag 评估只在 rank 0 上运行。若其他 rank 直接进入下一轮训练，它们会先到达下一次 DDP 通信，而 rank 0 仍在评估，可能长时间等待甚至触发 NCCL timeout。

正确的最小改法：

```python
if ddp:
    dist.barrier()

if master_process:
    # generate 或 evaluate_hellaswag
    ...

if ddp:
    dist.barrier()
```

这应分别包围：

- rank 0 文本采样；
- rank 0 HellaSwag 评估。

**当前压缩包中的 `train_gpt2.py` 尚未真正加入这些 barrier。笔记已经记录了问题，但代码仍需要补上。**

更高效的后续方案是让所有 rank 分担 HellaSwag 样本，再 all-reduce 正确数和样本数。但当前 200 条小规模评估使用 barrier 方案更简单。

---

# 第九部分：TinyStories 数据处理

## 35. 为什么使用 TinyStories

TinyStories 规模明显小于通用网页语料，更适合：

- 验证训练链路；
- 在有限硬件上观察 loss 下降；
- 快速完成端到端项目；
- 生成相对结构完整的短故事。

它仍然不是“小到可以整体长期放在 Python list 中”的数据，所以采用 token shard。

---

## 36. Tokenization

每篇文档前加入：

```text
<|endoftext|>
```

作用：

- 标记文档边界；
- 防止相邻故事被模型无条件视为同一篇连续文本；
- 让模型学习文档结束/开始模式。

使用：

```python
enc.encode_ordinary(doc["text"])
```

避免普通文本被意外解释为 special token。

---

## 37. 为什么保存为 `uint16`

GPT-2 token id 小于 `2^16`，因此 token 可以保存为：

```text
uint16
```

相较 `int32/int64`，磁盘占用更小。

加载后转为：

```text
torch.long
```

因为 PyTorch Embedding 的索引通常要求 long 类型。

---

## 38. 为什么分 shard

token shard 的收益：

1. 不必一次加载完整数据；
2. 降低内存压力；
3. 便于 train/validation 分离；
4. 便于 DDP 按连续区间读取；
5. 数据准备失败时不必全部重做；
6. 便于统计、调试和后续断点恢复；
7. 可扩展为异步预取、mmap 或多 worker pipeline。

当前每个 shard：

```text
1,000,000 tokens
```

文件名：

```text
tinystories_train_000000.npy
tinystories_train_000001.npy
...
tinystories_val_000000.npy
```

与 Karpathy FineWeb 代码不同，TinyStories 本身有官方 validation split，因此无需把训练集第一个 shard 人为当作验证集。

---

## 39. 多进程 tokenization

```python
with mp.Pool(nprocs) as pool:
    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        ...
```

CPU 多进程并行处理文档。主进程负责：

- 拼接 token；
- 维护 shard buffer；
- 写 `.npy`；
- 更新进度条。

注意事项：

- `os.cpu_count() // 2` 只是经验值；
- 多进程过多可能增加内存和调度开销；
- Windows/WSL、容器和共享内存环境需要实际测试。

---

# 第十部分：DataLoaderLite

## 40. 两种数据来源

当前 DataLoader 支持：

1. 单个本地文本文件；
2. 预处理后的 `.npy` shard。

训练主流程使用 shard。

---

## 41. Batch 构造

```python
buf = tokens[pos : pos + B*T + 1]
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
```

多取一个 token 是为了构造右移后的 target。

---

## 42. Shard 切换

当前 shard 剩余 token 不足以覆盖后续各 rank 的 batch 时：

```text
current_shard = (current_shard + 1) % num_shards
重新加载 token
重置 current_position
```

到最后一个 shard 后回到第一个 shard，相当于进入下一 epoch。

当前 DataLoader 没有随机 shuffle，读取顺序固定。对学习工程足够直观，但正式训练可进一步考虑：

- 每个 epoch 打乱 shard 顺序；
- 固定随机 seed；
- 保存 shard index 与 position 以支持精确 resume；
- 异步预取下一个 shard。

---

# 第十一部分：训练步数与 token 预算

## 43. `total_batch_size` 的真实含义

这里的单位是：

```text
tokens / optimizer step
```

不是样本条数，也不是 `B`。

每次 `optimizer.step()` 处理：

```text
total_batch_size
= B × T × grad_accum_steps × world_size
```

总训练 token 数：

```text
total_tokens = total_batch_size × max_steps
```

若数据集 token 数为 `D`，近似 epoch 数：

```text
epochs ≈ total_tokens / D
```

---

## 44. 当前参数属于 smoke test

当前：

```text
total_batch_size = 16,384
max_steps = 51
```

只训练：

```text
16,384 × 51 = 835,584 tokens
```

相对于约 473,992,236 个 TinyStories token，仅约：

```text
0.176%
```

因此当前训练的目的主要是验证：

- loss 能否下降；
- 数据是否正常；
- DDP 是否能运行；
- validation/HellaSwag/checkpoint/generate 是否闭环。

若希望约训练一遍数据：

```text
max_steps ≈ floor(dataset_tokens / total_batch_size)
```

- `total_batch_size=16,384` 时约 `28930` steps；
- `total_batch_size=524,288` 时约 `904` steps。

实际训练不一定必须正好一 epoch，需要根据：

- 模型规模；
- 计算预算；
- validation loss；
- 过拟合趋势；
- 生成质量；

共同决定。

---

# 第十二部分：验证、采样与 HellaSwag

## 45. Validation loss

每隔一定 step：

1. `model.eval()`；
2. `val_loader.reset()`；
3. `torch.no_grad()`；
4. 连续取多个 validation batch；
5. 求平均；
6. DDP 下 all-reduce；
7. rank 0 记录日志。

当前：

```text
val_loss_steps = 20
eval interval = 50 steps
```

validation loss 衡量模型对未参与当前更新的验证 token 的平均 next-token 预测能力。

训练 loss 与 validation loss 的关系：

- 两者一起下降：通常表示模型在学习；
- train 降、val 持平或上升：可能过拟合或分布差异；
- 两者剧烈波动：检查学习率、batch、数据、精度和实现；
- 短训练中单次 val 有噪声，不宜过度解读。

---

## 46. 训练中采样

固定 prompt：

```text
Hello, I'm a language model,
```

生成多条序列并固定随机 seed。

采样的意义：

- 直观观察字符/单词/句法是否逐步形成；
- 发现重复、乱码、特殊 token 等问题；
- 与 loss 形成互补。

采样不能替代定量评估，因为少量样本容易受随机性和 prompt 影响。

---

## 47. HellaSwag 在评估什么

一个样本有：

```text
context
4 个 ending
正确 label
```

Causal LM 没有分类头，所以对每个候选构造：

```text
context + ending_i
```

只计算 ending token 的平均负对数似然，平均 loss 最低者作为预测。

### 为什么只统计 ending

四个选项共享 context。评估目标是：

```text
P(ending | context)
```

共享 context 的损失不应影响候选间比较。

### 为什么使用平均 loss

ending 长度不同。总 loss 会天然惩罚较长 ending，所以使用：

```text
ending loss sum / ending token count
```

---

## 48. `render_example()`

输出：

```text
tokens: [4, max_len]
mask:   [4, max_len]
label:  int
```

- context mask 为 0；
- ending mask 为 1；
- padding mask 为 0。

padding token id 使用 0，但 padding 不参与 loss，因此其具体 id 不影响候选分数。

---

## 49. Shift 的必要性

GPT 在位置 `t` 的 logits 预测位置 `t+1`：

```python
shift_logits = logits[:, :-1, :]
shift_tokens = tokens[:, 1:]
shift_mask = mask[:, 1:]
```

如果 mask 不一起 shift，就会把“输入位置”与“被预测位置”错配。

---

## 50. 完整 HellaSwag 流程

```text
提前下载 validation split 并 save_to_disk
    ↓
load_from_disk
    ↓
遍历 example
    ↓
render_example
    ↓
检查是否超过 block_size
    ↓
forward
    ↓
token-level cross entropy
    ↓
只保留 ending mask
    ↓
每个 ending 求平均 loss
    ↓
argmin
    ↓
统计准确率与 skipped
```

训练阶段仅评估 200 条样本是为了降低时间成本。小样本准确率方差较大，不能等同完整 benchmark 结果。

---

# 第十三部分：日志、Checkpoint、绘图和交互式生成

## 51. 日志格式

```text
step stream value
```

例如：

```text
0 train 10.914177
0 val 10.9127
0 hella 0.2450
```

优点：

- 简单；
- 易解析；
- 可扩展不同 stream；
- 不依赖 TensorBoard/W&B。

缺点：

- 不包含时间、学习率、梯度 norm、tokens/sec；
- 恢复训练时会被当前代码清空；
- 不支持多实验自动分目录。

---

## 52. Checkpoint

当前保存：

```python
{
    "model": raw_model.state_dict(),
    "config": raw_model.config,
    "step": step,
    "val_loss": val_loss,
}
```

它足够用于：

```text
加载模型 → eval → generate
```

但不足以严格 resume training。完整恢复还应保存：

- `optimizer.state_dict()`；
- scheduler 或当前 step；
- CPU RNG state；
- CUDA RNG state；
- DataLoader 当前 shard 和 position；
- AMP scaler（如果使用 FP16 + GradScaler）；
- 实验配置；
- Git commit 或代码版本。

需要区分：

```text
load checkpoint for inference
≠
resume training exactly
```

---

## 53. 交互式生成

`generate.py`：

1. 在 `log/` 中寻找最新 `model_*.pt`；
2. 加载 config 与 state_dict；
3. 清理 `_orig_mod.` 或 `module.` 前缀；
4. 模型只加载一次；
5. 循环读取终端 prompt；
6. 按回车生成；
7. 输入 `q/quit/exit` 退出。

这种方式比每次命令行传 prompt 更适合反复测试模型。

---

## 54. 绘图

日志解析后绘制：

- train loss；
- validation loss；
- HellaSwag accuracy。

注意：

- 不应随意固定过小的 y 轴上限，否则会裁掉初期 loss；
- 训练 loss 点远多于 validation 点是正常的；
- HellaSwag 只评估少量样本时，曲线可能有较大噪声；
- 绘图脚本不应在 stream 不存在时强制调用空 legend。

---

# 第十四部分：当前实现中的严谨性检查

## 55. 已经完成并形成闭环的部分

```text
GPT-2 结构
HuggingFace 权重迁移
自定义初始化
生成
AdamW 参数分组
梯度累计
DDP
TinyStories shard
validation
HellaSwag
checkpoint
日志与绘图
交互式推理
Shell 启动脚本
```

---

## 56. 仍应修正或补充的部分

### 模型结构

- 增加 `n_embd % n_head == 0` 断言。
- 当前 `GPTConfig.bias` 没有真正传入 Linear/LayerNorm；若未来设置 `bias=False`，代码不会按配置生效。
- `temperature <= 0` 应显式报错。
- vocab padding 后可考虑限制生成到 tokenizer 的有效词表范围。

### DDP

- 采样和 HellaSwag 前后加入 `dist.barrier()`。
- 最好使用 `try/finally` 确保异常时销毁 process group。
- 后续让 HellaSwag 多 rank 并行，而不是只在 rank 0 运行。

### DataLoader

- `num_tokens()` 的文本文件分支尚未实现。
- 精确 resume 需要保存 `current_shard/current_position`。
- 当前不 shuffle shard。
- 可加入预取和 mmap。

### Checkpoint

- 增加 optimizer、RNG、DataLoader 状态。
- 避免每次启动都直接清空 `log.txt`。
- 将每次实验放入独立目录，防止覆盖。

### 测试

- `from_pretrained` 测试硬编码 CUDA 和本地路径，应标记为集成测试并按环境 skip。
- 增加 block size 超限测试。
- 增加 weight tying、初始化尺度、optimizer 分组测试。
- 增加 DataLoader DDP 不重叠测试。
- 增加 checkpoint load/generate 测试。

---

# 第十五部分：如何判断训练是否健康

建议同时观察：

| 指标 | 作用 |
|---|---|
| train loss | 当前训练数据拟合情况 |
| val loss | 泛化到验证集的 next-token 能力 |
| gradient norm | 梯度是否异常 |
| learning rate | 与 loss 波动对应 |
| tokens/sec | 吞吐和性能回归 |
| step time | 是否出现数据或同步阻塞 |
| GPU utilization | GPU 是否被喂饱 |
| memory allocated/reserved | 显存瓶颈 |
| HellaSwag accuracy | 有限的常识续写能力 |
| generated samples | 直观语言质量 |

异常排查顺序：

```text
数据与 target 对齐
→ token id 是否越界
→ 初始 loss 是否接近 ln(V)
→ 是否出现 NaN/Inf
→ 学习率与梯度 norm
→ autocast/精度
→ DDP 数据与同步
→ 性能瓶颈
```

---

# 第十六部分：阶段总结

这一阶段的核心成果不是“训练出了一个很强的模型”，而是亲手建立了一个能够运行的语言模型训练系统：

```text
文本
→ tokenizer
→ token shards
→ distributed dataloader
→ GPT forward
→ loss
→ backward
→ optimizer
→ validation
→ benchmark
→ checkpoint
→ inference
```

从 AI Infra 的角度，最重要的认知是：

1. **模型代码只是训练系统的一部分。**
2. **shape、dtype、device、数据位置和同步语义同样决定正确性。**
3. **吞吐优化必须建立在数值正确和可复现的基础上。**
4. **梯度累计与 DDP 共同决定全局 batch，不能分别孤立理解。**
5. **日志、checkpoint、评估和生成不是附属功能，而是训练闭环的必要组成。**
6. **当前项目已经从“跟写模型”进入“构建训练基础设施”的阶段。**

---

# 我的理解与感想

> 建议在完成一次更长训练、一次 DDP 实验和一次 checkpoint 恢复后再填写。

## 1. 我现在真正理解了什么

```text



```

## 2. 我仍然模糊的地方

```text



```

## 3. 我在调试中遇到的最关键问题

```text



```

## 4. 这次复现对我理解 AI Infra 的改变

```text



```

## 5. 下一阶段我准备验证的假设

```text



```
