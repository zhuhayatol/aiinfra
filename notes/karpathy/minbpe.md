## minBPE Tokenizer 实现笔记

当前 tokenizer 主要由三个文件组成：

```text
base.py：通用工具函数和 Tokenizer 基类
basic.py：Basic BPE Tokenizer
regex.py：RegexTokenizer，支持 regex 预分词和 special tokens
```

整体关系是：

```text
Tokenizer 基类提供通用能力
BPETokenizer 继承 Tokenizer，实现普通 byte-level BPE
RegexTokenizer 继承 Tokenizer，实现 regex 分块后的 byte-level BPE，并进一步支持 special tokens
```

---

## 1. Tokenizer 基类

`base.py` 中主要包含两部分：

```text
工具函数：get_stats、merge
Tokenizer 基类：decode、train、encode、save、load、save_vocab、_build_vocab_
```

其中 `train()` 和 `encode()` 在基类中不具体实现，而是交给子类补全。

### 1.1 `get_stats`

`get_stats()` 用于统计当前 token 序列中相邻 pair 的出现频次。

例如输入：

```python
[1, 2, 3, 2, 3]
```

会统计：

```text
(1, 2): 1
(2, 3): 2
(3, 2): 1
```

它既用于训练阶段寻找最高频 pair，也用于编码阶段判断当前还有哪些 pair 可以根据已有 merges 合并。

### 1.2 `merge`

`merge(tokens, pair, idx)` 用于扫描 token 序列，将所有非重叠出现的指定 pair 替换为新的 token id。

例如：

```text
tokens = [1, 2, 1, 2, 3]
pair = (1, 2)
idx = 256
```

合并后得到：

```text
[256, 256, 3]
```

这个函数同时用于：

```text
train 阶段：把最高频 pair 合并成新 token
encode 阶段：根据已经学好的 merges 合并新文本
```

### 1.3 `_build_vocab_`

`_build_vocab_()` 根据 `merges` 构造 `vocab`。

首先初始化 0~255 的基础 byte token：

```text
0 -> b"\x00"
1 -> b"\x01"
...
255 -> b"\xff"
```

随后遍历 `merges`。如果有：

```text
(201, 4) -> 258
```

则构造：

```python
vocab[258] = vocab[201] + vocab[4]
```

因为 `vocab[201]` 和 `vocab[4]` 都是 bytes，所以拼接后的结果仍然是 bytes。

加入 special tokens 后，`_build_vocab_()` 还需要把 special token 也加入 vocab：

```python
vocab[special_id] = special_token.encode("utf-8")
```

这样基类 `decode()` 才能直接还原 special token。

### 1.4 `decode`

`decode()` 的作用是将 token ids 还原为文本。

流程是：

```text
token ids
-> 按 id 查 vocab，得到每个 token 对应的 bytes
-> b"".join() 拼接成完整 bytes
-> UTF-8 decode 成字符串
```

例如：

```python
ids = [260, 32, 97]
tokens = b"".join(vocab[idx] for idx in ids)
text = tokens.decode("utf-8")
```

### 1.5 `save / load / save_vocab`

`save()` 会把 tokenizer 的训练结果保存为 `.model` 文件，主要包括：

```text
version
pattern
special_tokens
merges
```

`.model` 中保存的是 merges 的顺序，而不是完整 vocab。因为每条 merge 的 id 是从 256 开始按顺序递增的，所以 load 时只要按顺序读回 pair，就能恢复对应的 token id。

示例：

```text
minbpev1

0
115 101
44 32
257 118
```

含义是：

```text
minbpev1：版本号
空行：pattern，目前 BasicTokenizer 不使用
0：special token 数量
后续每行：一条 merge pair
```

`load()` 读取 `.model` 后，需要恢复：

```text
pattern
special_tokens
merges
vocab
```

其中 `vocab` 不直接从文件读取，而是根据恢复出的 `merges` 和 `special_tokens` 重新构建。

`save_vocab()` 用于生成方便人阅读的 `.vocab` 文件，可以展示新 token 是由哪两个子 token 拼接得到的，例如：

```text
[s][e] -> [se] 256
[,][ ] -> [, ] 257
[de][f] -> [def] 260
```

---

## 2. BPETokenizer

`BPETokenizer` 位于 `basic.py`，继承自 `Tokenizer`。

它主要补全两个方法：

```text
train()
encode()
```

### 2.1 `train`

训练流程是：

```text
文本
-> UTF-8 编码成 bytes
-> 转为 byte id 列表
-> 统计相邻 pair 频次
-> 找到最高频 pair
-> 将该 pair merge 成新的 token id
-> 重复直到达到目标 vocab_size
```

训练结束后得到：

```text
merges: pair -> token_id
vocab: token_id -> bytes
```

### 2.2 `encode`

编码流程是：

```text
文本
-> UTF-8 bytes
-> byte id 列表
-> 根据已训练好的 merges 反复合并
-> 返回 token ids
```

需要注意：`encode()` 阶段不重新统计全局最高频 pair，而是在当前文本已有 pair 中，选择在 `merges` 中优先级最高的 pair 进行合并。

---

## 3. RegexTokenizer

`RegexTokenizer` 位于 `regex.py`。

它与 `BPETokenizer` 的核心区别是：

```text
BPETokenizer：直接对整段文本做 BPE
RegexTokenizer：先用 regex 把文本切成多个 chunk，再在每个 chunk 内部单独做 BPE
```

### 3.1 `train`

RegexTokenizer 的训练流程是：

```text
文本
-> regex 分块，得到多个 chunk
-> 每个 chunk 转为 byte id 列表
-> 每轮对每个 chunk 单独统计 pair
-> 汇总所有 chunk 内部 pair 的频次
-> 选择全局最高频 pair
-> 对每个 chunk 分别执行 merge
```

关键点是：

```text
统计 pair 时可以汇总所有 chunk 的频次
但 merge 时必须在每个 chunk 内部分别执行
不能把所有 chunk 拼成一条长序列
```

否则会产生跨 chunk 边界的 pair，破坏 regex 分块的意义。

### 3.2 `encode_ordinary`

`encode_ordinary()` 是普通文本编码流程，不考虑 special tokens。

流程是：

```text
文本
-> regex 分块
-> 每个 chunk 内部根据已有 merges 执行 BPE
-> 拼接所有 chunk 的 token ids
```

---

## 4. Special Tokens

Special token 不是通过 BPE 训练得到的，而是人为指定的特殊控制 token，例如：

```text
<|endoftext|>
```

它在编码时不能被 regex 拆开，也不能参与普通 BPE merge，而是应该被整体识别，并直接映射为固定 id。

### 4.1 新增数据结构

在 `RegexTokenizer` 中维护：

```text
special_tokens: str -> int
inverse_special_tokens: int -> str
```

例如：

```text
"<|endoftext|>" -> 320
320 -> "<|endoftext|>"
```

其中 `special_tokens` 用于 encode。由于 `_build_vocab_()` 已经会把 special tokens 加入 `self.vocab`，所以当前实现可以继续复用基类 `decode()`，不需要单独重写 `RegexTokenizer.decode()`。

### 4.2 `register_special_tokens`

`register_special_tokens()` 用于注册 special tokens，主要完成：

```text
1. 保存 self.special_tokens
2. 构造 self.inverse_special_tokens
3. 重新构建 self.vocab
```

注意：special token 不应该加入 `merges`。`merges` 只记录普通 BPE 的 pair 合并规则，而 special token 是人为指定的整体 token。

### 4.3 `train()` 不处理 special tokens

`train()` 只负责学习普通文本中的 BPE merge 规则。special tokens 不是训练出来的，而是在训练完成后注册进去的。

因此整体流程是：

```text
train()：学习普通 BPE merges
register_special_tokens()：注册特殊 token
encode()：普通文本走 BPE，special token 直接变 id
```

如果训练文本中本身包含 `<|endoftext|>`，当前实现会把它当作普通文本处理。正式使用时，应避免让 special token 字符串进入 BPE 训练语料。

### 4.4 新的 `encode`

加入 special tokens 后，新的 `encode()` 负责处理普通文本和 special token 的混合输入。

它本质上是一个分流函数：

```text
普通文本片段 -> encode_ordinary()
special token 片段 -> 直接 append special id
```

`allowed_special` 用于控制 special token 的处理方式：

```text
"none"：不识别 special token，全部按普通文本处理
"all"：允许所有已注册 special tokens 被整体识别
"none_raise"：不允许文本中出现 special token，出现则报错
set(...)：只允许集合中指定的 special tokens 被整体识别
```

### 4.5 special token 的切分

special token 的优先级高于 regex 分块和 BPE merge：

```text
special token 边界 > regex chunk 边界 > BPE merge 边界
```

因此 `encode()` 中要先按 special token 切分文本，再对普通文本片段调用 `encode_ordinary()`。

核心步骤：

```text
1. 根据 allowed_special 得到本次允许识别的 special token 集合
2. 按长度从大到小排序，避免短 token 抢先匹配
3. 使用 re.escape() 转义 special token
4. 用捕获组构造 special_pattern
5. 用 re.split() 切分文本
6. 遍历切分结果：
   - special token 直接变 id
   - 普通文本调用 encode_ordinary()
```

例如：

```text
hello <|endoftext|> world
```

会先切分为：

```text
["hello ", "<|endoftext|>", " world"]
```

最终编码为：

```text
encode_ordinary("hello ") + [special_id] + encode_ordinary(" world")
```

---

## 5. 实现过程中遇到的问题

### 5.1 save 时 id 后不能多空格

错误写法：

```python
f.write(f"{special} {idx} \n")
```

这样会在 id 后面多保存一个空格，导致 load 时：

```python
special, idx = line.rsplit(" ", 1)
```

得到：

```text
idx = ""
```

从而 `int(idx)` 报错。

正确写法：

```python
f.write(f"{special} {idx}\n")
```

### 5.2 load 时读取 special token 应使用 `rsplit`

读取 `.model` 中的 special token 行时，应该使用：

```python
special, idx = line.rsplit(" ", 1)
```

因为 id 一定位于最右侧，从右边切更稳妥。

### 5.3 `_build_vocab_()` 使用的属性要提前初始化

`_build_vocab_()` 中需要访问 `self.special_tokens`，所以在基类 `Tokenizer.__init__()` 中，必须先初始化 `self.special_tokens`，再调用 `_build_vocab_()`：

```python
self.merges = {}
self.pattern = ""
self.special_tokens = {}
self.vocab = self._build_vocab_()
```

### 5.4 `super().__init__()` 应放在子类初始化最前面

`RegexTokenizer.__init__()` 中应该先调用：

```python
super().__init__()
```

然后再设置 `pattern`、`compiled_pattern` 和 `inverse_special_tokens`。否则如果把 `super().__init__()` 放在最后，基类初始化可能会覆盖子类中已经设置好的字段。

---

## 6. 总结

当前实现结构如下：

```text
train()：只训练普通 BPE merges
register_special_tokens()：注册特殊 token
encode_ordinary()：普通文本编码
encode()：处理普通文本 + special token
decode()：复用基类 decode
```

核心思想是：

```text
special token 不是 BPE merge 规则，而是 encode 入口处的最高优先级分流规则。
```
