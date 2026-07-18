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

## 5. GPT4Tokenizer

`GPT4Tokenizer` 位于 `gpt4.py`，它不是从零训练出来的 tokenizer，而是一个 pretrained tokenizer wrapper。它的目标是复现 tiktoken 中的 `cl100k_base`，使自己的 tokenizer 编码结果和官方 GPT-4 tokenizer 保持一致。

也就是说，目标是：

```python
GPT4Tokenizer().encode(text) == tiktoken.get_encoding("cl100k_base").encode(text)
```

`GPT4Tokenizer` 继承自 `RegexTokenizer`，但它和普通 `RegexTokenizer` 有三个核心区别：

```text
1. 不通过 train() 训练 merges，而是从 tiktoken 的 cl100k_base 中恢复 merges
2. 使用 GPT-4 自己的 regex split pattern 和 special tokens
3. 需要处理 GPT-4 tokenizer 中特殊的 byte_shuffle
```

---

### 5.1 GPT-4 tokenizer 为什么不能直接用普通 RegexTokenizer

普通 `RegexTokenizer` 默认认为：

```text
原始 byte 值 == 初始 token id
```

例如：

```text
b"a" -> 97
b"b" -> 98
```

但是 GPT-4 的 `cl100k_base` 不是这样。它的单 byte token id 被重新排列过。例如：

```text
b"a" -> 某个 token id
b"b" -> 某个 token id
```

这个 token id 不一定等于原始 byte 值。

因此，如果直接把 UTF-8 bytes 送进普通 BPE 流程，会导致初始 token id 不一致，后续 merge 结果也无法和 tiktoken 对齐。

所以 `GPT4Tokenizer` 必须额外处理：

```text
原始 byte -> GPT-4 byte token id
```

这一步就是 `byte_shuffle`。

---

### 5.2 mergeable_ranks

tiktoken 中的核心数据结构是：

```python
mergeable_ranks: dict[bytes, int]
```

它表示：

```text
bytes token -> token id / rank
```

例如：

```text
b"a"      -> 某个 id
b"hello"  -> 某个 id
b" world" -> 某个 id
```

这里要注意：`mergeable_ranks` 里面既包含单 byte token，也包含 BPE 合并出来的多 byte token。

可以分成两类：

```text
长度为 1 的 bytes：
    基础 byte token
    不需要写入 merges
    但需要用于 byte_shuffle

长度大于 1 的 bytes：
    BPE merge 产生的 token
    需要恢复成 merges
```

普通 tokenizer 中的 merges 格式是：

```python
(pair_token_id_1, pair_token_id_2) -> new_token_id
```

而 tiktoken 给的是：

```python
bytes_token -> rank
```

所以 `GPT4Tokenizer` 需要从 `mergeable_ranks` 中恢复出自己的 `self.merges`。

---

### 5.3 recover_merges

`recover_merges()` 的作用是把 tiktoken 的：

```python
bytes -> rank
```

恢复成自己 tokenizer 使用的：

```python
(int, int) -> int
```

也就是：

```text
两个已有 token id 合并成一个新 token id
```

对于长度为 1 的 bytes token，例如：

```python
b"a"
```

它本身就是最底层的 byte token，不是由两个 token merge 出来的，所以在 `recover_merges()` 中要跳过。

对于长度大于 1 的 bytes token，例如：

```python
b"hello"
```

它可能是由更小的两个 token 合并得到的，例如：

```text
b"he" + b"llo" -> b"hello"
```

恢复 merges 时，需要找到这两个直接子 token，并得到：

```python
(id_of_b"he", id_of_b"llo") -> id_of_b"hello"
```

这就是 `recover_merges()` 的核心目的。

---

### 5.4 bpe 辅助函数

`bpe()` 是 `recover_merges()` 中使用的辅助函数。

它的作用不是普通 encode，而是为了恢复某个 bytes token 的直接来源。

例如现在要恢复：

```python
b"hello"
```

那么 `bpe()` 会从单 byte 开始：

```text
b"h", b"e", b"l", b"l", b"o"
```

然后根据 `mergeable_ranks` 中已有的 rank，不断执行 BPE 合并。

但是恢复某个 token 时，不能使用比它自己更晚的 merge。比如当前 token 的 rank 是 5000，那么只能使用 rank 小于 5000 的 merge 规则。

因此 `bpe()` 中需要传入：

```python
max_rank
```

含义是：

```text
只允许使用 rank < max_rank 的合并
```

最后 `bpe()` 应该把当前 token 恢复成两个直接子 token。

例如：

```python
bpe(mergeable_ranks, b"hello", max_rank=rank_of_hello)
```

可能得到：

```python
[b"he", b"llo"]
```

随后就可以查：

```python
mergeable_ranks[b"he"]
mergeable_ranks[b"llo"]
```

得到两个子 token 的 id，并写入 merges。

---

### 5.5 byte_shuffle

GPT-4 tokenizer 中最容易出错的一点是 byte shuffle。

普通 byte-level BPE 中，初始 byte token 通常是：

```text
byte 0   -> token id 0
byte 1   -> token id 1
...
byte 97  -> token id 97
```

但是 `cl100k_base` 中，单 byte token 的 id 被重新排列过。因此需要构造：

```python
byte_shuffle = {
    原始 byte 值 -> GPT-4 中该 byte 对应的 token id
}
```

具体来自：

```python
byte_shuffle[i] = mergeable_ranks[bytes([i])]
```

同时还需要构造反向映射：

```python
inverse_byte_shuffle = {
    GPT-4 byte token id -> 原始 byte 值
}
```

用于 decode。

整体关系是：

```text
encode:
    原始 UTF-8 byte
    -> byte_shuffle
    -> GPT-4 初始 token id
    -> BPE merge

decode:
    token id
    -> vocab 中的 shuffled bytes
    -> inverse_byte_shuffle
    -> 原始 UTF-8 bytes
    -> 字符串
```

---

### 5.6 GPT4Tokenizer 的初始化流程

`GPT4Tokenizer.__init__()` 主要做以下几件事：

```text
1. 使用 GPT4_SPLIT_PATTERN 初始化 RegexTokenizer
2. 通过 tiktoken.get_encoding("cl100k_base") 加载官方 tokenizer
3. 读取 enc._mergeable_ranks
4. 通过 recover_merges() 恢复 self.merges
5. 根据 self.merges 构建 self.vocab
6. 构建 byte_shuffle 和 inverse_byte_shuffle
7. 注册 GPT-4 special tokens
```

其中 GPT-4 special tokens 包括：

```python
{
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}
```

因为 GPT4Tokenizer 是预训练 tokenizer，所以它不应该再调用 `train()`。

因此：

```python
train()
save()
load()
```

都可以直接设置为 `NotImplementedError`。

---

### 5.7 GPT4Tokenizer 的 encode_chunk

普通 `RegexTokenizer.encode_ordinary()` 的流程是：

```text
文本
-> regex 分块
-> 每个 chunk 转成 UTF-8 bytes
-> self.encode_chunk(chunk_bytes)
```

由于这里调用的是 `self.encode_chunk()`，所以当对象是 `GPT4Tokenizer` 时，会自动调用 GPT4Tokenizer 自己的 `encode_chunk()`。

GPT4Tokenizer 的 `encode_chunk()` 多做一步 byte shuffle：

```text
原始 UTF-8 bytes
-> byte_shuffle
-> 父类 encode_chunk()
-> BPE merge
```

也就是说：

```python
chunk = bytes(self.byte_shuffle[ch] for ch in chunk)
return super().encode_chunk(chunk)
```

这里的 `chunk` 是 bytes，遍历 bytes 得到的是 int，所以可以直接用 `self.byte_shuffle[ch]` 查表。

---

### 5.8 GPT4Tokenizer 的 decode

普通 `RegexTokenizer.decode()` 可以直接复用基类，因为 vocab 中保存的就是原始 bytes。

但是 `GPT4Tokenizer` 不能直接复用基类 decode。原因是 GPT-4 的普通 token 对应的是 shuffled byte 空间，需要在 decode 时还原。

因此普通 token 的 decode 流程是：

```text
token id
-> self.vocab[id]
-> 得到 shuffled bytes
-> inverse_byte_shuffle
-> 原始 UTF-8 bytes
```

同时，special token 要单独处理。

例如：

```text
100257 -> "<|endoftext|>"
```

special token 对应的 bytes 本身就是普通 UTF-8 字符串，不应该再经过 `inverse_byte_shuffle`。

所以 GPT4Tokenizer 的 decode 应该区分两种情况：

```text
如果 idx 是 special token id：
    直接还原成 special token 字符串

如果 idx 是普通 vocab id：
    查 vocab 得到 shuffled bytes
    对每个 byte 执行 inverse_byte_shuffle
    得到原始 bytes
```

也就是说，GPT4Tokenizer 的 decode 逻辑比普通 tokenizer 多了一层 byte unshuffle。

---

### 5.9 GPT4Tokenizer 的测试

GPT4Tokenizer 的核心测试不是只验证 `decode(encode(text)) == text`，而是要和 tiktoken 对齐。

主要测试包括：

```text
1. 普通文本 encode 结果和 tiktoken 一致
2. 中英文、韩文、emoji、标点、数字混合文本能正确 encode/decode
3. special tokens 在 allowed_special="all" 时编码结果和 tiktoken 一致
4. special tokens 能 decode 回原字符串
5. 默认 allowed_special="none_raise" 时，遇到 special token 应该报错
6. allowed_special="none" 时，special token 应该被当作普通文本处理
```

例如：

```python
enc = tiktoken.get_encoding("cl100k_base")
tokenizer = GPT4Tokenizer()

assert tokenizer.encode(text) == enc.encode(text)
assert tokenizer.decode(tokenizer.encode(text)) == text
```

对于 special token：

```python
ids = tokenizer.encode("<|endoftext|>", allowed_special="all")

assert ids == [100257]
assert tokenizer.decode(ids) == "<|endoftext|>"
```

---

### 5.10 GPT4Tokenizer 总结

`GPT4Tokenizer` 可以理解为：

```text
RegexTokenizer + tiktoken 的 cl100k_base merges + byte_shuffle + GPT-4 special tokens
```

它不是重新训练一个 tokenizer，而是对官方 GPT-4 tokenizer 的复现。

最核心的三个点是：

```text
1. 从 mergeable_ranks 恢复 merges
2. 用 byte_shuffle 处理 GPT-4 单 byte token id 的排列
3. decode 时对普通 token 做 inverse_byte_shuffle，对 special token 直接还原
```

## 6. 实现过程中遇到的问题

### 6.1 save 时 id 后不能多空格

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

### 6.2 load 时读取 special token 应使用 `rsplit`

读取 `.model` 中的 special token 行时，应该使用：

```python
special, idx = line.rsplit(" ", 1)
```

因为 id 一定位于最右侧，从右边切更稳妥。

### 6.3 `_build_vocab_()` 使用的属性要提前初始化

`_build_vocab_()` 中需要访问 `self.special_tokens`，所以在基类 `Tokenizer.__init__()` 中，必须先初始化 `self.special_tokens`，再调用 `_build_vocab_()`：

```python
self.merges = {}
self.pattern = ""
self.special_tokens = {}
self.vocab = self._build_vocab_()
```

### 6.4 `super().__init__()` 应放在子类初始化最前面

`RegexTokenizer.__init__()` 中应该先调用：

```python
super().__init__()
```

然后再设置 `pattern`、`compiled_pattern` 和 `inverse_special_tokens`。否则如果把 `super().__init__()` 放在最后，基类初始化可能会覆盖子类中已经设置好的字段。

---

## 7. 总结

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

---



***Tokenizer 阶段已完成：实现 BasicTokenizer、RegexTokenizer、Special Tokens、GPT4Tokenizer，并通过 encode/decode 与 tiktoken 对齐测试。下一阶段进入 Dataset/get_batch，将文本 token ids 转换为语言模型训练样本。***