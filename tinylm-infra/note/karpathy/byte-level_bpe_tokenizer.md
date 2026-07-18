## byte-level BPE tokenizer

### 1. 训练过程

训练本身是为了得到一组 BPE 合并规则，也就是 `merge_dict`。它记录哪些 byte/token 组合应该被合并成更大的 token。

训练流程如下：

* 将输入文本通过 UTF-8 编码转换为 bytes，再把 bytes 转成整数列表。由于 byte 的取值范围是 `0~255`，所以 byte-level BPE 的基础词表大小是 256。
* 使用函数 `get_stats` 统计当前 token 序列中所有相邻 pair 的出现频次。
* 选择出现频次最高的 pair，将它合并成一个新的 token id。
* 新 token id 从 256 开始编号。每做一次新的 merge，就产生一个新的 token id。
* 如果一共做 `num_merges` 次合并，那么最终 `vocab_size = 256 + num_merges`。
* 每次合并时，都要记录这条合并规则，形成 `merge_dict`：

```python
merge_dict[pair] = idx
```

其中 `pair` 是被合并的两个 token，`idx` 是合并后生成的新 token id。

因此，BPE tokenizer 的训练过程本质上就是：从训练文本中不断找到最高频 pair，并生成一组固定的合并规则。

---

### 2. 推理过程

tokenizer 的推理过程就是使用已经训练好的 `merge_dict` 和 `vocab`，对新文本进行编码或解码。

---

### encode

encode 的作用是：

```text
文本 -> token ids
```

流程如下：

* 将新的文本和训练过程一样，先转成 UTF-8 bytes，再转成整数列表。
* 统计当前 token 序列中存在的相邻 pair。
* 注意，encode 阶段不是重新训练 tokenizer，因此不会再根据当前文本的频次选择最高频 pair。
* encode 阶段会从当前存在的 pair 中，选择已经出现在 `merge_dict` 中、并且 merge rank 最高的 pair 进行合并。
* 由于训练时新 token id 是按顺序递增生成的，所以 token id 越小，说明这条 merge 规则越早出现，优先级越高。
* 每次合并后，需要重新统计当前序列中的 pair。因为新的 token 生成后，可能会产生新的 pair。
* 不断重复这个过程，直到当前 token 序列中不存在任何可以根据 `merge_dict` 合并的 pair。

---

### decode

decode 的作用是：

```text
token ids -> 文本
```

decode 需要先根据 `merge_dict` 构造出 `vocab`：

```python
vocab = {i: bytes([i]) for i in range(256)}

for (a1, a2), idx in merge_dict.items():
    vocab[idx] = vocab[a1] + vocab[a2]
```

其中：

```text
merge_dict: pair -> token_id
vocab: token_id -> bytes
```

构造 `vocab` 时，必须先初始化 `0~255` 这 256 个基础 byte token，然后再根据训练得到的 merge 规则，依次构造 256 及以上的新 token。

解码时，将 token id 序列中的每个 id 通过 `vocab` 转成对应的 bytes，然后用 `b"".join()` 拼接成完整 bytes，最后通过 UTF-8 decode 得到原始文本：

```python
tokens = b"".join(vocab[idx] for idx in ids)
text = tokens.decode("utf-8")
```

因此，只要 token ids 是由当前 tokenizer 正确 encode 得到的，就应该满足：

```python
decode(encode(text)) == text
```
