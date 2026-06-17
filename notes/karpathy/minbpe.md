## Tokenizer类
（代码在base.py中）

base.py 中主要包含两部分：
1. 工具函数：get_stats、merge
2. Tokenizer 基类：decode、encode、train、save、load、save_vocab、_build_vocab_


#### decode

decode 是将输入的 token ids 按顺序逐个查 vocab，得到每个 token id 对应的 bytes，然后用 b"".join() 拼接成完整 bytes，最后通过 UTF-8 decode 还原成文本。

``` python
ids = [260, 32, 97]
tokens = b"".join(vocab[idx] for idx in ids)
``` 

#### \_build_vocab_

既然提到了vocab就先讨论它，我们在train的过程会得到一个merges的表，这个表表示不同的pair所映射的数字，它参与了vocab的组成

- 首先将0-255所有数字的bytes与数字本身作为一组value和key放入vocab字典中。
- 随后遍历merges，取出对应的key和value：比如(201, 4), 258,于是vocab的key为258的`value = vocab[201] + vocab[4],`因为256之前的vocab的value都是bytes类型，所以相加后的结果也是bytes。
- 遍历结束，vocab构建完成

#### encode
在基类中未编写

#### train
在基类中未编写

#### get_stats

目的是统计当前文本中的pair的频次

``` python 
def get_stats(text:list) -> dict[tuple, int]:
    stats = {}
    for i,j in zip(text[:-1], text[1:]):
        stats[(i,j)] = stats.get((i,j), 0) + 1
    return stats
```
#### merge(tokens, pair, count):

merge 的作用是扫描 tokens 序列，将所有非重叠出现的指定 pair 替换为新的 token id。它既用于 train 阶段，也用于 encode 阶段。

#### save / save_vocab

这个是讲训练之后的merges保存为单独的文件，其中包括 version， pattern， special_tokens, merges。

将训练结果保存为.model文件，比如
``` text
minbpe v1

0
115 101
44 32
257 118
100 101
259 102
260 32

```
其中：
-  minbpe v1是version
- 中间空的一行是还没编写的pattern
- 0 表示 special_token的数量，因为还没有设置，所以没有对应的行
- 接下来的数字 + 空格 + 数字的组合就是merge中的pair，之所以不写count，是因为它本身就是按顺序从256开始依次向下增加
- .model 文件中保存的是 merges 的顺序，而不是 vocab 的完整内容。load 时只要按顺序读取 pair，就可以从 256 开始恢复每条 merge 对应的 token id。

save_vocab 是把 vocab 保存成人类可读的可视化文件。对于 merge 生成的新 token，会展示它是由哪两个子 token 拼接得到的。
比如：

``` text
[s][e] -> [se] 256
[,][ ] -> [, ] 257
[, ][v] -> [, v] 258
[d][e] -> [de] 259
[de][f] -> [def] 260
[def][ ] -> [def ] 261
``` 

#### load
读取model文件中的数据并且挨个解析，注意得到merges之后也需要构建vocab。

## BPETokenizer
（代码在basic.py中）

BPETokenizer 继承 Tokenizer，主要补全 Basic BPE 的 train 和 encode。decode、save、load、save_vocab 等通用逻辑直接复用基类。

train和encode中的代码与[byte-level_bpe_tokenizer.md]中的内容几乎没有区别。故不多赘述。


## RegexTokenizer
(代码在regex.py中)

主要包括：`train`, `encode`, `encode_chunk`

RegexTokenizer 与 BasicTokenizer 的区别是：BasicTokenizer 直接对整段文本做 BPE，而 RegexTokenizer 会先用正则表达式把文本切成多个 chunk，再在每个 chunk 内部单独做 BPE。

训练时，RegexTokenizer 会对每个 chunk 单独统计 pair，然后把所有 chunk 内部的 pair 频次汇总，选择全局最高频 pair 作为新的 merge 规则。合并时也必须对每个 chunk 分别 merge，不能把所有 chunk 拼成一条长序列，否则会产生跨 chunk 边界的 pair。

编码时，RegexTokenizer 先对新文本做 regex 分块，然后对每个 chunk 按已有 merges 的优先级执行 BPE，最后再把所有 chunk 的 token ids 拼接成最终结果。