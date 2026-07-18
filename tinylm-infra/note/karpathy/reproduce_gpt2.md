# hugging face
首先做的是参考huggingface中的gpt2来配置自己的gpt，包括模型结构、层数配置、模块命名以及参数名称。这样做的核心目的，是让自己实现的 GPT 模型能够与 HuggingFace GPT-2 的 `state_dict` 对齐，从而加载官方预训练权重，并验证模型结构是否正确。

## gptconfig
`GPTConfig` 用于集中保存 GPT-2 模型的相关超参数，方便模型构造、复用和切换不同规模的 GPT-2。

## gpt类
里面主要是关于gpt2的结构，forward，from_pretrained的代码。
### 结构
与transformer论文中的结构相似但是有很多不同，比如：

- GPT-2 是 decoder-only 结构，只使用 Transformer Decoder 的思想，不包含 Encoder 部分。因此，GPT-2 的任务是根据前面的 token 预测下一个 token。
- 整体结构包括：
    ``` text
    token embedding
    position embedding
    多个 Transformer Block
    final LayerNorm
    lm_head
    ```
    其中每个 Block 包含：
    ``` text
    LayerNorm
    Causal Self-Attention
    残差连接
    LayerNorm
    MLP / FeedForward
    残差连接
    ```
- 需要注意的是，GPT-2 使用的是 Pre-LayerNorm 结构，也就是在进入 attention 和 MLP 之前先做 LayerNorm：

    ``` python
    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))
    ```

- 最后，模型会经过一个 final LayerNorm，然后通过 lm_head 映射到词表维度，得到每个位置对下一个 token 的预测 logits。

### forward
内容和上一个transformer中的区别不大。唯一需要注意的是每个tensor的shape以及所使用的函数对tensor的shape的要求

### generate
这里添加了一个topk，具体做法是筛选出每个batch中下一个token概率的最大k个，再将其余小于第k个概率的所有内容变为-inf，然后才会进行softmax。

需要注意：更标准的做法是在 logits 层面 进行筛选，而不是先 softmax 得到概率再筛选。

### form_pretrained
这个主要是导入参数以及根据不同需求选择不同配置

需要注意的是，在huggingface中的版本和我们使用的版本有俩个区别：
1. 有一些参数需要忽略
    HuggingFace GPT-2 的 state_dict 中包含一些 attention mask 相关的 buffer，例如：

    ``` text
    attn.bias
    attn.masked_bias
    ```
    这些主要用于旧版本实现中的 causal mask。

    而自己实现的 attention 中使用了：

    ``` python
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ```

    因此不需要这些 mask buffer。在对齐参数时，需要将它们过滤掉，否则两边的 key 数量会不一致。
2. 有一些参数需要转置
    HuggingFace GPT-2 中有些线性层沿用了 OpenAI GPT-2 原始实现中的 Conv1D 风格，其权重存储方向和 PyTorch 标准的 nn.Linear 不同。因此复制前需要转置。

3. 以及在最后的转置的过程中注意使用copy而不是直接赋值。

## 阶段总结
本阶段完成了 GPT-2 结构复现和 HuggingFace 权重加载。通过 from_pretrained()，可以将 HuggingFace GPT-2 的预训练参数迁移到自己实现的 GPT 模型中。

这一阶段的核心价值是：用官方 GPT-2 权重验证自己实现的模型结构是否正确。如果模型能够成功加载权重，并使用 generate() 生成较自然的文本，说明以下部分基本正确：


