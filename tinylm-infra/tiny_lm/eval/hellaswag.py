"""
HellaSwag 评估脚本。

HellaSwag 是一个多选题式的语言模型评估任务：
    给定一个上下文 ctx
    给出 4 个候选结尾 endings
    模型需要判断哪个 ending 最合理

对于 GPT 这种 causal language model,没有单独的分类头。
因此评估方式不是让模型生成 A/B/C/D,
而是分别计算：

    ctx + ending_0
    ctx + ending_1
    ctx + ending_2
    ctx + ending_3

这四个序列中 ending 部分的平均 loss。
平均 loss 最低的 ending,就是模型认为最可能的答案。

使用前需要先运行：
    python3 prepare_hellaswag.py

该脚本会把 Hugging Face 上的 validation split 保存到：
    data/hellaswag/validation/
"""

import torch
import tiktoken
import torch.nn.functional as F
from datasets import load_from_disk

import json
from pathlib import Path


# 找到下载的本地的文件
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELLASWAG_DIR = PROJECT_ROOT / "data" / "hellaswag" / "validation"

enc = tiktoken.get_encoding("gpt2")

def load_hellaswag():
    """
    从本地磁盘读取已经下载好的 HellaSwag validation 数据。

    注意：
        这里不再使用 load_dataset 联网下载。
        数据应提前通过 prepare_hellaswag.py 下载并 save_to_disk。

    返回：
        Hugging Face Dataset 对象。
    """
    if not HELLASWAG_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find {HELLASWAG_DIR}. "
            f"Run: python3 prepare_hellaswag.py"
        )

    return load_from_disk(str(HELLASWAG_DIR))

def render_example(example, device):
    """
    将一个 HellaSwag 样本转换成模型可以输入的 tokens 和 loss mask。

    一个 example 大致包含：
        example["ctx"]      : 上下文
        example["endings"]  : 4 个候选结尾
        example["label"]    : 正确答案编号，0/1/2/3

    对每个 ending，我们都构造：
        ctx + ending_i

    最终返回：
        tokens: [4, T]
            4 表示 4 个候选答案。
            T 是当前样本中最长候选序列的长度，短的会被 padding 到同样长度。

        mask: [4, T]
            mask 用来标记哪些 token 属于 ending。
            context 部分为 0，ending 部分为 1。
            评估时只统计 ending 部分的 loss。

        label: int
            正确答案编号。
    """
    context = example["ctx"]     # str
    endings = example["endings"] # list[str] shape:4 
    label   = int(example["label"])   # int

    # 对上下文进行编码。
    # ctx 是四个候选答案共享的部分。
    context = enc.encode(context)
    
    tok_rows = []
    mask_rows = []

    for ending in endings:
        # ending 是接在 ctx 后面的
        # 加一个前导空格是常见做法，因为英文中它往往和前文之间有空格。
        ending_tokens = enc.encode(" " + ending)

        # 当前候选答案的完整 token 序列：上下文 + 候选结尾
        tokens = context + ending_tokens

        # mask 只让 ending 部分参与 loss 计算。
        # context 部分只是条件，不参与四个选项之间的打分。
        mask = [0] * len(context) + [1] * len(ending_tokens)

        tok_rows.append(tokens)
        mask_rows.append(mask)
    # 四个候选 ending 长度可能不同，需要 padding 到同一个长度，
    # 才能组成一个 [4, T] 的 batch。
    max_len = max(len(row) for row in tok_rows)

    # 用 0 作为 padding token id。
    # 这里 padding 部分不会参与 loss，因为 mask 对应位置为 0。
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.float32)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row, dtype=torch.long)
        mask[i, :len(mask_row)] = torch.tensor(mask_row, dtype=torch.float32)

    return tokens.to(device), mask.to(device), label

@torch.no_grad()
def get_most_likely_row(
    model,
    tokens,
    mask,
    device_type: str,
    amp_dtype=None,
):
    """
    计算四个候选 ending 的平均 loss,并返回 loss 最低的候选编号。

    参数：
        model:
            原始 GPT 模型 raw_model

        tokens: [4, T]
            四个候选序列的 token ids

        mask: [4, T]
            标记哪些位置属于 ending
            只有 mask=1 的位置会参与 loss 统计。

        device_type:
            设备类型，例如 "cuda" 或 "cpu"。
            注意这里不是 "cuda:0" 或 "cuda:1"。

        amp_dtype:
            混合精度类型，例如 torch.bfloat16 或 torch.float16。
            如果为 None,则不用 autocast。

    返回：
        pred: int
            模型认为最可能的候选 ending 编号。
    """

    # 评估阶段不需要梯度，函数外层已经用 @torch.no_grad() 包住。
    # 如果开启 AMP，则前向时使用 autocast。
    if amp_dtype is not None and device_type == "cuda":
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            logits, _ = model(tokens)
    else:
        logits, _ = model(tokens)

    # logits 形状：
    #     [4, T, vocab_size]
    #
    # GPT 的预测方式是：
    #     位置 t 的 logits 用来预测位置 t+1 的 token。
    #
    # 因此需要错位：
    #     logits[:, :-1, :] 预测 tokens[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_tokens = tokens[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    # 对每一个 token 位置分别计算 cross entropy。
    #
    # cross_entropy 要求输入形状为：
    #     input:  [N, vocab_size]
    #     target: [N]
    #
    # 所以需要先把 [4, T-1, vocab_size] 展平成 [4*(T-1), vocab_size]。
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction="none",
    )

    # 再恢复成 [4, T-1]，对应四个候选答案中每个 token 的 loss。
    losses = losses.view(tokens.size(0), -1)

    # 只保留 ending 部分的 loss。
    # context 部分的 mask 为 0，所以不会计入最终分数。
    masked_losses = losses * shift_mask

    # 每个候选 ending 的平均 loss。
    #
    # 不能直接用总 loss，因为四个 ending 的长度可能不同。
    # 如果用总 loss，长 ending 会天然吃亏。
    avg_loss = masked_losses.sum(dim=1) / shift_mask.sum(dim=1)

    # loss 越低，表示模型认为这个 continuation 越自然。
    pred = avg_loss.argmin().item()

    return pred


@torch.no_grad()
def evaluate_hellaswag(
    model,
    device,
    device_type: str,
    amp_dtype=None,
    max_examples: int | None = 200,
):
    """
    在 HellaSwag validation split 上评估模型。

    参数：
        model:
            建议传 raw_model。
            DDP 训练时不要传 DDP(model)，因为评估不需要梯度同步。

        device:
            具体设备，例如 "cuda:0"、"cuda:1" 或 "cpu"。
            tokens 和 mask 会被移动到这个设备上。

        device_type:
            设备类型，例如 "cuda" 或 "cpu"。
            用于 autocast。

        amp_dtype:
            混合精度类型。
            例如：
                torch.bfloat16
                torch.float16
                None

        max_examples:
            最多评估多少条样本。
            训练过程中建议先设小一点，比如 100 或 200。
            如果设为 None,则评估完整 validation set。

    返回：
        dict:
            {
                "acc": 准确率,
                "num_correct": 答对数量,
                "num_total": 实际评估数量,
                "num_skipped": 因超过 block_size 被跳过的样本数,
            }
    """
    dataset = load_hellaswag()

    model.eval()

    num_correct = 0
    num_total = 0
    num_skipped = 0
    answer, label_list = [], []

    # 你的 GPT 模型有最大上下文长度 block_size。
    # 如果某个 HellaSwag 样本的 ctx + ending 超过 block_size，
    # 这里为了简单直接跳过。
    block_size = getattr(model.config, "block_size", None)

    for example in dataset:
        if max_examples is not None and num_total >= max_examples:
            break

        tokens, mask, label = render_example(example, device)

        if block_size is not None and tokens.size(1) > block_size:
            num_skipped += 1
            continue

        pred = get_most_likely_row(
            model=model,
            tokens=tokens,
            mask=mask,
            device_type=device_type,
            amp_dtype=amp_dtype,
        )

        answer.append(pred)
        label_list.append(label)

        num_correct += int(pred == label)
        num_total += 1


    acc = num_correct / max(1, num_total)

    return {
        "answer":answer,
        "label_list":label_list,
        "acc": acc,
        "num_correct": num_correct,
        "num_total": num_total,
        "num_skipped": num_skipped,
    }