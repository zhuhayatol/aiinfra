# tiny_lm/generate.py

from pathlib import Path

import torch
import tiktoken

from tiny_lm.model.gpt2 import GPT, GPTConfig



# 默认从 log 文件夹里找最新的 checkpoint
LOG_DIR = Path("outputs/tinystories_124m_2gpu/checkpoints")


def find_latest_checkpoint(log_dir=LOG_DIR):
    """
    自动寻找 log 文件夹下最新的模型文件。

    文件名格式假设为：
        model_00500.pt
        model_01000.pt
    """
    ckpts = sorted(log_dir.glob("model_*.pt"))

    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}")

    return ckpts[-1]


def load_checkpoint_model(ckpt_path, device):
    """
    从 checkpoint 加载模型。
    """
    # 先在 CPU 上加载整个 Checkpoint，避免 optimizer 和 RNG 状态
    # 被一并映射到 GPU，造成额外显存占用。
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["model_config"]

    # 兼容 config 保存成 dict 的情况
    if isinstance(config, dict):
        config = GPTConfig(**config)

    model = GPT(config)

    state_dict = checkpoint["model"]

    # 兼容 torch.compile 或 DDP 可能带来的前缀，需要消除
    fixed_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]

        if k.startswith("module."):
            k = k[len("module."):]

        fixed_state_dict[k] = v

    model.load_state_dict(fixed_state_dict)

    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def generate_once(
    model,
    prompt,
    device,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    num_return_sequences=1,
):
    """
    根据一次输入的 prompt 生成文本。
    """
    enc = tiktoken.get_encoding("gpt2")

    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)

    # [T] -> [1, T] -> [num_return_sequences, T]
    x = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    for i in range(num_return_sequences):
        text = enc.decode(y[i].tolist())

        print()
        print(f"[sample {i}]")
        print(text)
        print("-" * 80)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = find_latest_checkpoint(LOG_DIR)

    model, checkpoint = load_checkpoint_model(
        ckpt_path=ckpt_path,
        device=device,
    )

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"step: {checkpoint.get('step')}")
    print(f"val_loss: {checkpoint.get('val_loss')}")
    print(f"device: {device}")
    print("=" * 80)

    print("输入 prompt 后按回车生成。")
    print("输入 q / quit / exit 退出。")
    print("=" * 80)

    while True:
        prompt = input("\nPrompt> ").strip()

        if prompt.lower() in {"q", "quit", "exit"}:
            break

        if not prompt:
            continue

        generate_once(
            model=model,
            prompt=prompt,
            device=device,
            max_new_tokens=100,
            temperature=1.0,
            top_k=50,
            num_return_sequences=1,
        )


if __name__ == "__main__":
    main()