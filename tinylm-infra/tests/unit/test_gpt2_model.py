from tiny_lm.model.gpt2 import GPT, GPTConfig
import torch
import tiktoken
import pytest

def test_gpt_forward_shape():
    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)

    idx = torch.randint(0, config.vocab_size, (4, 8))
    logits, loss = model(idx)

    assert logits.shape == (4, 8, config.vocab_size)
    assert loss is None

def test_gpt_forward_loss():
    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)

    idx = torch.randint(0, config.vocab_size, (4, 8))
    targets = torch.randint(0, config.vocab_size, (4, 8))

    logits, loss = model(idx, targets)

    assert logits.shape == (4, 8, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0

def test_generate_shape():
    config = GPTConfig(
        block_size=8,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)

    idx = torch.randint(0, config.vocab_size, (2, 4))
    out = model.generate(idx, max_new_tokens=6, top_k=10)

    assert out.shape == (2, 10)

def test_generate_over_block_size():
    config = GPTConfig(
        block_size=8,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)

    idx = torch.randint(0, config.vocab_size, (2, 12))
    out = model.generate(idx, max_new_tokens=3, top_k=10)

    assert out.shape == (2, 15)

# 测试给模型的输入大于blocksize的时候是否会报错
def test_gpt_rejects_sequence_over_block_size():
    config = GPTConfig(
        block_size=8,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)

    idx = torch.randint(0, config.vocab_size, (2, 9))

    with pytest.raises(AssertionError):
        model(idx)

# 测试 GPT 模型在 eval() 模式下是否具有确定性（即相同输入是否产生相同输出）
def test_gpt_forward_is_deterministic_in_eval_mode():
    torch.manual_seed(42)

    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits_1, _ = model(idx)
        logits_2, _ = model(idx)

    torch.testing.assert_close(logits_1, logits_2)