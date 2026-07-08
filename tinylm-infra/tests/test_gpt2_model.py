from tiny_lm.model.gpt2 import GPT, GPTConfig
import torch
import tiktoken

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

def test_from_pretrained():

    enc = tiktoken.get_encoding("gpt2")

    model = GPT.from_pretrained("gpt2", model_path="../tiny_lm/model/gpt2_huggingface")
    model.eval()
    model.to("cuda")

    text = "hello, im from Peking university"
    idx = torch.tensor(enc.encode(text), dtype=torch.long)[None, :].to("cuda")

    out = model.generate(
        idx,
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
    )

    assert out.shape == (1, 50 + len(idx[0, :]))