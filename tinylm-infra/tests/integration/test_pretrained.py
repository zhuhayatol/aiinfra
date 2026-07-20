import os

import pytest
import torch
import tiktoken

from tiny_lm.model.gpt2 import GPT


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="设置 RUN_INTEGRATION_TESTS=1 后运行外部模型集成测试",
)
def test_from_pretrained():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor(
        enc.encode("Hello, I am"),
        dtype=torch.long,
        device=device,
    )[None, :]

    output = model.generate(
        idx,
        max_new_tokens=2,
        temperature=1.0,
        top_k=20,
    )

    assert output.shape == (1, idx.size(1) + 2)