import os

import pytest
import torch
import tiktoken

from tiny_lm.model.gpt2 import GPT


@pytest.mark.integration
def test_from_pretrained():
    local_model_path = os.environ.get("GPT2_LOCAL_PATH")

    if local_model_path:
        model = GPT.from_pretrained(
            model_type="gpt2",
            model_path=local_model_path,
        )
        model_source = f"local: {local_model_path}"
    else:
        model = GPT.from_pretrained(
            model_type="gpt2",
            model_path=None,
        )
        model_source = "Hugging Face cache or Hub: gpt2"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.eval()
    model.to(device)

    tokenizer = tiktoken.get_encoding("gpt2")

    text = "Hello, I am from Peking University."
    input_ids = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()

    generator.manual_seed(42)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=2,
        temperature=1.0,
        top_k=50,
        generator=generator,
    )

    print(f"Loaded model from {model_source}")

    assert output_ids.shape == (
        1,
        input_ids.size(1) + 2,
    )