# tests/unit/config/test_config.py

from pathlib import Path

import pytest

from tiny_lm.config import (
    load_config,
    validate_config,
)


VALID_CONFIG = """
experiment:
  name: unit_test
  seed: 42
  output_dir: outputs/unit_test

model:
  block_size: 32
  vocab_size: 100
  n_layer: 2
  n_head: 4
  n_embd: 32

data:
  data_dir: data/test
  batch_size: 2
  sequence_length: 16

training:
  max_steps: 10
  total_batch_size: 128
  max_learning_rate: 0.0006
  min_learning_rate: 0.00006
  warmup_steps: 2
  weight_decay: 0.1
  gradient_clip: 1.0
  dtype: float32
  use_compile: false
  use_fused_adamw: false

evaluation:
  eval_interval: 5
  eval_steps: 2
  run_hellaswag: false

checkpoint:
  save_interval: 5
  save_optimizer: true
  save_rng_state: true
  save_dataloader_state: true
  resume_from: null
"""


def write_config(
    path: Path,
    content: str = VALID_CONFIG,
) -> None:
    """向临时路径写入 YAML 配置。"""

    path.write_text(
        content,
        encoding="utf-8",
    )


def test_load_valid_config(
    tmp_path: Path,
) -> None:
    """合法 YAML 应被转换为带类型的 Config 对象。"""

    config_path = tmp_path / "train.yaml"
    write_config(config_path)

    config = load_config(config_path)

    assert config.experiment.name == "unit_test"
    assert config.model.n_layer == 2
    assert config.model.n_head == 4
    assert config.data.batch_size == 2
    assert config.training.max_steps == 10
    assert config.checkpoint.resume_from is None


def test_validate_valid_config(
    tmp_path: Path,
) -> None:
    """合法配置在单卡和双卡情况下都应通过检查。"""

    config_path = tmp_path / "train.yaml"
    write_config(config_path)

    config = load_config(config_path)

    validate_config(
        config,
        world_size=1,
    )

    validate_config(
        config,
        world_size=2,
    )


def test_load_config_rejects_missing_group(
    tmp_path: Path,
) -> None:
    """缺少 checkpoint 配置组时应明确报错。"""

    config_path = tmp_path / "invalid.yaml"

    invalid_config = VALID_CONFIG.replace(
        """
checkpoint:
  save_interval: 5
  save_optimizer: true
  save_rng_state: true
  save_dataloader_state: true
  resume_from: null
""",
        "",
    )

    write_config(
        config_path,
        invalid_config,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint",
    ):
        load_config(config_path)


def test_validation_rejects_invalid_attention_dimensions(
    tmp_path: Path,
) -> None:
    """隐藏维度不能被注意力头数整除时应报错。"""

    config_path = tmp_path / "train.yaml"
    write_config(config_path)

    config = load_config(config_path)

    config.model.n_embd = 30
    config.model.n_head = 4

    with pytest.raises(
        ValueError,
        match="n_embd",
    ):
        validate_config(
            config,
            world_size=1,
        )


def test_validation_rejects_invalid_global_batch(
    tmp_path: Path,
) -> None:
    """全局 token batch 无法整除 micro batch 时应报错。"""

    config_path = tmp_path / "train.yaml"
    write_config(config_path)

    config = load_config(config_path)

    config.training.total_batch_size = 130

    with pytest.raises(
        ValueError,
        match="total_batch_size",
    ):
        validate_config(
            config,
            world_size=1,
        )