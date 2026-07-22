# tests/unit/train/test_checkpoint.py

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from tiny_lm.train.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


class DummyTrainLoader:
    """
    用于测试 Checkpoint 的简化 DataLoader。

    这里只验证 Checkpoint 能否正确保存和恢复 DataLoader 状态，
    不依赖真实 TinyStories 数据文件。
    """

    def __init__(
        self,
        current_shard: int = 0,
        current_position: int = 0,
    ) -> None:
        # 当前正在读取的数据分片编号。
        self.current_shard = current_shard

        # 当前分片中的 token 读取位置。
        self.current_position = current_position

    def state_dict(self) -> dict[str, int]:
        """返回需要写入 Checkpoint 的 DataLoader 状态。"""

        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state_dict(
        self,
        state: dict[str, int],
    ) -> None:
        """从 Checkpoint 恢复 DataLoader 读取位置。"""

        self.current_shard = state["current_shard"]
        self.current_position = state["current_position"]


def build_model(
    device: torch.device,
) -> nn.Module:
    """
    创建用于 Checkpoint 测试的小型模型。

    单元测试不需要使用完整 GPT-2，否则测试速度慢，
    并且容易把模型问题和 Checkpoint 问题混在一起。
    """

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.GELU(),
        nn.Linear(8, 2),
    )

    return model.to(device)


def run_one_optimization_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    执行一次训练步骤。

    这样 AdamW 会产生 exp_avg、exp_avg_sq 等内部状态，
    测试才能真正验证优化器状态是否被恢复。
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    inputs = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        device=device,
    )

    outputs = model(inputs)
    loss = outputs.square().mean()

    loss.backward()
    optimizer.step()


def assert_nested_equal(
    actual: Any,
    expected: Any,
) -> None:
    """
    递归比较包含字典、列表和 Tensor 的嵌套状态。

    optimizer.state_dict() 是多层嵌套结构，
    不能只使用普通的 == 进行比较。
    """

    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)

        torch.testing.assert_close(
            actual.cpu(),
            expected.cpu(),
            rtol=0,
            atol=0,
        )
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()

        for key in expected:
            assert_nested_equal(
                actual[key],
                expected[key],
            )

        return

    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)

        for actual_item, expected_item in zip(
            actual,
            expected,
        ):
            assert_nested_equal(
                actual_item,
                expected_item,
            )

        return

    assert actual == expected


def test_checkpoint_restores_training_state(
    tmp_path: Path,
) -> None:
    """
    验证模型、优化器、step 和 DataLoader 状态均可恢复。
    """

    device = torch.device("cpu")

    # 固定随机种子，保证测试结果稳定。
    torch.manual_seed(42)

    original_model = build_model(device)
    original_optimizer = torch.optim.AdamW(
        original_model.parameters(),
        lr=1e-3,
    )

    original_loader = DummyTrainLoader(
        current_shard=3,
        current_position=2048,
    )

    # 先执行一次优化，使优化器内部状态不再为空。
    run_one_optimization_step(
        original_model,
        original_optimizer,
        device,
    )

    expected_model_state = copy.deepcopy(
        original_model.state_dict()
    )

    expected_optimizer_state = copy.deepcopy(
        original_optimizer.state_dict()
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=original_model,
        optimizer=original_optimizer,
        step=7,
        train_loader=original_loader,
    )

    assert checkpoint_path.is_file()

    # 创建全新的对象，确保恢复结果确实来自 Checkpoint，
    # 而不是原对象本身仍保留着状态。
    restored_model = build_model(device)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=1e-3,
    )

    restored_loader = DummyTrainLoader(
        current_shard=99,
        current_position=99999,
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        train_loader=restored_loader,
    )

    # Checkpoint 中的 step 表示已经完成的步骤。
    assert checkpoint["step"] == 7

    assert_nested_equal(
        restored_model.state_dict(),
        expected_model_state,
    )

    assert_nested_equal(
        restored_optimizer.state_dict(),
        expected_optimizer_state,
    )

    assert restored_loader.current_shard == 3
    assert restored_loader.current_position == 2048


def test_checkpoint_restores_cpu_rng_state(
    tmp_path: Path,
) -> None:
    """
    验证 CPU 随机数生成器能够恢复到保存时的状态。

    恢复后产生的随机数，应当与保存后原本要产生的随机数完全一致。
    """

    device = torch.device("cpu")

    model = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint_path = tmp_path / "rng_checkpoint.pt"

    # 在保存前重新设置种子，避免模型初始化消耗随机数
    # 干扰测试逻辑。
    torch.manual_seed(2026)

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=3,
    )

    # 这是从 Checkpoint 保存的 RNG 状态继续生成时，
    # 理论上应该得到的随机数。
    expected_random_values = torch.rand(16)

    # 故意推进当前 RNG 状态。
    _ = torch.rand(128)

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
    )

    # 恢复 RNG 后，应重新得到 expected_random_values。
    actual_random_values = torch.rand(16)

    torch.testing.assert_close(
        actual_random_values,
        expected_random_values,
        rtol=0,
        atol=0,
    )


def test_load_checkpoint_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    """Checkpoint 文件不存在时，应给出明确异常。"""

    device = torch.device("cpu")
    model = build_model(device)

    missing_path = tmp_path / "missing.pt"

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint",
    ):
        load_checkpoint(
            path=missing_path,
            model=model,
            optimizer=None,

        )


def test_load_checkpoint_rejects_missing_model_state(
    tmp_path: Path,
) -> None:
    """Checkpoint 缺少模型参数时，不能静默继续运行。"""

    device = torch.device("cpu")
    model = build_model(device)

    invalid_path = tmp_path / "invalid.pt"

    torch.save(
        {
            "step": 5,
            "optimizer": {},
        },
        invalid_path,
    )

    with pytest.raises(
        KeyError,
        match="model",
    ):
        load_checkpoint(
            path=invalid_path,
            model=model,
            optimizer=None,

        )


def test_load_checkpoint_requires_optimizer_state_when_resuming(
    tmp_path: Path,
) -> None:
    """
    当调用方要求恢复优化器时，
    Checkpoint 中缺少 optimizer 应直接报错。
    """

    device = torch.device("cpu")
    model = build_model(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    invalid_path = tmp_path / "no_optimizer.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "step": 5,
        },
        invalid_path,
    )

    with pytest.raises(
        KeyError,
        match="optimizer",
    ):
        load_checkpoint(
            path=invalid_path,
            model=model,
            optimizer=optimizer,
        )