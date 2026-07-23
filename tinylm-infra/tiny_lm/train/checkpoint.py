# tiny_lm/train/checkpoint.py

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def unwrap_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    移除 DDP 和 torch.compile 的外层包装，返回原始模型。

    DDP 通常将原始模型保存在 model.module 中；
    torch.compile 通常将原始模型保存在 model._orig_mod 中。
    """

    if hasattr(model, "module"):
        model = model.module

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model


def serialize_config(
    config: Any,
) -> dict[str, Any]:
    """
    将配置对象转换为可写入 Checkpoint 的普通字典。

    支持：
    1. dataclass 配置；
    2. 普通字典。

    配置为 None 时，不应调用该函数。
    """

    if is_dataclass(config):
        return asdict(config)

    if isinstance(config, dict):
        return config

    raise TypeError(
        "config 必须是 dataclass 或 dict，"
        f"实际类型为：{type(config).__name__}"
    )


def infer_model_device(
    model: torch.nn.Module,
) -> torch.device:
    """
    根据模型参数推断模型所在设备。

    无参数模型会默认返回 CPU。
    """

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_value_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    """
    递归地将嵌套结构中的 Tensor 移动到指定设备。

    optimizer.state 中可能包含 Tensor、字典、列表或元组，
    因此不能只处理第一层。
    """

    if isinstance(value, torch.Tensor):
        return value.to(device)

    if isinstance(value, dict):
        return {
            key: _move_value_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _move_value_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _move_value_to_device(item, device)
            for item in value
        )

    return value


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    将优化器内部状态移动到模型所在设备。

    AdamW 的 exp_avg、exp_avg_sq 等状态会参与后续参数更新，
    必须和对应模型参数位于同一个设备。
    """

    for parameter, parameter_state in optimizer.state.items():
        optimizer.state[parameter] = _move_value_to_device(
            parameter_state,
            device,
        )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    train_loader: Any | None = None,
    config: Any | None = None,
    val_loss: float | None = None,
    save_rng_state: bool = True,
) -> None:
    """
    保存训练 Checkpoint。

    参数：
        path:
            Checkpoint 保存路径。

        model:
            模型对象，可以是普通模型、DDP 模型或 compile 模型。

        optimizer:
            优化器。只保存模型时可以传入 None。

        step:
            已经完成的训练 step。

        train_loader:
            可选的数据加载器，需要实现 state_dict()。

        config:
            可选的完整训练配置。
            应由 train_gpt2.py 显式传入，而不是从 model.config 获取。

        val_loss:
            保存时对应的验证损失。

        save_rng_state:
            是否保存 CPU 和 CUDA 随机数状态。
    """

    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_model = unwrap_model(model)

    checkpoint: dict[str, Any] = {
        "model": raw_model.state_dict(),
        "step": step,
    }
    # GPT 模型通常具有 config 属性，但通用测试模型不一定具有。
    # 只有模型确实提供配置时，才保存模型结构配置。
    model_config = getattr(raw_model, "config", None)
    if model_config is not None:
        checkpoint["model_config"] = serialize_config(model_config)

    # 完整训练配置由训练入口显式传入。
    if config is not None:
        checkpoint["train_config"] = serialize_config(config)

    # 只在调用方提供优化器时保存优化器状态。
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if val_loss is not None:
        checkpoint["val_loss"] = val_loss

    if train_loader is not None:
        if not hasattr(train_loader, "state_dict"):
            raise TypeError(
                "train_loader 必须实现 state_dict()"
            )

        checkpoint["train_loader"] = (
            train_loader.state_dict()
        )

    if save_rng_state:
        # torch.get_rng_state() 返回 CPU ByteTensor。
        checkpoint["cpu_rng_state"] = (
            torch.get_rng_state()
        )

        if torch.cuda.is_available():
            # CUDA RNG 状态也以 CPU ByteTensor 列表形式保存。
            checkpoint["cuda_rng_state"] = (
                torch.cuda.get_rng_state_all()
            )

    # 先写入临时文件，成功后再替换正式文件。
    # 可以降低程序中途退出造成 Checkpoint 损坏的风险。
    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device | None = None,
    train_loader: Any | None = None,
    restore_rng_state: bool = True,
) -> dict[str, Any]:
    """
    加载 Checkpoint，并恢复训练状态。

    Checkpoint 首先加载到 CPU，再根据各状态的语义分别恢复：

    - 模型参数：复制到模型当前所在设备；
    - 优化器状态：移动到模型设备；
    - CPU RNG：保留在 CPU；
    - CUDA RNG：通过 CUDA RNG 接口恢复。
    """

    checkpoint_path = Path(path).expanduser().resolve()

    # 提前检查路径，从而提供清晰、稳定的异常信息。
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint 文件不存在：{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint 最外层必须是字典"
        )

    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint 中缺少 model 状态"
        )

    raw_model = unwrap_model(model)

    # strict=True 保证模型结构和保存时完全匹配。
    raw_model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    if device is None:
        target_device = infer_model_device(raw_model)
    else:
        target_device = torch.device(device)

    if optimizer is not None:
        # 调用方提供 optimizer，说明当前目的是恢复训练。
        # 这时 optimizer 状态缺失应当直接报错，而不是静默跳过。
        if "optimizer" not in checkpoint:
            raise KeyError(
                "Checkpoint 中缺少 optimizer 状态"
            )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        move_optimizer_to_device(
            optimizer,
            target_device,
        )

    if train_loader is not None:
        train_loader_state = checkpoint.get(
            "train_loader"
        )

        if train_loader_state is not None:
            if not hasattr(
                train_loader,
                "load_state_dict",
            ):
                raise TypeError(
                    "train_loader 必须实现 "
                    "load_state_dict()"
                )

            train_loader.load_state_dict(
                train_loader_state
            )

    if restore_rng_state:
        # RNG 字段属于可选恢复状态。
        # 旧版 Checkpoint 中不存在时，允许跳过。
        cpu_rng_state = checkpoint.get(
            "cpu_rng_state"
        )

        if cpu_rng_state is not None:
            # torch.set_rng_state() 要求 CPU ByteTensor。
            cpu_rng_state = cpu_rng_state.to(
                device="cpu",
                dtype=torch.uint8,
            )

            torch.set_rng_state(
                cpu_rng_state
            )

        cuda_rng_states = checkpoint.get(
            "cuda_rng_state"
        )

        if (
            cuda_rng_states is not None
            and torch.cuda.is_available()
        ):
            cuda_rng_states = [
                state.to(
                    device="cpu",
                    dtype=torch.uint8,
                )
                for state in cuda_rng_states
            ]

            torch.cuda.set_rng_state_all(
                cuda_rng_states
            )

    return checkpoint