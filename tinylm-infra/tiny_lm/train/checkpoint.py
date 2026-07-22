from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """兼容普通模型、DDP 和 torch.compile。"""
    if hasattr(model, "module"):
        model = model.module

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model

# 将 GPTConfig 转换为字典
def serialize_config(config: Any) -> dict:
    if is_dataclass(config):
        return asdict(config)

    if isinstance(config, dict):
        return config

    raise TypeError(
        f"Config must be a dataclass or dict, got {type(config)}"
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loader=None,
    val_loss: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_model(model)

    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": serialize_config(raw_model.config),
        "step": step,
        "val_loss": val_loss,
        # 获取并保存当前 CPU 上的随机数生成器（RNG）状态
        # 方便复现和断点续训
        "cpu_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    if train_loader is not None:
        checkpoint["train_loader"] = train_loader.state_dict()

    # 先写临时文件，再替换，减少异常中断留下损坏文件的风险。
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    train_loader=None,
    device: str | torch.device = "cpu",
) -> dict:
    
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    raw_model = unwrap_model(model)
    raw_model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "cpu_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["cpu_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    if train_loader is not None and "train_loader" in checkpoint:
        train_loader.load_state_dict(checkpoint["train_loader"])

    return checkpoint