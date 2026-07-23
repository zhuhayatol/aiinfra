# tiny_lm/config/loader.py

from pathlib import Path
from typing import Any

import yaml

from tiny_lm.config.schema import (
    CheckpointConfig,
    Config,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


def load_config(path: str | Path) -> Config:
    """
    从 YAML 文件中读取训练配置。

    参数：
        path:
            YAML 配置文件路径，可以是字符串或 Path 对象。

    返回：
        解析并转换完成的 Config 对象。

    异常：
        FileNotFoundError:
            配置文件不存在。

        ValueError:
            配置文件为空、缺少配置组，或者字段名称错误。
    """

    # expanduser() 用于展开路径中的 "~"。
    # resolve() 将路径转换为绝对路径，减少工作目录变化带来的影响。
    config_path = Path(path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在：{config_path}"
        )

    # safe_load 不会执行 YAML 中潜在的任意 Python 对象，
    # 比普通 yaml.load 更适合读取项目配置。
    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] | None = yaml.safe_load(file)

    if raw_config is None:
        raise ValueError(
            f"配置文件为空：{config_path}"
        )

    if not isinstance(raw_config, dict):
        raise ValueError(
            "配置文件最外层必须是键值映射结构"
        )

    try:
        # 将 YAML 中的嵌套字典分别转换为对应的 dataclass。
        #
        # 例如：
        # raw_config["model"]
        # 会被转换为 ModelConfig 对象。
        config = Config(
            experiment=ExperimentConfig(
                **raw_config["experiment"]
            ),
            model=ModelConfig(
                **raw_config["model"]
            ),
            data=DataConfig(
                **raw_config["data"]
            ),
            training=TrainingConfig(
                **raw_config["training"]
            ),
            evaluation=EvaluationConfig(
                **raw_config["evaluation"]
            ),
            checkpoint=CheckpointConfig(
                **raw_config["checkpoint"]
            ),
        )

    except KeyError as error:
        # 当 experiment、model 等整个配置组缺失时，
        # 会进入此分支。
        missing_group = error.args[0]

        raise ValueError(
            f"配置文件缺少配置组：{missing_group}"
        ) from error

    except TypeError as error:
        # 常见原因包括：
        # 1. 字段名称拼写错误；
        # 2. 出现 dataclass 中未定义的字段；
        # 3. 某个必填字段缺失。
        raise ValueError(
            f"配置字段名称或结构错误：{error}"
        ) from error

    return config