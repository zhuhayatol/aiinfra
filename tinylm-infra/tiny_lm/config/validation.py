# tiny_lm/config/validation.py

from tiny_lm.config.schema import Config


# 当前训练代码允许使用的浮点精度。
SUPPORTED_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
}


def validate_config(
    config: Config,
    world_size: int = 1,
) -> None:
    """
    检查训练配置是否合法。

    参数：
        config:
            已经由 load_config() 解析完成的配置对象。

        world_size:
            分布式训练进程数。
            单卡训练时通常为 1。

    异常：
        ValueError:
            任意配置不满足约束时抛出。
    """

    if world_size <= 0:
        raise ValueError(
            "world_size 必须大于 0"
        )

    # 每个注意力头负责的维度为：
    #
    # head_dim = n_embd // n_head
    #
    # 因此隐藏维度必须能够被注意力头数整除。
    if config.model.n_embd % config.model.n_head != 0:
        raise ValueError(
            "model.n_embd 必须能够被 "
            "model.n_head 整除"
        )

    # 训练序列不能超过模型支持的最大上下文长度。
    if (
        config.data.sequence_length
        > config.model.block_size
    ):
        raise ValueError(
            "data.sequence_length 不能超过 "
            "model.block_size"
        )

    if config.data.batch_size <= 0:
        raise ValueError(
            "data.batch_size 必须大于 0"
        )

    if config.data.sequence_length <= 0:
        raise ValueError(
            "data.sequence_length 必须大于 0"
        )

    if config.training.max_steps <= 0:
        raise ValueError(
            "training.max_steps 必须大于 0"
        )

    if config.training.warmup_steps < 0:
        raise ValueError(
            "training.warmup_steps 不能小于 0"
        )

    # Warmup 不能比整个训练过程还长。
    if (
        config.training.warmup_steps
        > config.training.max_steps
    ):
        raise ValueError(
            "training.warmup_steps 不能大于 "
            "training.max_steps"
        )

    if (
        config.training.max_learning_rate
        <= 0
    ):
        raise ValueError(
            "training.max_learning_rate 必须大于 0"
        )

    if (
        config.training.min_learning_rate
        < 0
    ):
        raise ValueError(
            "training.min_learning_rate 不能小于 0"
        )

    # 通常最小学习率不应该高于最大学习率。
    if (
        config.training.min_learning_rate
        > config.training.max_learning_rate
    ):
        raise ValueError(
            "training.min_learning_rate 不能大于 "
            "training.max_learning_rate"
        )

    if config.training.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"不支持的 dtype："
            f"{config.training.dtype}，"
            f"支持的值为："
            f"{sorted(SUPPORTED_DTYPES)}"
        )

    # 一个 micro batch 中参与训练的 token 数：
    #
    # 单进程 token 数：
    # batch_size × sequence_length
    #
    # DDP 下所有进程合计：
    # batch_size × sequence_length × world_size
    tokens_per_micro_batch = (
        config.data.batch_size
        * config.data.sequence_length
        * world_size
    )

    # 全局 Batch Token 数通常通过梯度累积得到：
    #
    # gradient_accumulation_steps
    # =
    # total_batch_size / tokens_per_micro_batch
    #
    # 因此这里必须能够整除。
    if (
        config.training.total_batch_size
        % tokens_per_micro_batch
        != 0
    ):
        raise ValueError(
            "training.total_batch_size 必须能够被 "
            "data.batch_size × "
            "data.sequence_length × "
            "world_size 整除"
        )

    if config.evaluation.eval_interval <= 0:
        raise ValueError(
            "evaluation.eval_interval 必须大于 0"
        )

    if config.evaluation.eval_steps <= 0:
        raise ValueError(
            "evaluation.eval_steps 必须大于 0"
        )

    if config.checkpoint.save_interval <= 0:
        raise ValueError(
            "checkpoint.save_interval 必须大于 0"
        )

    print("validate yaml pass!!!!")