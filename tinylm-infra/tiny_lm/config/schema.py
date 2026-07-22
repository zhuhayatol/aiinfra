# tiny_lm/config/schema.py

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """一次训练实验的基本信息。"""

    # 实验名称，用于区分不同训练任务。
    name: str

    # 全局随机种子，用于尽可能保证实验可复现。
    seed: int

    # 日志、权重和配置副本等实验产物的保存目录。
    output_dir: str


@dataclass
class ModelConfig:
    """GPT 模型的结构配置。"""

    # 模型能够处理的最大序列长度。
    block_size: int

    # 词表大小，即模型输出 logits 的最后一个维度。
    vocab_size: int

    # Transformer Block 的层数。
    n_layer: int

    # 多头注意力中的注意力头数量。
    n_head: int

    # Token 隐藏向量的维度。
    n_embd: int


@dataclass
class DataConfig:
    """训练数据及 DataLoader 的相关配置。"""

    # 预处理后训练数据所在目录。
    data_dir: str

    # 每个进程一次读取的样本数量。(B)
    batch_size: int

    # 每个训练样本包含的 token 数量。(T)
    sequence_length: int


@dataclass
class TrainingConfig:
    """训练循环、优化器和精度相关配置。"""

    # 本次训练总共执行的优化步骤数。
    max_steps: int

    # 一次参数更新所对应的全局 token 数量。
    #
    # 当 micro batch 较小时，会通过梯度累积达到该值。
    total_batch_size: int

    # 学习率调度中的最大学习率。
    max_learning_rate: float

    # 余弦退火结束时的最小学习率。
    min_learning_rate: float

    # 学习率从较小值增长到最大学习率所需的步数。
    warmup_steps: int

    # AdamW 中的权重衰减系数。
    weight_decay: float

    # 梯度裁剪的最大范数，用于减少梯度爆炸风险。
    gradient_clip: float

    # 训练所使用的数据类型。
    # 常见值为 float32、float16、bfloat16。
    dtype: str = "bfloat16"

    # 是否使用 torch.compile 编译模型。
    use_compile: bool = False

    # 是否在硬件支持时使用 fused AdamW。
    use_fused_adamw: bool = True


@dataclass
class EvaluationConfig:
    """验证集评估及外部评测配置。"""

    # 每隔多少个训练 step 执行一次验证。
    eval_interval: int

    # 每次验证使用多少个验证 batch。
    eval_steps: int

    # 是否运行 HellaSwag 评测。
    run_hellaswag: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint 保存与恢复相关配置。"""

    # 每隔多少个训练 step 保存一次 checkpoint。
    save_interval: int

    # 是否保存优化器状态。
    # 若要恢复训练，一般应设置为 True。
    save_optimizer: bool = True

    # 是否保存 CPU 和 CUDA 随机数状态。
    save_rng_state: bool = True

    # 是否保存 DataLoader 当前读取位置。
    save_dataloader_state: bool = True

    # 默认恢复的 checkpoint 路径。
    # 为 None 时表示从头开始训练。
    resume_from: str | None = None


@dataclass
class Config:
    """
    整个训练任务的顶层配置。

    该类将实验、模型、数据、训练、评估等配置统一组织起来，
    训练代码只需要接收一个 Config 对象。
    """

    # 实验名称、随机种子及输出目录。
    experiment: ExperimentConfig

    # GPT 模型结构。
    model: ModelConfig

    # 数据集及批处理设置。
    data: DataConfig

    # 优化器、学习率、训练步数等设置。
    training: TrainingConfig

    # 验证和 HellaSwag 设置。
    evaluation: EvaluationConfig

    # Checkpoint 保存和恢复设置。
    checkpoint: CheckpointConfig
