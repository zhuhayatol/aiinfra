# tiny_lm/utils/plot.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_log(
    log_path: str | Path,
    save_path: str | Path,
) -> None:
    """
    读取训练日志并绘制训练曲线。

    日志格式：
        step stream value

    例如：
        0 train 10.9234
        0 val 10.8123
        50 hella 0.2450
    """

    log_path = Path(log_path)
    save_path = Path(save_path)

    if not log_path.is_file():
        raise FileNotFoundError(
            f"训练日志不存在：{log_path}"
        )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    streams: dict[str, dict[int, float]] = {}

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            step, stream, value = line.split()
            streams.setdefault(stream, {})[int(step)] = float(value)

    streams_xy = {
        stream: tuple(zip(*sorted(values.items())))
        for stream, values in streams.items()
        if values
    }

    print("Found streams:", list(streams_xy.keys()))

    plt.figure(figsize=(14, 5))

    # 图1：训练集和验证集 Loss。
    plt.subplot(1, 2, 1)

    if "train" in streams_xy:
        xs, ys = streams_xy["train"]
        ys = np.asarray(ys)
        plt.plot(xs, ys, marker="o", label="train loss")
        print("Min train loss:", ys.min())

    if "val" in streams_xy:
        xs, ys = streams_xy["val"]
        ys = np.asarray(ys)
        plt.plot(xs, ys, marker="o", label="val loss")
        print("Min val loss:", ys.min())

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 图2：HellaSwag 准确率。
    plt.subplot(1, 2, 2)

    if "hella" in streams_xy:
        xs, ys = streams_xy["hella"]
        ys = np.asarray(ys)
        plt.plot(xs, ys, marker="o", label="HellaSwag acc")
        print("Max HellaSwag acc:", ys.max())

    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("HellaSwag eval")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved figure to: {save_path}")


def parse_args() -> argparse.Namespace:
    """解析日志路径和图片保存路径。"""

    parser = argparse.ArgumentParser(
        description="绘制 TinyLM 训练曲线"
    )

    parser.add_argument(
        "--log",
        required=True,
        help="训练日志路径",
    )

    parser.add_argument(
        "--save",
        default="training_curves.png",
        help="图片保存路径",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_log(
        log_path=args.log,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
