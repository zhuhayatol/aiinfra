# tiny_lm/utils/plot_log.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_log(
    log_path="log/log.txt",
    save_path="log/training_curves.png",
):
    """
    读取训练日志并画图。

    日志格式：
        step stream value

    例如：
        0 train 10.9234
        0 val 10.8123
        50 hella 0.2450
    """

    log_path = Path(log_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    streams = {}

    # 读取日志
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            step, stream, value = line.split()

            if stream not in streams:
                streams[stream] = {}

            streams[stream][int(step)] = float(value)

    # 转换为按 step 排序后的 x/y
    streams_xy = {}

    for stream, values in streams.items():
        xy = sorted(values.items())
        steps, vals = zip(*xy)
        streams_xy[stream] = (list(steps), list(vals))

    print("Found streams:", list(streams_xy.keys()))

    plt.figure(figsize=(14, 5))

    # --------------------------------------------------
    # 图 1：loss 曲线
    # --------------------------------------------------
    plt.subplot(1, 2, 1)

    if "train" in streams_xy:
        xs, ys = streams_xy["train"]
        ys = np.array(ys)
        plt.plot(xs, ys, marker="o", label="train loss")
        print("Min train loss:", ys.min())

    if "val" in streams_xy:
        xs, ys = streams_xy["val"]
        ys = np.array(ys)
        plt.plot(xs, ys, marker="o", label="val loss")
        print("Min val loss:", ys.min())

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --------------------------------------------------
    # 图 2：HellaSwag accuracy
    # --------------------------------------------------
    plt.subplot(1, 2, 2)

    if "hella" in streams_xy:
        xs, ys = streams_xy["hella"]
        ys = np.array(ys)
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


if __name__ == "__main__":
    plot_log()