from pathlib import Path
import time
import math
import os
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from tiny_lm.dataloader.dataloader import DataLoaderLite
from tiny_lm.model.gpt2 import GPT, GPTConfig
from tiny_lm.eval.hellaswag import evaluate_hellaswag
from tiny_lm.train.checkpoint import load_checkpoint, save_checkpoint
from tiny_lm.config import load_config, validate_config

# simple launch
# python3 train_gpt2.py

# DDP launch for e.g 4 GPUs:
# torchrun --standalone --nproc_per_node=4 train_gpt2.py

def parse_args():
    parser = argparse.ArgumentParser(
        description="训练 TinyLM GPT 模型"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_gpt2.yaml",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():

    # 读取命令行参数
    args = parse_args()

    # 从YAML文件中加载完整训练配置
    config = load_config(args.config)

    print(f"使用配置文件：{args.config}")
    print(f"实验名称：{config.experiment.name}")

    # 判断是否能进行ddp
    # 初始化分布式进程组。NCCL 是 NVIDIA GPU 上 DDP 的常用后端。
    ddp = int(os.environ.get('RANK', -1)) != -1
    print(f"can we ddp?  {ddp}")

    if ddp:
        # ddp初始化
        assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
        init_process_group(backend="nccl")

        # ddp_rank：全局进程编号
        # ddp_local_rank：当前进程在本机绑定的 GPU 编号，用于选择 cuda:local_rank。
        # ddp_world_size： 总进程数量
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        # 全局 rank 为 0 的进程作为主进程，负责打印日志和保存 checkpoint
        master_process = ddp_rank == 0

        print(f"ddp_rank: {ddp_rank}, ddp_world_size: {ddp_world_size}")
    
    else:
        # 单卡训练
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # 判断设备类型
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print(f"device is {device}")

    # 在除了to(device)的场景之外使用，因为如果是多卡训练，device是cuda:0这种类型后续难使用
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # 验证yaml文件
    validate_config(config, ddp_world_size)

    enc = tiktoken.get_encoding("gpt2")

    # 梯度累计的部分
    # 总训练批次
    # DDP 下 global tokens = B * T * grad_accum_steps * ddp_world_size。
    total_batch_size = config.training.total_batch_size # 2**19, ~0.5M, in number of tokens
    B = config.data.batch_size # micro batch size
    T = config.data.sequence_length # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    
    # 每个进程本地需要累积的 micro step 数。
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # 2 ** 19 / 4 / 512 / ? = 256
    
    # 只有主进程打印一次
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # 训练数据
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, 
                             num_processes= ddp_world_size, split="train", local_dir=config.data.data_dir)

    # 验证数据集
    val_loader  = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, 
                             num_processes= ddp_world_size, split="val", local_dir=config.data.data_dir)
    
    # 允许pytorch在执行float32的矩阵乘法的时候，采用tf32的精度来加速计算
    # torch.set_float32_matmul_precision("high")

    # model = GPT.from_pretrained("gpt2", model_path="./gpt2_huggingface")
    model = GPT(GPTConfig(vocab_size=config.model.vocab_size))
    # model.eval()
    model.to(device)

    raw_model = model

    # 调整学习率
    max_lr = config.training.max_learning_rate
    min_lr = config.training.min_learning_rate

    # 预热步骤
    warmup_steps = config.training.warmup_steps # max_steps * 0.035          31
    max_steps = config.training.max_steps # 总tokens除以total_batch_size  904

    # 带预热阶段的余弦衰减学习率调度
    def get_lr(step):

        # 在预热阶段内是线性增长
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
    
        # 超过整个步骤就只输出最小学习率
        if step > max_steps:
            return min_lr
        
        # 在衰减范围内计算衰减率
        ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # 创建优化器
    optimizer = raw_model.configure_optimizers(weight_decay=config.training.weight_decay, learning_rate= config.training.max_learning_rate, device_type=device_type) 
    
    # optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device_type=device_type) 
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas=(0.9, 0.95), eps=1e-8)
    
    # 按需恢复训练

    # 命令行参数优先于 YAML 配置。
    resume_path = (
        args.resume
        if args.resume is not None
        else config.checkpoint.resume_from
    )
    # 没有指定 Checkpoint 时，从第 0 步开始训练。
    start_step = 0

    if resume_path is not None:
        checkpoint = load_checkpoint(
            path=resume_path,

            # 此时 model 尚未经过 compile 和 DDP 包装。
            model=raw_model,
            # 恢复 AdamW 的动量等优化器状态。
            optimizer=optimizer,
            # 恢复数据集当前读取位置。
            train_loader=train_loader,
        )

        if "step" not in checkpoint:
            raise KeyError(
                "Checkpoint 中缺少 step"
            )

        # Checkpoint 保存的是已经完成的训练步骤，
        # 因此恢复后从下一步继续。
        start_step = checkpoint["step"] + 1

        if master_process:
            print(
                f"已从 Checkpoint 恢复：{resume_path}"
            )
            print(
                f"已完成 step:{checkpoint['step']}"
            )
            print(
                f"将从 step {start_step} 继续训练"
            )

    else:
        if master_process:
            print("未指定 Checkpoint:将从头训练")

    # 使用compile预先编译模型，加速训练
    use_compile = config.training.use_compile
    if use_compile:
        model = torch.compile(model)

    if ddp:
        # DDP可以在反向传播的过程中，将所有计算节点上的梯度进行平均处理并且同步
        model = DDP(model, device_ids = [ddp_local_rank])

    raw_model = model.module if ddp else model

    # 创建log文件，将checkpoint和log写进去
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{config.experiment.name}.txt")

    if resume_path is not None:
        with open(log_file, "a") as f:
            pass
    else:
        with open(log_file, "w") as f:
            pass

    for step in range(start_step, max_steps):

        t0 = time.time()
        last_step = (step == max_steps - 1)

        # 每50步进行一次验证 或者 最后一次
        # 与训练过程类似，没有backward
        if step % config.evaluation.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = config.evaluation.eval_steps
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                
                # 写入文件
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

                if step > 0 and (step % config.evaluation.eval_interval == 0 or last_step):
                    # 模型checkpoint
                    # 这里保存了模型的权重，方便下一次加载
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")

                    save_checkpoint(checkpoint_path, 
                                    raw_model,
                                    optimizer,
                                    step,
                                    train_loader,
                                    config=raw_model.config,
                                    val_loss=val_loss_accum.item())


        # 验证的过程顺便进行一次采样
        if ((step > 0 and step % config.evaluation.eval_interval == 0) or last_step) and (not use_compile):
            model.eval()

            if ddp:
                dist.barrier()

            # 只需要主进程进行采样
            if master_process:
                # 一条prompt生成的结果次数
                # 一条生成序列的最大的token数量
                num_return_sequences = 4
                max_length = 32
            
                # 1. 用 GPT-2 tokenizer 把 prompt 编码成 token id
                # 2. 转成 torch tensor
                # 3. unsqueeze(0)：从 [T] 变成 [1, T]
                # 4. repeat(num_return_sequences, 1)：复制成多条序列
                tokens = enc.encode("Hello, I'm a language model,")
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
           
                # 随机数生成器：
                sample_rng = torch.Generator(device=device)
                sample_rng.manual_seed(config.experiment.seed)

                with torch.no_grad():
                    # 调用原始 GPT 模型的自定义生成函数。
                    y = raw_model.generate(
                        xgen,
                        max_new_tokens=max_length,
                        temperature=1.0,
                        top_k=50,
                        generator=sample_rng,
                    )

                for i in range(num_return_sequences):
                    decoded = enc.decode(y[i].tolist())
                    print(f"sample {i}: {decoded}")
            
            if ddp:
                dist.barrier()

        # 顺便也需要进行hellaswag评估
        if (step % config.evaluation.eval_interval == 0 or last_step) and (not use_compile):
            model.eval()

            # 确保所有进程都完成前面的验证和生成阶段，
            # 再让 rank 0 开始 HellaSwag 评估。
            if ddp:
                dist.barrier()

            if master_process:
                hs = evaluate_hellaswag(
                    raw_model,
                    device,
                    device_type,
                    amp_dtype=torch.bfloat16,
                    max_examples=200,
                )

                print(
                    f"answer: {hs['answer']}\n"
                    f"label: {hs['label_list']}\n"
                    f"hellaswag acc: "
                    f"{hs['num_correct']}/{hs['num_total']} "
                    f"= {hs['acc']:.4f}, "
                    f"skipped={hs['num_skipped']}"
                )

                with open(log_file, "a") as f:
                    f.write(f"{step} hella {hs['acc']:.4f}\n")
            
            # rank 0 完成评估之前，其他进程不能进入下一轮训练。
            if ddp:
                dist.barrier()

        # 训练过程
        # 采样结束后切回训练模式。
        model.train()
        # 在开始新一轮反向传播之前，把上一轮参数中保存的梯度清空。
        optimizer.zero_grad()

        # 损失值的累加器，必须进行累加否则直接返回loss的值本质上是最后一次microstep的loss值
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            # 得到下一批次的x，y
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # 使用混合精度，一些参数使用bf16， 一些仍然保留float32，加速矩阵计算
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            # 在梯度累积中，每个小批次需要放缩 grad_accum_steps 倍，这样才能补偿loss
            # 因为loss整个过程都会进行均值归一化
            # 作为一个整体，loss会除以整个batch
            # 可是把整个batch拆分为了多个小batch，那么每个小batch中也必须补偿除以batch数才能得到整体的loss
            loss = loss / grad_accum_steps

            # 使用detach的目的是将loss_accum摘出整个backward的graph，否则在backward的时候也会影响到这个值
            loss_accum += loss.detach()
            
            # 梯度累积时，前面的 micro step 只在本地累积梯度；
            # 最后一个 micro step 才触发 DDP 梯度同步，减少通信开销。   
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            loss.backward()
    
        # 每个进程都会有自己的loss_accum，这里是将所有进程的损失累计值求平均并且同步为新的平均值
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # 为了防止梯度突然变得很大，导致一次 optimizer.step() 把参数更新得太离谱。
        # 限制所有梯度的范式：计算所有梯度整体的 L2 Norm。
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 根据当前步骤获取现在的学习率
        lr = get_lr(step)

        # 深入优化器直接修改对应的学习率的值
        for p in optimizer.param_groups:
            p["lr"] = lr
        
        optimizer.step()

        # 等待 GPU 把之前所有提交的 CUDA 任务全部执行完成，再继续执行后面的 CPU 代码。
        # 用于准确计时
        if device_type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0) 
        
        if master_process:
            print(f"step {step:5d} , lr = {lr:.4e}, loss is {loss_accum.item():.5f}, norm = {norm:.4f},  dt = {dt:.2f} ms, token/sec = {tokens_per_sec:.2f}")

            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    if ddp:
        destroy_process_group()
if __name__ == "__main__":
    main()