import tiktoken
import torch
import os
import numpy as np

# 加载shard文件中的numpy数据并且将他们转为格式为long类型的tensor 再返回
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) 
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split: str= None, file_name:str = None, local_dir:str = None):
        self.B = B
        self.T = T

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.split = split
        self.file_name = file_name
        self.local_dir = local_dir

        # 如果是加载本地文件
        if file_name != None:
            with open(file_name, "r") as f:
                text = f.read()

            enc = tiktoken.get_encoding("gpt2")
            token = enc.encode(text)
            self.token = torch.tensor(token)

            print(f"loaded {len(self.token)} tokens")
            print(f"1 epoch = {len(token) // B*T} batches")

            # 记录当前加载的数据的位置
            # 每个gpu所需要的数据也是不一样的
            # 根据各自的rank来进行调整
            self.current_position = self.B * self.T * self.process_rank

        # 如果是加载数据集的shard
        else:
            assert split in {"train", "val"}

            # 找到数据集的目录
            data_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), f"../../data/{local_dir}")
            )

            # 将每个shard文件的路径排序存放在shards中
            shards = os.listdir(data_root)
            shards = sorted([s for s in shards if split in s])
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards

            assert len(shards) > 0, f"no shards found for split {split}"
            if process_rank == 0:
                print(f"found {len(shards)} shards for split {split}")
            self.reset()

    # 初始化第一个shard
    # current_position则是在每个shard中维护的位置指针
    def reset(self):
        self.current_shard = 0
        self.token = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank



    def next_batch(self):
        B, T = self.B, self.T
        buf = self.token[self.current_position: self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1: ].view(B, T)

        # 每次都需要更新这个位置
        self.current_position += B * T * self.num_processes

        # 如果当前的shard中的token耗尽
        # 我们则切换到下一个shard，并且重置current_position
        if self.current_position > len(self.token) - B*T*self.num_processes - 1:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.token = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

    def num_tokens(self):
        # 如果是加载本地文件
        if self.file_name != None:
            print(f"loaded {len(self.token)} tokens")
        else:
            count_tokens = 0
            for s in self.shards:
                tokens = load_tokens(s)
                count_tokens += len(tokens)
            print(f"the {self.local_dir} has {count_tokens} tokens")