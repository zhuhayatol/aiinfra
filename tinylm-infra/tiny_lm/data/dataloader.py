import tiktoken
import torch

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, file_name:str):
        self.B = B
        self.T = T

        self.process_rank = process_rank
        self.num_processes = num_processes

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

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.token[self.current_position: self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1: ].view(B, T)

        # 每次都需要更新这个位置
        self.current_position += B * T * self.num_processes

        if self.current_position > len(self.token) - B*T*self.num_processes - 1:
            self.current_position = self.B * self.T * self.process_rank
        return x, y
