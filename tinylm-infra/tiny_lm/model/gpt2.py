from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from pathlib import Path

@dataclass
class GPTConfig:
    block_size: int = 1024 # 最长的上下文
    vocab_size: int = 50257 # token数量: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # block层数
    n_head: int = 12 # 多头注意力中的头的个数
    n_embd: int = 768 # 嵌入层的输出维度
    bias: bool  = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        # 将q,k,v组合在一个矩阵中
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # 包含这个标记的层，需要进行残差初始化缩放
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)

        # 得到qkv，注意shape
        q, k, v = qkv.split(self.n_embd, dim = 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #v.shape = [B, n_head, T, head_dim]
        
        # 上文注意shape的原因就是该函数的输入有要求 [B, n_head, T, head_dim]
        # 这里是一种flash attention的实现
        # is_causal 代表只看过去不看未来，等同于mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # transpose这里转置将数据变得不连续
        # 类似本身的[[123][456]]，我们转置之后认为是[[14][25][36]],但是内存顺序没有变化，仅仅是逻辑顺序变了
        # 而view需要在连续的内存上进行变换，所以这里需要将数据变为连续
        # 也就是会开辟一个新的内存，按照转置之后的逻辑顺序存储：142536
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # 包含这个标记的层，需要进行残差初始化缩放
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd)
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        # 将嵌入输入层与分类器输出层的参数关联起来，减少大量参数和显存开销。
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weight)

    def _init_weight(self, module):
        # 控制linear层的输出分布
        if isinstance(module, nn.Linear):
            std = 0.02
            
            # 对这部分参与残差计算的层进行放缩
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # 因为每层block中有俩残差，所以我们是2 * n_layer
                std *= (2 * self.config.n_layer) ** -0.5
            
            # 通过控制参数的初始分布，
            # 使每一层在训练开始时输出的统计分布保持稳定，
            # 避免随着网络加深而不断放大或缩小。
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)

            # bias偏差只会对输出进行平移， 整体数据的方差没有变化，初始化的时候直接置零
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # 注意因为linear与embed的工作方式不同，所以需要区别开来
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, target=None):
        B, T = idx.size()

        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        # 位置编码和文本编码
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # [T]
        pos_emb = self.transformer.wpe(pos) # [T, n_embd]
        tok_emb = self.transformer.wte(idx) # [B, T, n_embd]

        idx = pos_emb + tok_emb # [B, T, n_embd]

        for block in self.transformer.h:
            idx = block(idx) # [B, T, C]
        
        idx = self.transformer.ln_f(idx) # [B, T, C]

        logits = self.lm_head(idx) # [B, T, vocab_size]
        loss = None

        if target is not None:
            
            # param1 = [B*T, vocab_size]; parm2 = [B*T]
            # 注意参数的shape
            loss:torch.tensor = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss # logits:[B, T, vocab_size]
    

    @torch.no_grad()
    # idx.shape = (1, xxx)
    # 创建idx的时候需要
    # idx = torch.tensor(enc.encode(text), dtype=torch.long)[None, :].to(device)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
        for _ in range(max_new_tokens):
            # 注意idx的大小：超过blocksize则取最后blocksize个数据
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            # 这里只取最后一个时间的结果
            logits = logits[:, -1, :] #[B, vocab_size]

            logits = logits / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                topk_probs, _ = torch.topk(logits, top_k, dim=-1)
                # topk_probs[:, [-1]]就是每一个batch中概率排第k的概率值
                # 将logits每个batch中小于该概率的部分赋值为-inf
                logits[logits < topk_probs[:, [-1]]] = -float('inf')
            # 在筛选完topk之后再进行softmax
            topk_probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(topk_probs, num_samples=1, generator=generator)
            # 合并旧的和新的idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type="gpt2", model_path:str | Path | None=None):

        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # 始终不变
        config_args['block_size'] = 1024 # 始终不变

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        if model_path != None:
            model_path = Path(model_path).expanduser().resolve()

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Local pretrained model path does not exist: "
                    f"{model_path}"
                )

            if not model_path.is_dir():
                raise NotADirectoryError(
                    f"Local pretrained model path must be a directory: "
                    f"{model_path}"
                )
            
            model_source = str(model_path)
            
            # 用户显式指定本地路径时，禁止静默访问网络。
            model_hf = GPT2LMHeadModel.from_pretrained(model_source, local_files_only=True)
        else:
            # 没有指定本地目录时：
            # Hugging Face 会使用本地缓存，缓存没有时再访问 Hub。
            model_source = model_type 
            model_hf = GPT2LMHeadModel.from_pretrained(model_source)
        
        # 模型中所有参数的字典
        # sd_hf是huggingface中的， sd是初始化模型之后的， 在加载hf的数据之前需要先对齐
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        # 过滤 HuggingFace 中不需要的 key
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        assert len(sd_keys) == len(sd_keys_hf), (
                f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"
            )
        # hg and myself 
        # 需要转置的部分

        # transformer.h.0.attn.c_attn.weight            (768, 2304)          
        # transformer.h.0.attn.c_attn.weight            (2304, 768)

        # transformer.h.0.attn.c_proj.weight           (768, 768)           
        # transformer.h.0.attn.c_proj.weight           (768, 768)

        # transformer.h.0.mlp.c_fc.weight               (768, 3072)          
        # transformer.h.0.mlp.c_fc.weight               (3072, 768)

        # transformer.h.0.mlp.c_proj.weight             (3072, 768)          
        # transformer.h.0.mlp.c_proj.weight             (768, 3072)

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # 加载hf的参数 到 刚刚初始化的模型中，直接将对应参数进行操作后复制过去
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    # 如果是需要转置的参数， 则直接将参数转置之后copy_到现在的模型中
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    # 剩余的不需要转置， 直接复制
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        
        # 取得模型参数
        param_dict = {n : p for n, p in self.named_parameters()}
        param_dict = {n : p for n, p in param_dict.items() if p.requires_grad }

        # 按照维度分组
        decay_params = [param for param in param_dict.values() if param.dim() >= 2]
        no_decay_params = [param for param in param_dict.values() if param.dim() < 2]

        # 构造parameter groups
        optim_groups = [
            {"params": decay_params,"weight_decay": weight_decay,},
            {"params": no_decay_params,"weight_decay": 0.0},
        ]

        # 参数数量统计
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        print(f"we have the number of tensor which needed to be decayed : {len(decay_params)}, they have {num_decay_params} params")
        print(f"we have the number of tensor which needed not to be decayed : {len(no_decay_params)}, they have {num_no_decay_params} params")

        # 判断是否可以使用fused
        use_fused = False
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            if device_type == "cuda":
                use_fused = True
        
        optimizer = torch.optim.AdamW(optim_groups, learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer