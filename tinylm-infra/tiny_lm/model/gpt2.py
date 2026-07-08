from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    bias: bool  = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss # logits:[B, T, vocab_size]
    

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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
            idx_next = torch.multinomial(topk_probs, num_samples=1)
            # 合并旧的和新的idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type="gpt2", model_path=None):

        from transformers import GPT2LMHeadModel


        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        if model_path != None:
            model_hf = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        
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

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
 



if __name__ == "__main__":
    import torch
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    model = GPT.from_pretrained("gpt2", model_path="./gpt2_huggingface")
    model.eval()
    model.to("cuda")

    text = "hello, im from Peking university"
    idx = torch.tensor(enc.encode(text), dtype=torch.long)[None, :].to("cuda")

    out = model.generate(
        idx,
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
    )

    print(enc.decode(out[0].tolist()))