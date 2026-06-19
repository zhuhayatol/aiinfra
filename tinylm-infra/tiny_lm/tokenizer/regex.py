import regex as re
from .base import Tokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern :str | None = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens:dict[str, int] = {}
        self.inverse_special_tokens = {}

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: i for i, v in self.special_tokens.items()}
        self.vocab = self._build_vocab_()

    def train(self, text, vocab_size, verbose=False):
        """
        首先是把文本通过regex切分为块，
        随后在块中分别找高频pair，
        再把所有的pair汇总
        为全部的pair，
        拿到最高频的pair之后对所有的chunk进行merge
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be greater than or equal to 256")

        text_regex = re.findall(self.compiled_pattern, text)

        text_list = [list(map(int, t.encode('utf-8'))) for t in text_regex]

        idx = 256
        nums_merge = vocab_size - 256
        merges = {}
        for k in range(nums_merge):
            stats = {}
 
            for t in text_list:
                temp = get_stats(t)
                for i, j in temp.items():
                    stats[i] = stats.get(i, 0) + j
            
            if stats == {}:
                break

            pair = max(stats, key=stats.get)
            
            for t in range(len(text_list)):
                t1 = merge(text_list[t], pair, idx + k)
                text_list[t] = t1
            merges[pair] = idx + k
            if verbose == True:
                print(f"merge {pair} to {idx + k}.")
        
        result = []
        for t in text_list:
            result.extend(t)
        self.merges = merges
        self.vocab  = self._build_vocab_()

        # print([j for i, j in self.vocab.items() if i > 255])

    def encode_chunk(self, chunk) -> list[int]:
        chunk = list(chunk)
        while len(chunk) >= 2:
            stats = get_stats(chunk)
            if stats == {}:
                break

            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break
            
            chunk = merge(chunk, pair, self.merges[pair])
        return chunk

    def encode_ordinary(self, text) -> list[int]:
        """
        encode_ordinary的过程：
        首先使用regex分块，
        随后再块内分别根据merges来编码合并，
        最后将多个块全部合并
        (不考虑这个special_tokens)
        """
        text_regex = re.findall(self.compiled_pattern, text)
        
        text_chunk = [self.encode_chunk(t.encode('utf-8')) for t in text_regex]

        tokens = []
        for t in text_chunk:
            tokens.extend(t)
        
        return tokens 
    

    def encode(self, text, allowed_special="none_raise"):
        special = {}
        if allowed_special == "none_raise":
            for i,j in self.special_tokens.items():
                if i in text:
                    raise ValueError("wo dont need any special token!")
            return self.encode_ordinary(text)
        elif allowed_special == "none":
            return self.encode_ordinary(text)
        elif allowed_special == "all":
            specials = self.special_tokens
        elif isinstance(allowed_special, set):
            specials = {k:v for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        # 从text中分离出special中的所有元素
        if not specials:
            return self.encode_ordinary(text)
        special_tokens_sorted = sorted(specials, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens_sorted) + ")"
        special_chunks = re.split(special_pattern, text)

        tokens = []
        for sc in special_chunks:
            if sc in specials:
                tokens.append(specials[sc])
            else:
                tokens.extend(self.encode_ordinary(sc))
        return tokens
    
    
    def load(self, model_file):
        super().load(model_file)
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}