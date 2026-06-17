import regex as re
from .base import Tokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern :str | None = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
    
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
    
    def encode_chunk(self, chunk:str) -> list[int]:
        chunk_list = list(map(int, chunk.encode("utf-8")))
        
        while len(chunk_list) >= 2:
            stats = get_stats(chunk_list)
            if stats == {}:
                break

            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break
            
            chunk_list = merge(chunk_list, pair, self.merges[pair])
        return chunk_list

    def encode(self, text:str) -> list[int]:
        """
        encode的过程：
        首先使用regex分块，
        随后再块内分别根据merges来编码合并，
        最后将多个块全部合并
        """
        text_regex = re.findall(self.compiled_pattern, text)
        
        text_chunk = [self.encode_chunk(t) for t in text_regex]

        tokens = []
        for t in text_chunk:
            tokens.extend(t)
        
        return tokens 
    