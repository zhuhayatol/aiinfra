import tiktoken
from .regex import RegexTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks, token, max_rank):
        tokens = [bytes([t]) for t in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(tokens[:-1], tokens[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and rank < max_rank and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (min_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            tokens = tokens[:min_idx] + [tokens[min_idx] + tokens[min_idx + 1]] + tokens[min_idx + 2:]
        return tokens


def recover_merges(mergeable_ranks:dict[bytes, int]):
    merges = {}
    for token, rank in sorted(mergeable_ranks.items(), key=lambda x:x[1]):
            if len(token) == 1:
                continue
            else:
                pair = tuple(bpe(mergeable_ranks, token, rank))
                assert len(pair) == 2
                idx1 = mergeable_ranks.get(pair[0])
                idx2 = mergeable_ranks.get(pair[1])
                merges[(idx1, idx2)] = rank
    return merges
    

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern = GPT4_SPLIT_PATTERN)

        # 加载tiktoken的cl100k_base,拿到gpt-4中的内部merges
        # mergeable_ranks : bytes      -- > int
        # self.merges     : (int, int) -- > int
        # self.vocab      : int        -- > bytes
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)

        
        self.vocab = self._build_vocab_()

        self.byte_shuffle = {i : mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k,v in self.byte_shuffle.items()}
                
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)
    
    def encode_chunk(self, chunk):
        chunk = bytes(self.byte_shuffle[ch] for ch in chunk)
        return super().encode_chunk(chunk)

    def decode(self, ids):
        result = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                result.append(self.inverse_special_tokens[idx].encode('utf-8'))
            elif idx in self.vocab:
                text_bytes = self.vocab[idx]
                text_shuffle_bytes = bytes(self.inverse_byte_shuffle[i] for i in text_bytes)
                result.append(text_shuffle_bytes)
            else:
                raise ValueError(f"invalid token id {idx}")
        text = b''.join(result)
        return text.decode('utf-8')
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")
    
    def save_vocab(self, vocab_file):
        # just for visualization purposes let's output the GPT-4 tokens
        # in the exact same format as the base class would.
        # simple run as:
        # python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"
        from .base import render_token
        # build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # now merge the shuffled bytes and write to file
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
