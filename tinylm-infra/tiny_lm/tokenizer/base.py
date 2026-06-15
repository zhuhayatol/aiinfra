
def get_stats(text:list) -> dict[tuple, int]:
    stats = {}
    for i,j in zip(text[:-1], text[1:]):
        stats[(i,j)] = stats.get((i,j), 0) + 1
    return stats

def merge(tokens:list, pair:tuple[int, int] , count:int):
    i = 0
    result = []
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]: 
            result.append(count)
            i = i + 2
        else:
            result.append(tokens[i])
            i = i + 1
    return result

class Tokenizer:
    def __init__(self):
        self.merges:dict[tuple, int] = {}
        self.vocab:dict[int, bytes] = self._build_vocab_()
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    def encode(self, text : str):
        token = list(map(int, text.encode('utf-8')))
        return token
    def decode(self, ids:list):
        ids = b"".join(self.vocab[idx] for idx in ids)
        return ids.decode('utf-8')
    def _build_vocab_(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
    def print_merge(self):
        result = "\n".join(
            f"{k}\t -> \t{v}"
            for k,v in self.merges.items()
        )
        return result 

