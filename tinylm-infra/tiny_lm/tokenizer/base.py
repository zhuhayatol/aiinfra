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
        self.pattern:str = ""
        self.special_tokens:dict[str, int] = {}

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text : str):
        token = list(map(int, text.encode('utf-8')))
        return token
    
    def decode(self, ids:list):
        ids = b"".join(self.vocab[idx] for idx in ids)
        return ids.decode('utf-8')
    
    def _build_vocab_(self):
        """
        从 self.merges 构建一个 vocab 出来
        """
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
    
    def save(self, file_prefix:str):
        """
        保存两个文件：
        file_prefix.model:用于 load
        file_prefix.vocab:用于人类查看
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding='utf-8') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")

            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx} \n") 
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")

        self.save_vocab(file_prefix + ".vocab")           

    def load(self, model_file:str):
        """
        从 .model 文件加载 tokenizer。
        """
        assert model_file.endswith(".model")
        with open(model_file, "r", encoding="utf-8") as f: 

            version = f.readline().strip()
            if version != "minbpe v1":
                raise ValueError(f"Unsupported model version: {version}")
            
            self.pattern = f.readline().strip()

            num_special = int(f.readline().strip())
            self.special_tokens = {}

            for _ in range(num_special):
                line = f.readline().rstrip("\n")
                special, idx = line.rsplit(" ", 1)
                self.special_tokens[special] = int(idx)
                self.special_tokens[special] = int(idx)

            self.merges = {}

            idx = 256 
            for line in f:
                line = line.strip()
                if not line:
                    continue

                p0, p1 = map(int, line.split())
                self.merges[(p0, p1)] = idx
                idx += 1

        self.vocab = self._build_vocab_()

    def save_vocab(self, vocab_file: str):
        """
        保存一个给人看的 vocab 文件。
        """
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                token_repr = token.decode("utf-8", errors="replace")

                if idx in inverted_merges:
                    p0, p1 = inverted_merges[idx]
                    p0_repr = self.vocab[p0].decode("utf-8", errors="replace")
                    p1_repr = self.vocab[p1].decode("utf-8", errors="replace")
                    f.write(f"[{p0_repr}][{p1_repr}] -> [{token_repr}] {idx}\n")
                else:
                    f.write(f"[{token_repr}] {idx}\n")