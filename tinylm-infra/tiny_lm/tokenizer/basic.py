from .base import Tokenizer, merge, get_stats

class BPETokenizer(Tokenizer):
    def train(self, text:str, vocab_size, verbose=False):
        text  = list(map(int, text.encode('utf-8')))
        merges = {}
        
        if vocab_size < 256:
            raise ValueError("vocab_size must larger than or equal to 256")
        
        num_merge = vocab_size - 256
        
        for i in range(num_merge):
            stats = get_stats(text)
            pair =  max(stats, key=lambda p: stats.get(p, 0))
            if verbose:
                print(f"the pair {pair} is merged to count {256 + i}")
            text = merge(text, pair, 256 + i)
            merges[pair] = 256 + i

        self.merges = merges
        self.vocab = self._build_vocab_()

    def encode(self, text):
        """
        需要根据train之后的merge来encode,找到当前文本中存在于merge中的count最小的pair,
        随后替换这个pair
        直到文本中不再拥有关于merge内的pair为止
        """
        merges = self.merges
        text =  list(map(int, text.encode("utf-8")))

        while len(text) > 1:
            stats = get_stats(text)
            pair  = min(stats, key = lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            text = merge(text, pair, merges[pair])

        return text
    def decode(self, ids):
        """
        decode比较简单吧,直接使用vocab挨个从小往大替换即可
        """
        # byte = b"".join(self.vocab[idx] for idx in ids)
        # return byte.decode('utf-8')
        return super().decode(ids)

# if __name__ == "__main__":
#     text = "def train(self, text, vocab_size, verbose=False)"
#     text1 = "def self"
#     print(f"text1 = {list(text1.encode('utf-8'))}")
#     a = BPETokenizer()
#     a.train(text, vocab_size=262, verbose=True)
#     print(a.print_merge())
#     print(a.decode(a.encode(text1)) == text1)
