from tiny_lm.tokenizer.basic import BPETokenizer
from tiny_lm.tokenizer.base import merge, get_stats

def test_get_stats():
    ids = [1, 2, 3, 1, 2]
    stats = get_stats(ids)

    assert stats[(1, 2)] == 2
    assert stats[(2, 3)] == 1
    assert stats[(3, 1)] == 1


def test_merge():
    ids = [1, 2, 3, 1, 2]
    out = merge(ids, (1, 2), 99)

    assert out == [99, 3, 99]


def test_roundtrip_english():
    text = "def train(self, text, vocab_size, verbose=False)"

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=270)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_roundtrip_chinese():
    text = "加缪是一个很有计划的人。"

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=280)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_empty_and_short_text():
    tokenizer = BPETokenizer()
    tokenizer.train("hello hello", vocab_size=260)

    assert tokenizer.decode(tokenizer.encode("")) == ""
    assert tokenizer.decode(tokenizer.encode("a")) == "a"