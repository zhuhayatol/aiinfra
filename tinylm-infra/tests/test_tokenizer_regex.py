import pytest

from tiny_lm.tokenizer.regex import RegexTokenizer


@pytest.mark.parametrize(
    "text,vocab_size",
    [
        ("hello world!", 280),
        ("hello, world! this is a test.", 300),
        ("加缪是一个很有计划的人。", 320),
        ("hello, 世界! 123", 320),
        ("hello\nworld\n加缪", 320),
        ("I'm learning tokenizer. you're learning too.", 340),
    ],
)
def test_regex_tokenizer_roundtrip(text, vocab_size):
    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=vocab_size)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_regex_tokenizer_encode_returns_list_of_ints():
    text = "hello, world!"

    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=280)

    ids = tokenizer.encode(text)

    assert isinstance(ids, list)
    assert all(isinstance(x, int) for x in ids)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "!",
        "。",
        "\n",
        "你",
    ],
)
def test_regex_tokenizer_handles_empty_and_short_text(text):
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world!", vocab_size=280)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_regex_tokenizer_train_stops_when_no_pairs():
    """
    这个测试专门检查：
    如果 regex 把文本切成单字符 chunk，那么每个 chunk 内部都没有 pair，
    train 不应该报错，而应该直接停止。
    """
    tokenizer = RegexTokenizer(pattern=r".")

    tokenizer.train("aaaa", vocab_size=260)

    assert tokenizer.merges == {}


def test_regex_tokenizer_does_not_merge_across_chunks():
    """
    如果 pattern=r"."，文本会被切成：
    ["a", "a", "a", "a"]

    每个 chunk 内部只有一个字符，所以不能产生 (a, a) 这种 pair。
    如果你的实现错误地把 chunk 拼起来训练，就会学到 (97, 97) -> 256。
    """
    tokenizer = RegexTokenizer(pattern=r".")

    tokenizer.train("aaaa", vocab_size=260)

    ids = tokenizer.encode("aaaa")

    assert ids == [97, 97, 97, 97]
    assert tokenizer.decode(ids) == "aaaa"


def test_regex_tokenizer_vocab_size_too_small():
    tokenizer = RegexTokenizer()

    with pytest.raises(ValueError):
        tokenizer.train("hello world", vocab_size=255)


def test_regex_tokenizer_save_load(tmp_path):
    text = "hello, world! 加缪 is learning tokenizer."

    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=320)

    prefix = str(tmp_path / "regex_tokenizer")
    tokenizer.save(prefix)

    tokenizer2 = RegexTokenizer()
    tokenizer2.load(prefix + ".model")

    ids1 = tokenizer.encode(text)
    ids2 = tokenizer2.encode(text)

    assert ids1 == ids2
    assert tokenizer2.decode(ids2) == text
    assert (tmp_path / "regex_tokenizer.model").exists()
    assert (tmp_path / "regex_tokenizer.vocab").exists()