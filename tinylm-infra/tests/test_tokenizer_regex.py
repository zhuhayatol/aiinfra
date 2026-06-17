import pytest
import regex as re

from tiny_lm.tokenizer.regex import RegexTokenizer


def build_tokenizer():
    """
    构造一个已经训练好的 RegexTokenizer，方便后面测试 special tokens。
    special token 不是 train 学出来的，而是 train 后注册进去的。
    """
    tokenizer = RegexTokenizer()

    train_text = (
        "hello world! "
        "this is a tokenizer test. "
        "加缪是一个很有计划的人。"
    )

    tokenizer.train(train_text, vocab_size=320)
    return tokenizer


def test_special_split_keeps_special_tokens():
    """
    这个测试不是测你的 tokenizer，而是帮助确认：
    re.split + 捕获组确实可以把 special token 单独切出来，
    同时保留普通文本。
    """
    text = "hello <|endoftext|> world <|pad|>!"
    special = {
        "<|endoftext|>": 320,
        "<|pad|>": 321,
    }

    special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
    chunks = re.split(special_pattern, text)

    assert chunks == [
        "hello ",
        "<|endoftext|>",
        " world ",
        "<|pad|>",
        "!",
    ]


def test_register_special_tokens():
    tokenizer = build_tokenizer()

    special_tokens = {
        "<|endoftext|>": 320,
        "<|pad|>": 321,
    }

    tokenizer.register_special_tokens(special_tokens)

    assert tokenizer.special_tokens == special_tokens
    assert tokenizer.inverse_special_tokens == {
        320: "<|endoftext|>",
        321: "<|pad|>",
    }

    # 如果你采用 base.decode + vocab 的方案，那么 special token 也应该进入 vocab
    assert tokenizer.vocab[320] == b"<|endoftext|>"
    assert tokenizer.vocab[321] == b"<|pad|>"


def test_encode_allowed_special_all():
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
        "<|pad|>": 321,
    })

    text = "hello <|endoftext|> world <|pad|>!"

    ids = tokenizer.encode(text, allowed_special="all")

    assert 320 in ids
    assert 321 in ids
    assert tokenizer.decode(ids) == text


def test_encode_allowed_special_set():
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
        "<|pad|>": 321,
    })

    text = "hello <|endoftext|> world"

    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    assert 320 in ids
    assert tokenizer.decode(ids) == text


def test_encode_none_raise_raises_on_special_token():
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
    })

    text = "hello <|endoftext|> world"

    with pytest.raises(ValueError):
        tokenizer.encode(text, allowed_special="none_raise")


def test_encode_default_is_none_raise():
    """
    如果你把 allowed_special 的默认值设为 'none_raise'，
    那么 encode(text) 遇到 special token 时也应该报错。
    """
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
    })

    text = "hello <|endoftext|> world"

    with pytest.raises(ValueError):
        tokenizer.encode(text)


def test_encode_none_treats_special_as_ordinary_text():
    """
    allowed_special='none' 表示：
    即使文本中出现 <|endoftext|>，
    也不要把它当 special token，而是当普通文本走 regex+BPE。
    """
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
    })

    text = "hello <|endoftext|> world"

    ids = tokenizer.encode(text, allowed_special="none")

    assert 320 not in ids
    assert tokenizer.decode(ids) == text


def test_encode_special_tokens_exact_sequence():
    """
    这个测试检查 encode 的核心分流逻辑：

    普通文本 -> encode_ordinary
    special token -> 直接 append special id
    普通文本 -> encode_ordinary
    """
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
        "<|pad|>": 321,
    })

    text = "hello <|endoftext|> world <|pad|>!"

    ids = tokenizer.encode(text, allowed_special="all")

    expected = []
    expected.extend(tokenizer.encode_ordinary("hello "))
    expected.append(320)
    expected.extend(tokenizer.encode_ordinary(" world "))
    expected.append(321)
    expected.extend(tokenizer.encode_ordinary("!"))

    assert ids == expected
    assert tokenizer.decode(ids) == text


def test_special_token_is_not_added_to_merges():
    """
    special token 不应该出现在 merges 中。
    它不是 BPE merge 学出来的，而是人工注册的 token。
    """
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
    })

    assert "<|endoftext|>" in tokenizer.special_tokens
    assert all("<|endoftext|>" not in pair for pair in tokenizer.merges)


def test_save_load_special_tokens(tmp_path):
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|endoftext|>": 320,
        "<|pad|>": 321,
    })

    prefix = str(tmp_path / "regex_tokenizer")
    tokenizer.save(prefix)

    tokenizer2 = RegexTokenizer()
    tokenizer2.load(prefix + ".model")

    text = "hello <|endoftext|> world <|pad|>!"

    ids = tokenizer2.encode(text, allowed_special="all")

    assert tokenizer2.special_tokens == {
        "<|endoftext|>": 320,
        "<|pad|>": 321,
    }
    assert tokenizer2.decode(ids) == text


def test_longer_special_token_should_match_first():
    """
    如果 special token 有前缀重叠，长的应该优先匹配。

    例如：
    <|end|>
    <|end|>oftext

    如果不按长度从大到小排序，短的可能先匹配，导致长 special token 被切坏。
    """
    tokenizer = build_tokenizer()
    tokenizer.register_special_tokens({
        "<|end|>": 320,
        "<|end|>oftext": 321,
    })

    text = "hello <|end|>oftext world"

    ids = tokenizer.encode(text, allowed_special="all")

    assert 321 in ids
    assert 320 not in ids
    assert tokenizer.decode(ids) == text