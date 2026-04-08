import json
import re
from collections import Counter


class SimpleTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, vocab=None, lower=True):
        if vocab is None:
            vocab = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.BOS_TOKEN: 2,
                self.EOS_TOKEN: 3,
            }
        self.vocab = {token: int(index) for token, index in vocab.items()}
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}
        self.lower = lower

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_id(self):
        return self.vocab[self.UNK_TOKEN]

    @staticmethod
    def normalize_text(text, lower=True):
        normalized = " ".join(str(text).strip().split())
        return normalized.lower() if lower else normalized

    @classmethod
    def tokenize_text(cls, text, lower=True):
        normalized = cls.normalize_text(text, lower=lower)
        if not normalized:
            return []
        return re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)

    @classmethod
    def build(cls, texts, min_freq=1, max_vocab_size=None, lower=True):
        counter = Counter()
        for text in texts:
            counter.update(cls.tokenize_text(text, lower=lower))

        tokenizer = cls(lower=lower)
        for token, frequency in counter.most_common():
            if frequency < min_freq or token in tokenizer.vocab:
                continue
            if max_vocab_size is not None and len(tokenizer.vocab) >= max_vocab_size:
                break
            tokenizer.vocab[token] = len(tokenizer.vocab)

        tokenizer.inverse_vocab = {index: token for token, index in tokenizer.vocab.items()}
        return tokenizer

    def encode(self, text, max_length, add_special_tokens=True):
        tokens = self.tokenize_text(text, lower=self.lower)
        if add_special_tokens:
            tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]

        token_ids = [self.vocab.get(token, self.unk_id) for token in tokens][:max_length]
        attention_mask = [1] * len(token_ids)

        if len(token_ids) < max_length:
            padding_size = max_length - len(token_ids)
            token_ids.extend([self.pad_id] * padding_size)
            attention_mask.extend([0] * padding_size)

        return token_ids, attention_mask

    def decode(self, token_ids, skip_special_tokens=True):
        special_tokens = {self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(int(token_id), self.UNK_TOKEN)
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens).strip()

    def to_dict(self):
        return {
            "vocab": self.vocab,
            "lower": self.lower,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(vocab=payload["vocab"], lower=payload.get("lower", True))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
