# src/tokenizer.py
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, text: str):
        if not text:
            raise ValueError("Build a vocabulary from nonempty training text")
        vocab = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = [None] * len(vocab)
        for ch, i in stoi.items():
            itos[i] = ch
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str):
        unknown = set(s) - self.stoi.keys()
        if unknown:
            raise ValueError(f"Characters outside the training vocabulary: {sorted(unknown)!r}")
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]):
        if any(i < 0 or i >= len(self.itos) for i in ids):
            raise ValueError("Token id outside the vocabulary")
        return "".join(self.itos[i] for i in ids)
