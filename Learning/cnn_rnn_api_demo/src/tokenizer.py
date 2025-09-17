
# src/tokenizer.py
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, text: str):
        vocab = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = [None] * len(vocab)
        for ch, i in stoi.items():
            itos[i] = ch
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]):
        return ''.join(self.itos[i] for i in ids if 0 <= i < len(self.itos))
