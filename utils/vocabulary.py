"""
Vocabulary management for Sign Language Transformer.

Handles:
  - gloss vocabulary (sign glosses)
  - word vocabulary (target spoken language — English / German)

Compatible with PHOENIX14T and custom datasets.
"""

import json
import os
from typing import Dict, List, Optional


# Special tokens — must match model pad_idx / bos_idx / eos_idx
PAD_TOKEN = "<pad>"   # idx 0
BOS_TOKEN = "<bos>"   # idx 1
EOS_TOKEN = "<eos>"   # idx 2
UNK_TOKEN = "<unk>"   # idx 3

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocabulary:
    """
    Simple token ↔ index vocabulary.

    Usage
    -----
    vocab = Vocabulary()
    vocab.build_from_list(["hello", "world"])
    idx = vocab["hello"]       # 4  (0-3 are specials)
    tok = vocab.idx2token[4]   # "hello"
    """

    def __init__(self):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._add_specials()

    # ── Special tokens ───────────────────────────────────────────────────

    def _add_specials(self):
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    # ── Building ─────────────────────────────────────────────────────────

    def build_from_list(self, tokens: List[str]):
        """Add tokens from an iterable (duplicates silently ignored)."""
        for tok in tokens:
            self._add(tok)

    def build_from_corpus(self, sentences: List[List[str]], min_freq: int = 1):
        """
        Build from tokenised sentences with optional frequency filtering.

        Args:
            sentences: list of token lists
            min_freq:  discard tokens appearing fewer than min_freq times
        """
        from collections import Counter
        counts: Counter = Counter()
        for sent in sentences:
            counts.update(sent)
        tokens = [tok for tok, cnt in counts.most_common() if cnt >= min_freq]
        self.build_from_list(tokens)

    # ── Lookup ───────────────────────────────────────────────────────────

    def __getitem__(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[UNK_TOKEN])

    def __len__(self) -> int:
        return len(self.token2idx)

    def encode(self, tokens: List[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self[t] for t in tokens]
        if add_bos:
            ids = [self.token2idx[BOS_TOKEN]] + ids
        if add_eos:
            ids = ids + [self.token2idx[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        specials = set(self.token2idx[t] for t in SPECIAL_TOKENS)
        tokens = []
        for idx in ids:
            tok = self.idx2token.get(idx, UNK_TOKEN)
            if skip_special and idx in specials:
                continue
            tokens.append(tok)
        return tokens

    def decode_sentence(self, ids: List[int]) -> str:
        return " ".join(self.decode(ids, skip_special=True))

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls.__new__(cls)
        vocab.token2idx = {}
        vocab.idx2token = {}
        with open(path, "r", encoding="utf-8") as f:
            vocab.token2idx = json.load(f)
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        return vocab

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[UNK_TOKEN]
