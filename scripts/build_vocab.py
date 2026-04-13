"""
Build gloss and word vocabularies from an annotations JSON file.

Run this once before training to produce the vocab files that
train.py and infer.py depend on.

Usage
-----
    python scripts/build_vocab.py \\
        --train data/train.json \\
        --gloss_out data/gloss_vocab.json \\
        --word_out  data/word_vocab.json \\
        --min_gloss_freq 1 \\
        --min_word_freq  1

    # PHOENIX14T example paths:
    python scripts/build_vocab.py \\
        --train phoenix14t/train.json \\
        --dev   phoenix14t/dev.json \\
        --gloss_out phoenix14t/gloss_vocab.json \\
        --word_out  phoenix14t/word_vocab.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.vocabulary import Vocabulary


def load_annotations(paths):
    annotations = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            annotations.extend(json.load(f))
    return annotations


def build_vocabs(args):
    paths = [args.train]
    if args.dev:
        paths.append(args.dev)

    print(f"[vocab] Loading annotations from: {paths}")
    annotations = load_annotations(paths)
    print(f"[vocab] {len(annotations)} samples loaded")

    gloss_corpus = [ann["gloss"] for ann in annotations]
    word_corpus  = [ann["translation"] for ann in annotations]

    # Count stats
    all_glosses = [g for sent in gloss_corpus for g in sent]
    all_words   = [w for sent in word_corpus  for w in sent]

    print(f"[vocab] Gloss tokens: {len(all_glosses):,} total, "
          f"{len(set(all_glosses)):,} unique")
    print(f"[vocab] Word tokens:  {len(all_words):,} total, "
          f"{len(set(all_words)):,} unique")

    gloss_vocab = Vocabulary()
    gloss_vocab.build_from_corpus(gloss_corpus, min_freq=args.min_gloss_freq)

    word_vocab = Vocabulary()
    word_vocab.build_from_corpus(word_corpus, min_freq=args.min_word_freq)

    print(f"[vocab] Gloss vocab size (with specials): {len(gloss_vocab)}")
    print(f"[vocab] Word vocab size  (with specials): {len(word_vocab)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.gloss_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.word_out)),  exist_ok=True)

    gloss_vocab.save(args.gloss_out)
    word_vocab.save(args.word_out)

    print(f"[vocab] Saved → {args.gloss_out}")
    print(f"[vocab] Saved → {args.word_out}")

    # Print a few samples
    print("\n[vocab] First 10 gloss tokens:")
    for tok, idx in list(gloss_vocab.token2idx.items())[:14]:
        print(f"  {idx:4d}  {tok}")

    print("\n[vocab] First 10 word tokens:")
    for tok, idx in list(word_vocab.token2idx.items())[:14]:
        print(f"  {idx:4d}  {tok}")


def parse_args():
    p = argparse.ArgumentParser(description="Build sign language vocabularies")
    p.add_argument("--train",          required=True, help="Train annotations JSON")
    p.add_argument("--dev",            default=None,  help="Dev annotations JSON (optional)")
    p.add_argument("--gloss_out",      default="data/gloss_vocab.json")
    p.add_argument("--word_out",       default="data/word_vocab.json")
    p.add_argument("--min_gloss_freq", type=int, default=1)
    p.add_argument("--min_word_freq",  type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    build_vocabs(parse_args())
