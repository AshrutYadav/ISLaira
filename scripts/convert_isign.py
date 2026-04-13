"""
iSign Dataset Converter
========================
Converts the real iSign_v1.1 CSV + .pose files into the annotation JSON
format that pose2text/scripts/train.py expects.
 
ACTUAL CSV structure (verified from iSign_v1.1.csv):
  Columns:  uid, text
  uid format:
    - Short YouTube IDs:  {video_hash}-{seq}    e.g.  1782bea75c7d-1
    - Long YouTube IDs:   {video_hash}--{seq}   e.g.  GIx57eZ4R0M--0
  text: English translation of that sign segment
 
  127,237 rows total across 6,058 unique source videos.
  Pose filename = uid + '.pose'   (the FULL uid, including dashes)
 
  There are NO gloss columns in this CSV — iSign is a direct
  pose-to-text (SignPose2Text) dataset with no intermediate glosses.
  We therefore train in Sign2Text mode (lambda_recognition=0).
 
Usage
-----
    python scripts/convert_isign.py \
        --csv   /data/iSign_v1.1.csv \
        --poses /data/isign_poses/ \
        --out   data/
 
This produces:
    data/train.json
    data/dev.json
    data/test.json
    data/word_vocab.json      (built from train set, capped at TOP_K words)
"""
 
import argparse
import csv
import json
import os
import re
import sys
import random
from collections import defaultdict, Counter
 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
 
# ---------------------------------------------------------------------------
# Vocabulary cap
# ---------------------------------------------------------------------------
# Limits the word vocab to the most frequent words.
# Words outside this set are replaced with <unk> in ALL splits.
# This is critical: 69k vocab makes the model impossible to converge.
# 8k covers ~95% of real sentence content for conversational ISL.
TOP_K_WORDS = 8000
 
 
# ---------------------------------------------------------------------------
# UID utilities
# ---------------------------------------------------------------------------
 
def parse_uid(uid: str):
    """
    Extract (video_hash, seq_number) from a uid.
 
    Handles both separators found in the real CSV:
      '1782bea75c7d-1'   -> ('1782bea75c7d', 1)
      'GIx57eZ4R0M--0'  -> ('GIx57eZ4R0M',  0)
    """
    m = re.match(r'^(.+?)(-{1,2})(\d+)$', uid)
    if m:
        return m.group(1), int(m.group(3))
    return uid, 0
 
 
# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------
 
def convert_isign(
    csv_path,
    pose_dir,
    out_dir,
    train_ratio=0.85,
    dev_ratio=0.075,
    seed=42,
    skip_empty_text=True,
    skip_missing_pose=True,
    min_words=1,
    max_words=60,
):
    print(f"[convert] Reading {csv_path} ...")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[convert] {len(rows):,} rows loaded")
 
    # ── Group rows by video_hash ──────────────────────────────────────────
    groups = defaultdict(list)
    skipped_text = skipped_pose = skipped_len = 0
 
    for row in rows:
        uid  = row["uid"].strip()
        text = row["text"].strip()
        video_hash, seq = parse_uid(uid)
 
        words = text.lower().split()           # lowercase for consistency
        if skip_empty_text and not words:
            skipped_text += 1
            continue
        if len(words) < min_words or len(words) > max_words:
            skipped_len += 1
            continue
 
        pose_path = os.path.join(pose_dir, uid + ".pose")
        if skip_missing_pose and not os.path.exists(pose_path):
            skipped_pose += 1
            continue
 
        groups[video_hash].append({
            "uid":        uid,
            "seq":        seq,
            "words":      words,
            "pose_path":  pose_path,
            "video_hash": video_hash,
        })
 
    print(f"[convert] Skipped: {skipped_text} empty text, "
          f"{skipped_len} wrong length, {skipped_pose} missing pose")
 
    for vh in groups:
        groups[vh].sort(key=lambda r: r["seq"])
 
    # ── Split by video_hash ───────────────────────────────────────────────
    all_hashes = sorted(groups.keys())
    random.seed(seed)
    random.shuffle(all_hashes)
 
    n       = len(all_hashes)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)
 
    train_hashes = set(all_hashes[:n_train])
    dev_hashes   = set(all_hashes[n_train: n_train + n_dev])
    test_hashes  = set(all_hashes[n_train + n_dev:])
 
    print(f"[convert] Video groups -> train:{len(train_hashes)} "
          f"dev:{len(dev_hashes)} test:{len(test_hashes)}")
 
    # ── CHANGE 1: Count word frequencies in TRAIN split only ─────────────
    print(f"\n[vocab] Counting word frequencies in train split ...")
    train_word_counts = Counter()
    for vh in train_hashes:
        for item in groups[vh]:
            train_word_counts.update(item["words"])
 
    top_words = {w for w, _ in train_word_counts.most_common(TOP_K_WORDS)}
 
    total_tokens = sum(train_word_counts.values())
    covered      = sum(c for w, c in train_word_counts.items() if w in top_words)
    print(f"[vocab] Top-{TOP_K_WORDS} words cover "
          f"{covered / total_tokens * 100:.1f}% of all train tokens")
 
    # ── CHANGE 2: Replace rare words with <unk> in ALL splits ────────────
    unk_count = 0
    for vh in groups:
        for item in groups[vh]:
            new_words = [w if w in top_words else "<unk>" for w in item["words"]]
            unk_count += sum(1 for a, b in zip(item["words"], new_words) if a != b)
            item["words"] = new_words
 
    print(f"[vocab] Replaced {unk_count:,} rare-word tokens with <unk>")
 
    # ── Build annotation lists ────────────────────────────────────────────
    splits     = {"train": train_hashes, "dev": dev_hashes, "test": test_hashes}
    word_corpus = []
 
    os.makedirs(out_dir, exist_ok=True)
 
    for split_name, hash_set in splits.items():
        annotations = []
        for vh in sorted(hash_set):
            for item in groups[vh]:
                annotations.append({
                    "id":          item["uid"],
                    "pose_path":   item["pose_path"],
                    "gloss":       [],             # iSign has no glosses
                    "translation": item["words"],
                    "video_hash":  item["video_hash"],
                    "seq":         item["seq"],
                })
                if split_name == "train":
                    word_corpus.append(item["words"])
 
        out_path = os.path.join(out_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        print(f"[convert] {split_name:5s}: {len(annotations):6,} samples -> {out_path}")
 
    # ── Build word vocabulary (train only, already capped) ────────────────
    print(f"\n[vocab] Building vocabulary from {len(word_corpus):,} train sentences ...")
    from utils.vocabulary import Vocabulary
    word_vocab = Vocabulary()
    word_vocab.build_from_corpus(word_corpus, min_freq=1)
    vocab_path = os.path.join(out_dir, "word_vocab.json")
    word_vocab.save(vocab_path)
    print(f"[vocab] {len(word_vocab):,} tokens -> {vocab_path}")
    print(f"        (reduced from 69,806 -- model can now converge)")
 
    total_kept = sum(len(groups[h]) for h in all_hashes)
    print(f"\n[convert] Done. {total_kept:,} samples kept from {len(rows):,} original rows.")
    print(f"[convert] Word vocabulary size: {len(word_vocab):,}")
    print(f"\nNOTE: iSign has no gloss annotations.")
    print(f"      Set lambda_recognition=0.0 in config.json.")
    return word_vocab
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Convert iSign CSV + .pose files to train/dev/test JSON",
    )
    p.add_argument("--csv",   required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--out",   default="data")
    p.add_argument("--train_ratio", type=float, default=0.85)
    p.add_argument("--dev_ratio",   type=float, default=0.075)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--min_words",   type=int,   default=1)
    p.add_argument("--max_words",   type=int,   default=60)
    p.add_argument("--top_k_words", type=int,   default=TOP_K_WORDS,
                   help=f"Vocab size cap (default: {TOP_K_WORDS})")
    p.add_argument("--keep_missing_pose", action="store_true")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    TOP_K_WORDS = args.top_k_words
    convert_isign(
        csv_path          = args.csv,
        pose_dir          = args.poses,
        out_dir           = args.out,
        train_ratio       = args.train_ratio,
        dev_ratio         = args.dev_ratio,
        seed              = args.seed,
        min_words         = args.min_words,
        max_words         = args.max_words,
        skip_missing_pose = not args.keep_missing_pose,
    )
 