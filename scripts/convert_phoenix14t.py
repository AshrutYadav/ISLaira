"""
PHOENIX14T Dataset Converter
=============================
Converts the RWTH-PHOENIX-Weather-2014T dataset into the annotation JSON
format expected by this project's DataLoader.

PHOENIX14T download:
  https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

Expected PHOENIX14T directory structure:
  phoenix14t/
    phoenix-2014-T.v3/
      PHOENIX-2014-T/
        features/fullFrame-210x260px/{train,dev,test}/
          {SIGNER}/{DATE}-{SESSION}/
            {ID}.png  (one frame per file)
        annotations/manual/
          train.corpus.csv
          dev.corpus.csv
          test.corpus.csv

CSV columns (pipe-separated):
  name | video | start | end | speaker | orth | translation

Usage
-----
    python scripts/convert_phoenix14t.py \\
        --phoenix_root /data/phoenix14t/ \\
        --pose_dir     /data/poses/ \\
        --output_dir   data/ \\
        --split        train

    # Then extract poses from the video frames directory
    # (use extract_pose.py on each video folder)

Notes
-----
  This script converts the CSV annotations to JSON.
  Pose extraction from frames must be done separately with extract_pose.py.
  If you already have .npy pose files, just provide the mapping in --pose_map_json.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


PHOENIX_CSV_DELIMITER = "|"

SPLIT_FILES = {
    "train": "train.corpus.csv",
    "dev":   "dev.corpus.csv",
    "test":  "test.corpus.csv",
}


def convert_phoenix14t(
    phoenix_root: str,
    pose_dir: str,
    output_dir: str,
    split: str = "train",
    pose_ext: str = ".npy",
    skip_missing: bool = True,
) -> str:
    """
    Convert PHOENIX14T CSV annotations to project JSON format.

    Args:
        phoenix_root: root of PHOENIX14T dataset
        pose_dir:     directory containing .npy pose files
                      (filename = {sample_id}.npy)
        output_dir:   where to write the output JSON file
        split:        'train' | 'dev' | 'test'
        pose_ext:     pose file extension ('.npy' | '.npz' | '.pose')
        skip_missing: skip samples whose pose file does not exist

    Returns:
        Path to the output JSON file.
    """
    annotations_dir = os.path.join(
        phoenix_root, "phoenix-2014-T.v3", "PHOENIX-2014-T", "annotations", "manual"
    )
    csv_path = os.path.join(annotations_dir, SPLIT_FILES[split])

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cannot find PHOENIX14T CSV at: {csv_path}\n"
            f"Check --phoenix_root and ensure the dataset is extracted."
        )

    annotations = []
    skipped = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=PHOENIX_CSV_DELIMITER)
        for row in reader:
            sample_id = row["name"].strip()
            gloss_str = row.get("orth", "").strip()
            trans_str = row.get("translation", "").strip()

            gloss_tokens = gloss_str.split() if gloss_str else []
            word_tokens  = trans_str.split()  if trans_str  else []

            pose_path = os.path.join(pose_dir, sample_id + pose_ext)

            if skip_missing and not os.path.exists(pose_path):
                skipped += 1
                continue

            annotations.append({
                "id":          sample_id,
                "pose_path":   pose_path,
                "gloss":       gloss_tokens,
                "translation": word_tokens,
                "speaker":     row.get("speaker", "").strip(),
                "video":       row.get("video", "").strip(),
            })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"[phoenix] {split}: {len(annotations)} samples → {out_path}")
    if skipped:
        print(f"[phoenix] Skipped {skipped} samples with missing pose files")

    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Convert PHOENIX14T to project JSON format")
    p.add_argument("--phoenix_root", required=True,
                   help="Root directory of PHOENIX14T dataset")
    p.add_argument("--pose_dir", required=True,
                   help="Directory containing .npy pose files")
    p.add_argument("--output_dir", default="data",
                   help="Output directory for JSON files")
    p.add_argument("--split", default="all", choices=["train", "dev", "test", "all"],
                   help="Which split(s) to convert")
    p.add_argument("--pose_ext", default=".npy",
                   help="Pose file extension (.npy | .npz | .pose)")
    p.add_argument("--keep_missing", action="store_true",
                   help="Include samples even if pose file is missing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        convert_phoenix14t(
            phoenix_root=args.phoenix_root,
            pose_dir=args.pose_dir,
            output_dir=args.output_dir,
            split=split,
            pose_ext=args.pose_ext,
            skip_missing=not args.keep_missing,
        )
