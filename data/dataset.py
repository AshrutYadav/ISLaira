"""
Dataset utilities for Sign Language Transformer.
 
Supports:
  - pose_format (.pose files from pose_format library)
  - numpy .npy files (pre-extracted keypoints)
  - mediapipe output (dict with landmark arrays)
  - CSV/JSON annotations
 
Keypoint format (PHOENIX14T-compatible):
  - Mediapipe Holistic: 543 landmarks × 3 coords = 1629 dims
    (or a subset: face=468, left_hand=21, right_hand=21, pose=33)
"""
 
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Optional, Tuple, Union
 
from utils.vocabulary import Vocabulary
 
 
# ---------------------------------------------------------------------------
# Pose normalisation helpers
# ---------------------------------------------------------------------------
 
def normalise_pose(keypoints: np.ndarray, strategy: str = "z-score") -> np.ndarray:
    if strategy == "none":
        return keypoints.astype(np.float32)
    elif strategy == "z-score":
        mean = keypoints.mean(axis=0, keepdims=True)
        std  = keypoints.std(axis=0, keepdims=True) + 1e-8
        return ((keypoints - mean) / std).astype(np.float32)
    elif strategy == "minmax":
        lo = keypoints.min(axis=0, keepdims=True)
        hi = keypoints.max(axis=0, keepdims=True)
        return ((keypoints - lo) / (hi - lo + 1e-8)).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalisation strategy: {strategy}")
 
 
def load_pose_file(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
    elif ext == ".npz":
        d = np.load(path)
        arr = d["data"].astype(np.float32)
    elif ext == ".pose":
        try:
            from pose_format import Pose
            with open(path, "rb") as f:
                pose = Pose.read(f.read())
            data = pose.body.data[:, 0, :, :]
            arr = data.reshape(data.shape[0], -1).filled(0).astype(np.float32)
        except ImportError:
            raise ImportError(
                "pose_format is not installed. "
                "Install with: pip install pose_format"
            )
    else:
        raise ValueError(f"Unsupported pose file format: {ext}")
 
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    assert arr.ndim == 2, f"Expected 2D array (T, C), got shape {arr.shape}"
    return arr
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class SignPoseDataset(Dataset):
    """
    Dataset for pose-to-text sign language translation.
 
    gloss_vocab may be None for gloss-free datasets like iSign.
    In that case gloss_ids will be an empty tensor and gloss_len = 0.
    """
 
    def __init__(
        self,
        annotations: Union[str, List[Dict]],
        gloss_vocab: Optional[Vocabulary],   # None is fine for iSign
        word_vocab: Vocabulary,
        normalise: str = "z-score",
        max_pose_len: Optional[int] = None,
        max_text_len: Optional[int] = None,
    ):
        if isinstance(annotations, str):
            with open(annotations, "r", encoding="utf-8") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = annotations
 
        self.gloss_vocab  = gloss_vocab   # may be None
        self.word_vocab   = word_vocab
        self.normalise    = normalise
        self.max_pose_len = max_pose_len
        self.max_text_len = max_text_len
 
    def __len__(self) -> int:
        return len(self.annotations)
 
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
 
        # ── Pose ──────────────────────────────────────────────────────────
        pose = load_pose_file(ann["pose_path"])
        pose = normalise_pose(pose, self.normalise)
        if self.max_pose_len and pose.shape[0] > self.max_pose_len:
            pose = pose[: self.max_pose_len]
        pose_len = pose.shape[0]
 
        # ── Gloss (skipped when gloss_vocab is None) ───────────────────────
        if self.gloss_vocab is not None:
            gloss_ids = self.gloss_vocab.encode(ann.get("gloss", []))
            gloss_len = len(gloss_ids)
        else:
            gloss_ids = []   # empty — CTC loss disabled
            gloss_len = 0
 
        # ── Translation ───────────────────────────────────────────────────
        trans_ids = self.word_vocab.encode(
            ann["translation"],
            add_bos=True,
            add_eos=True,
        )
        if self.max_text_len and len(trans_ids) > self.max_text_len + 2:
            trans_ids = trans_ids[: self.max_text_len + 2]
        trans_len = len(trans_ids)
 
        return {
            "id":        ann.get("id", str(idx)),
            "pose":      torch.tensor(pose, dtype=torch.float32),
            "pose_len":  torch.tensor(pose_len, dtype=torch.long),
            "gloss_ids": torch.tensor(gloss_ids, dtype=torch.long),
            "gloss_len": torch.tensor(gloss_len, dtype=torch.long),
            "trans_ids": torch.tensor(trans_ids, dtype=torch.long),
            "trans_len": torch.tensor(trans_len, dtype=torch.long),
        }
 
 
# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------
 
def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    ids        = [item["id"] for item in batch]
    poses      = [item["pose"] for item in batch]
    pose_lens  = torch.stack([item["pose_len"]  for item in batch])
    gloss_lens = torch.stack([item["gloss_len"] for item in batch])
    trans_ids  = [item["trans_ids"] for item in batch]
    trans_lens = torch.stack([item["trans_len"] for item in batch])
 
    poses_padded = pad_sequence(poses,     batch_first=True, padding_value=0.0)
    trans_padded = pad_sequence(trans_ids, batch_first=True, padding_value=0)
 
    # Gloss: only pad if tokens exist; otherwise return a (B,1) zero placeholder
    gloss_seqs = [item["gloss_ids"] for item in batch]
    if any(g.numel() > 0 for g in gloss_seqs):
        gloss_padded = pad_sequence(gloss_seqs, batch_first=True, padding_value=0)
    else:
        gloss_padded = torch.zeros(len(batch), 1, dtype=torch.long)
 
    return {
        "ids":        ids,
        "pose":       poses_padded,
        "pose_lens":  pose_lens,
        "gloss_ids":  gloss_padded,
        "gloss_lens": gloss_lens,
        "trans_ids":  trans_padded,
        "trans_lens": trans_lens,
    }
 
 
# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
 
def build_dataloader(
    annotations: Union[str, List[Dict]],
    gloss_vocab: Optional[Vocabulary],   # None is fine for iSign
    word_vocab: Vocabulary,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    normalise: str = "z-score",
    max_pose_len: Optional[int] = None,
    max_text_len: Optional[int] = None,
) -> DataLoader:
    dataset = SignPoseDataset(
        annotations=annotations,
        gloss_vocab=gloss_vocab,
        word_vocab=word_vocab,
        normalise=normalise,
        max_pose_len=max_pose_len,
        max_text_len=max_text_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
 
 
# ---------------------------------------------------------------------------
# Synthetic data generator (for testing without real data)
# ---------------------------------------------------------------------------
 
def generate_dummy_annotations(
    n_samples: int = 100,
    pose_dir: str = "/tmp/pose2text_dummy",
    pose_dim: int = 1629,
    max_frames: int = 200,
    gloss_vocab_size: int = 50,
    word_vocab_size: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    os.makedirs(pose_dir, exist_ok=True)
 
    gloss_tokens = [f"GLOSS_{i}" for i in range(gloss_vocab_size)]
    word_tokens  = [f"word_{i}"  for i in range(word_vocab_size)]
 
    annotations = []
    all_glosses: List[str] = []
    all_words:   List[str] = []
 
    for i in range(n_samples):
        T = int(rng.integers(20, max_frames))
        pose = rng.standard_normal((T, pose_dim)).astype(np.float32)
        pose_path = os.path.join(pose_dir, f"sample_{i:04d}.npy")
        np.save(pose_path, pose)
 
        n_gloss = int(rng.integers(2, 10))
        n_words = int(rng.integers(3, 12))
        gloss       = rng.choice(gloss_tokens, n_gloss).tolist()
        translation = rng.choice(word_tokens,  n_words).tolist()
 
        all_glosses.extend(gloss)
        all_words.extend(translation)
 
        annotations.append({
            "id":          f"sample_{i:04d}",
            "pose_path":   pose_path,
            "gloss":       gloss,
            "translation": translation,
        })
 
    return annotations, all_glosses, all_words
 