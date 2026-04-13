"""
Quick inference script for Sign Language Transformer.
 
Usage:
    # With old 69k vocab checkpoint:
    python scripts/predict.py \
        --checkpoint checkpoints_old_69k/checkpoint_best.pt \
        --vocab      data_old/word_vocab_69k.json \
        --pose       data/isign_poses/1782bea75c7d-1.pose
 
    # With new 8k vocab checkpoint (after retraining):
    python scripts/predict.py \
        --checkpoint checkpoints/checkpoint_best.pt \
        --vocab      data/word_vocab.json \
        --pose       data/isign_poses/1782bea75c7d-1.pose
"""
 
import argparse
import sys
import os
 
import torch
 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.sign_language_transformer import SignLanguageTransformer
from utils.vocabulary import Vocabulary
 
 
# ---------------------------------------------------------------------------
# Pose loader
# ---------------------------------------------------------------------------
 
def load_pose(pose_path: str, device: torch.device) -> tuple:
    """
    Load a .pose file and return (pose_tensor, pose_length).
    Returns:
        pose:   (1, T, pose_dim) float tensor on device
        length: (1,) int tensor
    """
    try:
        from pose_format import Pose
        with open(pose_path, "rb") as f:
            pose_obj = Pose.read(f.read())
        data = pose_obj.body.data[:, 0, :, :].filled(0)   # (T, keypoints, dims)
        T = data.shape[0]
        flat = data.reshape(T, -1).astype("float32")       # (T, pose_dim)
    except Exception as e:
        print(f"[predict] ERROR loading pose: {e}")
        sys.exit(1)
 
    pose_tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, D)
    length      = torch.tensor([T], dtype=torch.long).to(device)                   # (1,)
    return pose_tensor, length
 
 
# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
 
def load_model(checkpoint_path: str, word_vocab: Vocabulary, device: torch.device):
    """Load model from checkpoint."""
    print(f"[predict] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
 
    cfg = state.get("model_config", {})
    if not cfg:
        print("[predict] WARNING: no model_config in checkpoint, using defaults")
 
    # Build model with saved config
    model = SignLanguageTransformer(
        pose_input_dim     = cfg.get("pose_input_dim", 1728),
        gloss_vocab_size   = cfg.get("gloss_vocab_size", 1),
        word_vocab_size    = cfg.get("word_vocab_size", len(word_vocab)),
        d_model            = cfg.get("d_model", 512),
        nhead              = cfg.get("nhead", 8),
        num_encoder_layers = cfg.get("num_encoder_layers", 3),
        num_decoder_layers = cfg.get("num_decoder_layers", 3),
        dim_feedforward    = cfg.get("dim_feedforward", 2048),
        dropout            = 0.0,          # disable dropout at inference
        max_seq_len        = cfg.get("max_seq_len", 4096),
        pad_idx            = word_vocab.pad_idx,
        bos_idx            = word_vocab.bos_idx,
        eos_idx            = word_vocab.eos_idx,
        blank_idx          = cfg.get("blank_idx", 0),
        lambda_recognition = 0.0,
        lambda_translation = 1.0,
    ).to(device)
 
    model.load_state_dict(state["model_state_dict"])
    model.eval()
 
    trained_epoch = state.get("epoch", "?")
    dev_loss      = state.get("metrics", {}).get("dev_loss", "?")
    print(f"[predict] Model loaded  (trained epoch={trained_epoch}  dev_loss={dev_loss})")
    return model
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def predict(checkpoint_path, vocab_path, pose_path, beam_size=5, max_len=100, cpu=False):
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    print(f"[predict] device = {device}")
 
    # Load vocab
    word_vocab = Vocabulary.load(vocab_path)
    print(f"[predict] Vocab size: {len(word_vocab)}")
 
    # Load model
    model = load_model(checkpoint_path, word_vocab, device)
 
    # Load pose
    print(f"[predict] Loading pose: {pose_path}")
    pose, length = load_pose(pose_path, device)
    print(f"[predict] Pose shape: {tuple(pose.shape)}  (frames={length.item()})")
 
    # Generate translation
    print(f"[predict] Generating translation (beam_size={beam_size}) ...")
    with torch.no_grad():
        pred_ids = model.generate(
            pose, length,
            beam_size=beam_size,
            max_len=max_len,
        )
 
    # Decode tokens to text
    translation = " ".join(word_vocab.decode(pred_ids[0], skip_special=True))
 
    print(f"\n{'─'*50}")
    print(f"  Pose file : {os.path.basename(pose_path)}")
    print(f"  Translation: {translation if translation else '(empty — model needs more training)'}")
    print(f"{'─'*50}\n")
 
    return translation
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args():
    p = argparse.ArgumentParser(description="Run inference with Sign Language Transformer")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--vocab",      required=True,
                   help="Path to word_vocab.json matching the checkpoint")
    p.add_argument("--pose",       required=True,
                   help="Path to a .pose file to translate")
    p.add_argument("--beam_size",  type=int,   default=5)
    p.add_argument("--max_len",    type=int,   default=100)
    p.add_argument("--cpu",        action="store_true",
                   help="Force CPU inference even if CUDA is available")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    predict(
        checkpoint_path = args.checkpoint,
        vocab_path      = args.vocab,
        pose_path       = args.pose,
        beam_size       = args.beam_size,
        max_len         = args.max_len,
        cpu             = args.cpu,
    )
