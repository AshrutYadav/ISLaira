"""
Inference script: translate a pose file to English text.

Usage
-----
    # Translate a single .npy pose file:
    python infer.py --model exports/model.pt --pose path/to/pose.npy

    # Translate with beam search (beam_size=5, length_penalty=1.0):
    python infer.py --model exports/model.pt --pose path/to/pose.npy \\
                    --beam_size 5 --length_penalty 1.0

    # Batch translate from a JSON annotations file:
    python infer.py --model exports/model.pt --annotations test.json \\
                    --output predictions.json

    # ONNX Runtime inference (encoder only):
    python infer.py --onnx exports/model_encoder.onnx \\
                    --model exports/model.pt \\
                    --pose path/to/pose.npy
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.dataset import load_pose_file, normalise_pose
from utils.exporter import load_exported_model
from utils.metrics import compute_bleu, compute_wer


# ---------------------------------------------------------------------------
# Single-file inference
# ---------------------------------------------------------------------------

def translate_pose_file(
    pose_path: str,
    model,
    word_vocab,
    device: torch.device,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    normalise: str = "z-score",
) -> str:
    """
    Translate a single pose file to a spoken-language sentence.

    Args:
        pose_path:      path to .npy / .npz / .pose file
        model:          loaded SignLanguageTransformer
        word_vocab:     target language Vocabulary
        device:         inference device
        beam_size:      beam width (1 = greedy)
        length_penalty: BLEU length-penalty alpha
        normalise:      pose normalisation strategy

    Returns:
        Translated sentence string.
    """
    pose = load_pose_file(pose_path)                          # (T, C)
    pose = normalise_pose(pose, normalise)                    # normalised
    pose_t = torch.tensor(pose, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, C)
    lens_t = torch.tensor([pose.shape[0]], dtype=torch.long, device=device)

    token_ids = model.generate(
        pose_t, lens_t,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )[0]

    return word_vocab.decode_sentence(token_ids)


# ---------------------------------------------------------------------------
# ONNX Runtime encoder (optional fast encoder path)
# ---------------------------------------------------------------------------

def onnx_encode(onnx_path: str, pose: np.ndarray) -> np.ndarray:
    """
    Run the exported ONNX encoder.

    Returns:
        encoder_output: (1, T, d_model) float32 numpy array
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Install onnxruntime:  pip install onnxruntime")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    pose_np = pose[np.newaxis].astype(np.float32)           # (1, T, C)
    lens_np = np.array([pose.shape[0]], dtype=np.int64)

    outputs = sess.run(None, {"pose": pose_np, "pose_lengths": lens_np})
    encoder_output, gloss_log_probs = outputs
    return encoder_output  # (1, T, d_model)


# ---------------------------------------------------------------------------
# Batch inference + BLEU evaluation
# ---------------------------------------------------------------------------

def batch_translate(
    annotations_path: str,
    model,
    word_vocab,
    device: torch.device,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    normalise: str = "z-score",
    output_path: str = None,
) -> dict:
    """
    Translate all samples in an annotations JSON file and compute BLEU.

    Returns:
        dict with 'bleu1', 'bleu4', 'predictions'
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    hypotheses = []
    references = []
    predictions = []

    print(f"[infer] Translating {len(annotations)} samples …")
    t0 = time.time()

    for ann in annotations:
        hyp = translate_pose_file(
            ann["pose_path"], model, word_vocab, device,
            beam_size=beam_size, length_penalty=length_penalty, normalise=normalise,
        )
        ref_tokens = ann.get("translation", [])
        ref_str = " ".join(ref_tokens)

        hypotheses.append(hyp.split())
        references.append([ref_str.split()])
        predictions.append({
            "id": ann.get("id", "?"),
            "hypothesis": hyp,
            "reference": ref_str,
        })

    bleu = compute_bleu(hypotheses, references)
    elapsed = time.time() - t0

    result = {
        "bleu1": bleu["bleu1"],
        "bleu4": bleu["bleu4"],
        "n_samples": len(annotations),
        "elapsed_s": round(elapsed, 2),
        "predictions": predictions,
    }

    print(f"[infer] BLEU-1={bleu['bleu1']:.2f}  BLEU-4={bleu['bleu4']:.2f}")
    print(f"[infer] {len(annotations)} samples in {elapsed:.1f}s  ({len(annotations)/elapsed:.1f} samples/s)")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[infer] Predictions saved → {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sign Language Transformer — Inference")
    p.add_argument("--model", required=True, help="Path to exported model.pt")
    p.add_argument("--pose", default=None, help="Path to a single .npy pose file")
    p.add_argument("--annotations", default=None, help="JSON annotations file for batch eval")
    p.add_argument("--output", default=None, help="Output JSON path (batch mode)")
    p.add_argument("--onnx", default=None, help="Optional ONNX encoder path")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--normalise", default="z-score", choices=["z-score", "minmax", "none"])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[infer] device = {device}")

    # Load model
    print(f"[infer] Loading model from {args.model} …")
    model, word_vocab, gloss_vocab = load_exported_model(args.model, device)
    model.eval()

    # Single file
    if args.pose:
        print(f"[infer] Input: {args.pose}")
        sentence = translate_pose_file(
            args.pose, model, word_vocab, device,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            normalise=args.normalise,
        )
        print(f"\n[result] → {sentence}\n")

    # Batch
    elif args.annotations:
        batch_translate(
            args.annotations, model, word_vocab, device,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            normalise=args.normalise,
            output_path=args.output,
        )

    else:
        print("Provide --pose (single file) or --annotations (batch eval).")
