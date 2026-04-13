"""
Model export utilities for Sign Language Transformer.

Supports three export formats for cross-device / cross-platform deployment:

  1. Portable .pt checkpoint
     - Saves model weights + config + vocabularies in a single file.
     - Fully self-contained: re-instantiate the model anywhere with
       load_exported_model().

  2. TorchScript (.torchscript.pt)
     - Converts the model to a static graph via torch.jit.trace.
     - Runs on any PyTorch installation without the source code.
     - Suitable for C++ / Android / iOS / server inference.

  3. ONNX (.onnx)
     - Open Neural Network Exchange format.
     - Runs with ONNX Runtime on CPU/GPU/NPU, TensorRT, OpenVINO, etc.
     - Truly framework-agnostic deployment.

Usage
-----
    from utils.exporter import export_model, load_exported_model

    # After training:
    export_model(model, word_vocab, gloss_vocab, cfg, export_dir="exports/")

    # On another machine / process:
    model, word_vocab, gloss_vocab = load_exported_model("exports/model.pt")
"""

import os
import json
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Lazy imports for optional dependencies
# (onnx / onnxruntime are optional — TorchScript always works)


# ---------------------------------------------------------------------------
# Portable checkpoint export / load
# ---------------------------------------------------------------------------

def export_portable_checkpoint(
    model: nn.Module,
    word_vocab,
    gloss_vocab,
    cfg: dict,
    path: str,
):
    """
    Save everything needed to reinstantiate the model in one .pt file.

    Contents:
      - model weights (state_dict)
      - model constructor config (get_config())
      - word vocabulary (token2idx mapping)
      - gloss vocabulary (token2idx mapping)
      - training config
    """
    bundle = {
        "model_state_dict": model.state_dict(),
        "model_config":     model.get_config(),
        "word_vocab":       word_vocab.token2idx,
        "gloss_vocab": gloss_vocab.token2idx if gloss_vocab is not None else {},
        "training_cfg":     cfg,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(bundle, path)
    print(f"[export] Portable checkpoint → {path}")


def load_exported_model(path: str, device: Optional[torch.device] = None):
    """
    Load a portable checkpoint exported with export_portable_checkpoint.

    Returns:
        model, word_vocab, gloss_vocab
    """
    import sys, os
    # Make project modules importable regardless of working directory
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    from models.sign_language_transformer import SignLanguageTransformer
    from utils.vocabulary import Vocabulary

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torch.load(path, map_location=device, weights_only=False)

    # Rebuild vocabularies
    word_vocab = Vocabulary()
    gloss_vocab = None
    if bundle.get("gloss_vocab"):
        gloss_vocab = Vocabulary()
        gloss_vocab.token2idx = bundle["gloss_vocab"]
        gloss_vocab.idx2token = {int(v): k for k, v in bundle["gloss_vocab"].items()}

    gloss_vocab = Vocabulary()
    gloss_vocab.token2idx = bundle["gloss_vocab"]
    gloss_vocab.idx2token = {int(v): k for k, v in bundle["gloss_vocab"].items()}

    # Rebuild model
    cfg = bundle["model_config"]
    model = SignLanguageTransformer(**cfg).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    return model, word_vocab, gloss_vocab


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------

def export_torchscript(
    model: nn.Module,
    pose_dim: int,
    path: str,
    device: torch.device,
    example_T: int = 50,
):
    """
    Export model to TorchScript via tracing.

    TorchScript models are self-contained and can be loaded with:
        model = torch.jit.load("model.torchscript.pt")

    Notes:
      - Tracing captures a single execution path; dynamic control-flow
        (e.g. beam search) is NOT fully captured. The encoder is traced;
        beam-search-based generation should be called via the Python API
        or re-implemented in C++.
      - We trace the encode() method which is the performance-critical
        part for deployment pipelines.
    """
    model.eval().to(device)

    # Trace example input
    B = 1
    dummy_pose = torch.randn(B, example_T, pose_dim, device=device)
    dummy_lens = torch.tensor([example_T], device=device)

    with torch.no_grad():
        traced = torch.jit.trace(
            model.encode,
            (dummy_pose, dummy_lens),
            strict=False,
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    traced.save(path)
    print(f"[export] TorchScript (encode) → {path}")
    return traced


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    pose_dim: int,
    path: str,
    device: torch.device,
    example_T: int = 50,
    opset: int = 17,
):
    """
    Export the encoder to ONNX format.

    The ONNX graph covers: PoseSpatialEmbedding → PositionalEncoding → SLRT encoder.
    It accepts dynamic batch size and sequence length.

    Load with ONNX Runtime:
        import onnxruntime as ort
        sess = ort.InferenceSession("model_encoder.onnx")
        z, gloss_logp = sess.run(None, {
            "pose": pose_np,          # (B, T, pose_dim) float32
            "pose_lengths": lens_np,  # (B,) int64
        })
    """
    try:
        import onnx
    except ImportError:
        print("[export] ONNX skipped — install with: pip install onnx")
        return

    model.eval().to(device)

    B = 1
    dummy_pose = torch.randn(B, example_T, pose_dim, device=device)
    dummy_lens = torch.tensor([example_T], dtype=torch.long, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Wrap encode() in a small nn.Module for cleaner ONNX export
    class EncoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pose, lens):
            z, gloss_logp, src_mask = self.m.encode(pose, lens)
            return z, gloss_logp

    wrapper = EncoderWrapper(model).to(device)

    torch.onnx.export(
        wrapper,
        (dummy_pose, dummy_lens),
        path,
        input_names=["pose", "pose_lengths"],
        output_names=["encoder_output", "gloss_log_probs"],
        dynamic_axes={
            "pose":           {0: "batch", 1: "seq_len"},
            "pose_lengths":   {0: "batch"},
            "encoder_output": {0: "batch", 1: "seq_len"},
            "gloss_log_probs":{0: "seq_len", 1: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Verify
    model_onnx = onnx.load(path)
    onnx.checker.check_model(model_onnx)
    print(f"[export] ONNX (encoder) → {path}  [opset={opset}, verified ✓]")


# ---------------------------------------------------------------------------
# Combined export function
# ---------------------------------------------------------------------------

def export_model(
    model: nn.Module,
    word_vocab,
    gloss_vocab,
    cfg: dict,
    export_dir: str,
    device: Optional[torch.device] = None,
    export_onnx_flag: bool = True,
    export_torchscript_flag: bool = True,
):
    """
    Export model in all formats.

    Creates:
      {export_dir}/model.pt               — portable checkpoint (always)
      {export_dir}/model_encoder.torchscript.pt  — TorchScript (optional)
      {export_dir}/model_encoder.onnx            — ONNX (optional)
      {export_dir}/config.json                   — human-readable config
    """
    if device is None:
        device = next(model.parameters()).device

    os.makedirs(export_dir, exist_ok=True)
    pose_dim = cfg.get("pose_dim", 1629)

    # 1. Portable checkpoint
    export_portable_checkpoint(
        model, word_vocab, gloss_vocab, cfg,
        path=os.path.join(export_dir, "model.pt"),
    )

    # 2. TorchScript
    if export_torchscript_flag:
        export_torchscript(
            model, pose_dim,
            path=os.path.join(export_dir, "model_encoder.torchscript.pt"),
            device=device,
        )

    # 3. ONNX
    if export_onnx_flag:
        export_onnx(
            model, pose_dim,
            path=os.path.join(export_dir, "model_encoder.onnx"),
            device=device,
        )

    # 4. Human-readable config
    config_bundle = {
        "model_config": model.get_config(),
        "training_cfg": cfg,
        "word_vocab_size": len(word_vocab),
        "gloss_vocab_size": len(gloss_vocab),
    }
    cfg_path = os.path.join(export_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_bundle, f, indent=2)
    print(f"[export] Config → {cfg_path}")
    print(f"\n[export] All exports complete → {export_dir}/")
    print(f"         model.pt                        (portable, always loadable)")
    print(f"         model_encoder.torchscript.pt    (TorchScript, C++/mobile)")
    print(f"         model_encoder.onnx              (ONNX Runtime, cross-platform)")
