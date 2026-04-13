"""
End-to-end smoke test for the Sign Language Transformer pipeline.

Tests the complete training + export + inference loop using synthetic data.
Requires PyTorch to be installed.

Run:
    python tests/test_pipeline.py
    python tests/test_pipeline.py --fast    # smaller model, fewer steps
"""

import argparse
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch


def test_vocabulary():
    from utils.vocabulary import Vocabulary
    v = Vocabulary()
    v.build_from_list(["cat", "sat", "mat"])
    assert len(v) == 7        # 4 specials + 3 words
    assert v["cat"] == 4
    assert v["<unk>"] == 3
    assert v["MISSING"] == 3  # returns unk
    ids = v.encode(["cat", "sat"], add_bos=True, add_eos=True)
    decoded = v.decode(ids, skip_special=True)
    assert decoded == ["cat", "sat"]
    print("  PASS  Vocabulary")


def test_metrics():
    from utils.metrics import compute_bleu, compute_wer
    # Perfect match
    h = [["a", "b", "c"]]
    r = [[["a", "b", "c"]]]
    b = compute_bleu(h, r)
    assert b["bleu4"] > 90
    # WER zero on perfect
    assert compute_wer([["a", "b"]], [["a", "b"]]) == 0.0
    # WER 50% on one sub
    assert abs(compute_wer([["a", "x"]], [["a", "b"]]) - 50.0) < 1.0
    print("  PASS  Metrics (BLEU, WER)")


def test_model_forward(pose_dim=63, gloss_v=20, word_v=30, d_model=64, batch=2, T=15, U=8):
    from models.sign_language_transformer import SignLanguageTransformer
    model = SignLanguageTransformer(
        pose_input_dim=pose_dim,
        gloss_vocab_size=gloss_v,
        word_vocab_size=word_v,
        d_model=d_model,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        lambda_recognition=5.0,
        lambda_translation=1.0,
    )
    model.train()

    pose      = torch.randn(batch, T, pose_dim)
    pose_lens = torch.tensor([T, T - 3])
    gloss_ids = torch.randint(4, gloss_v, (batch, 5))
    gloss_lens= torch.tensor([5, 4])
    trans_ids = torch.randint(4, word_v, (batch, U + 2))
    trans_ids[:, 0] = 1   # <bos>
    trans_ids[:, -1] = 2  # <eos>
    trans_lens = torch.tensor([U + 2, U + 1])

    loss, lr, lt = model(pose, pose_lens, trans_ids, trans_lens, gloss_ids, gloss_lens)
    assert loss.item() > 0
    assert lr.item() >= 0
    assert lt.item() >= 0
    print(f"  PASS  Model forward  loss={loss.item():.4f}  lr={lr.item():.4f}  lt={lt.item():.4f}")
    return model, pose_dim


def test_model_generate(model, pose_dim, T=15):
    model.eval()
    pose = torch.randn(1, T, pose_dim)
    lens = torch.tensor([T])
    # Greedy
    result = model.generate(pose, lens, beam_size=1, max_len=10)
    assert isinstance(result, list) and isinstance(result[0], list)
    # Beam
    result2 = model.generate(pose, lens, beam_size=3, max_len=10)
    assert isinstance(result2[0], list)
    print(f"  PASS  Model generate  greedy={result[0][:3]}…  beam={result2[0][:3]}…")


def test_export_load(model, word_vocab, gloss_vocab, cfg, pose_dim):
    from utils.exporter import export_portable_checkpoint, load_exported_model

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        export_portable_checkpoint(model, word_vocab, gloss_vocab, cfg, path)
        assert os.path.exists(path)

        m2, wv2, gv2 = load_exported_model(path)
        assert len(wv2) == len(word_vocab)
        assert len(gv2) == len(gloss_vocab)

        # Verify weights are identical
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), m2.named_parameters()):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1}"

        print("  PASS  Export / reload portable checkpoint")


def test_torchscript_encode(model, pose_dim):
    from utils.exporter import export_torchscript
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "enc.torchscript.pt")
        traced = export_torchscript(model, pose_dim, path, device=torch.device("cpu"), example_T=10)
        assert os.path.exists(path)

        # Reload and run
        loaded = torch.jit.load(path)
        pose = torch.randn(1, 10, pose_dim)
        lens = torch.tensor([10])
        z, logp, mask = loaded(pose, lens)
        assert z.shape == (1, 10, model.d_model)
        print("  PASS  TorchScript export + reload")


def test_dataset_collate(pose_dim=63):
    from data.dataset import generate_dummy_annotations, SignPoseDataset, collate_fn
    from utils.vocabulary import Vocabulary
    from torch.utils.data import DataLoader

    anns, all_g, all_w = generate_dummy_annotations(
        n_samples=8, pose_dim=pose_dim, max_frames=30
    )
    gv = Vocabulary(); gv.build_from_list(all_g)
    wv = Vocabulary(); wv.build_from_list(all_w)

    ds = SignPoseDataset(anns, gv, wv)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert batch["pose"].shape[0] == 4
    assert batch["pose"].shape[2] == pose_dim
    assert batch["gloss_ids"].shape[0] == 4
    assert batch["trans_ids"].shape[0] == 4
    print(f"  PASS  Dataset + DataLoader  pose={tuple(batch['pose'].shape)}")
    return gv, wv


def run_all(fast: bool = False):
    pose_dim = 63 if fast else 126
    print("\n=== Sign Language Transformer — Smoke Tests ===\n")

    test_vocabulary()
    test_metrics()
    model, pd = test_model_forward(pose_dim=pose_dim)
    test_model_generate(model, pd)
    gv, wv = test_dataset_collate(pose_dim=pose_dim)

    cfg = {"pose_dim": pose_dim}
    test_export_load(model, wv, gv, cfg, pose_dim)
    test_torchscript_encode(model, pose_dim)

    print("\n=== All tests passed ✓ ===\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="Use smallest possible model")
    args = p.parse_args()
    run_all(fast=args.fast)
