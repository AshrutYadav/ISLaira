

"""
Training script for Sign Language Transformer (pose → English/German).
 
Features:
  - Joint recognition + translation training (paper equation 8)
  - Adam optimiser with plateau LR scheduling (paper §5.1)
  - Checkpoint saving after every improvement
  - Resume from checkpoint after crash  ← NEW
  - Gradient accumulation               ← NEW
  - BLEU score evaluation on dev set
  - Model export after training
 
Usage:
    python train.py --config config.json
    python train.py --config config.json --resume          # auto-resumes from best/latest
    python train.py --config config.json --resume_from checkpoints/checkpoint_epoch0008.pt
    python train.py --dummy_data                           # smoke-test
"""
 
import argparse
import json
import os
import time
import sys
import math
import glob
from typing import Optional
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.sign_language_transformer import SignLanguageTransformer
from utils.vocabulary import Vocabulary
from utils.metrics import compute_bleu, compute_wer
from data.dataset import build_dataloader, generate_dummy_annotations
from utils.exporter import export_model
 
 
# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
 
def save_checkpoint(
    path, model, optimizer, scheduler,
    epoch, metrics, cfg, word_vocab, gloss_vocab, history,
):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics":              metrics,
        "model_config":         model.get_config(),
        "cfg":                  cfg,
        "history":              history,
    }, path)
 
 
def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load checkpoint. Returns (start_epoch, best_dev_loss, history)."""
    print(f"[resume] Loading checkpoint: {path}")
    state = torch.load(path, map_location=device)
 
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
 
    epoch         = state["epoch"]
    history       = state.get("history", [])
    best_dev_loss = min((r["dev_loss"] for r in history), default=math.inf)
 
    print(f"[resume] Resumed from epoch {epoch}  best_dev_loss={best_dev_loss:.4f}")
    return epoch, best_dev_loss, history
 
 
def find_latest_checkpoint(ckpt_dir):
    """Returns most recent periodic checkpoint, falling back to best."""
    epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pt")))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    best = os.path.join(ckpt_dir, "checkpoint_best.pt")
    if os.path.exists(best):
        return best
    return None
 
 
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
 
def run_epoch(
    model, loader, optimizer, device,
    train=True, grad_clip=1.0, accumulation_steps=1,
):
    model.train(train)
    total_loss = total_lr = total_lt = n_batches = 0
 
    if train:
        optimizer.zero_grad()
 
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            pose       = batch["pose"].to(device)
            pose_lens  = batch["pose_lens"].to(device)
            gloss_ids  = batch["gloss_ids"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)
            trans_ids  = batch["trans_ids"].to(device)
            trans_lens = batch["trans_lens"].to(device)
 
            loss, lr, lt = model(
                pose, pose_lens,
                trans_ids, trans_lens,
                gloss_ids, gloss_lens,
            )
 
            if train:
                (loss / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
 
            total_loss += loss.item()
            total_lr   += lr.item()
            total_lt   += lt.item()
            n_batches  += 1
 
    # flush remaining gradients in last incomplete accumulation window
    if train and (n_batches % accumulation_steps != 0):
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
 
    n = max(n_batches, 1)
    return {"loss": total_loss / n, "loss_r": total_lr / n, "loss_t": total_lt / n}
 
 
@torch.no_grad()
def evaluate_translations(model, loader, word_vocab, device, beam_size=5, max_gen_len=100):
    model.eval()
    hypotheses, references = [], []
 
    for batch in loader:
        pose      = batch["pose"].to(device)
        pose_lens = batch["pose_lens"].to(device)
        trans_ids = batch["trans_ids"]
 
        preds = model.generate(pose, pose_lens, beam_size=beam_size, max_len=max_gen_len)
 
        for pred_ids, ref_ids in zip(preds, trans_ids.tolist()):
            pred_toks = word_vocab.decode(pred_ids, skip_special=True)
            ref_toks  = word_vocab.decode(ref_ids,  skip_special=True)
            hypotheses.append(pred_toks)
            references.append([ref_toks])
 
    bleu = compute_bleu(hypotheses, references)
    return {"bleu4": bleu["bleu4"], "bleu1": bleu["bleu1"]}
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def train(cfg, resume_path=None):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cfg.get("cpu", False) else "cpu"
    )
    print(f"[train] device = {device}")
 
    # ── Vocabularies + annotations ────────────────────────────────────────
    if cfg.get("dummy_data"):
        print("[train] Generating dummy data …")
        train_anns, all_g, all_w = generate_dummy_annotations(
            n_samples=cfg.get("dummy_n", 200),
            pose_dir=cfg.get("pose_dir", "/tmp/pose2text_dummy"),
            pose_dim=cfg.get("pose_dim", 1728),
        )
        dev_anns    = train_anns[:20]
        gloss_vocab = Vocabulary(); gloss_vocab.build_from_list(all_g)
        word_vocab  = Vocabulary(); word_vocab.build_from_list(all_w)
    else:
        use_gloss   = cfg.get("use_gloss", True)
        gloss_vocab = (
            Vocabulary.load(cfg["gloss_vocab_path"])
            if use_gloss and cfg.get("gloss_vocab_path") else None
        )
        word_vocab = Vocabulary.load(cfg["word_vocab_path"])
 
        train_key = "train_annotations" if "train_annotations" in cfg else "train_path"
        dev_key   = "dev_annotations"   if "dev_annotations"   in cfg else "dev_path"
        with open(cfg[train_key]) as f: train_anns = json.load(f)
        with open(cfg[dev_key])   as f: dev_anns   = json.load(f)
 
    use_gloss        = cfg.get("use_gloss", True)
    gloss_vocab_size = len(gloss_vocab) if gloss_vocab is not None else 1
    print(f"[vocab] gloss size={gloss_vocab_size}  word size={len(word_vocab)}")
 
    # ── DataLoaders ───────────────────────────────────────────────────────
    batch_size  = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 0)
    accum_steps = cfg.get("gradient_accumulation_steps", 4)
    print(f"[train] batch_size={batch_size}  accumulation_steps={accum_steps}"
          f"  effective_batch={batch_size * accum_steps}")
 
    train_loader = build_dataloader(
        train_anns, gloss_vocab, word_vocab,
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        normalise=cfg.get("normalise", "z-score"),
    )
    dev_loader = build_dataloader(
        dev_anns, gloss_vocab, word_vocab,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        normalise=cfg.get("normalise", "z-score"),
    )
 
    # ── Model ─────────────────────────────────────────────────────────────
    blank_idx = gloss_vocab.pad_idx if gloss_vocab is not None else 0
    model = SignLanguageTransformer(
        pose_input_dim      = cfg.get("pose_dim", 1728),
        gloss_vocab_size    = gloss_vocab_size,
        word_vocab_size     = len(word_vocab),
        d_model             = cfg.get("d_model", 512),
        nhead               = cfg.get("nhead", 8),
        num_encoder_layers  = cfg.get("num_encoder_layers", 3),
        num_decoder_layers  = cfg.get("num_decoder_layers", 3),
        dim_feedforward     = cfg.get("dim_feedforward", 2048),
        dropout             = cfg.get("dropout", 0.1),
        max_seq_len         = cfg.get("max_seq_len", 4096),
        pad_idx             = word_vocab.pad_idx,
        bos_idx             = word_vocab.bos_idx,
        eos_idx             = word_vocab.eos_idx,
        blank_idx           = blank_idx,
        lambda_recognition  = cfg.get("lambda_recognition", 5.0) if use_gloss else 0.0,
        lambda_translation  = cfg.get("lambda_translation", 1.0),
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] parameters = {n_params:,}")
 
    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg.get("lr", 1e-3),
        betas        = (cfg.get("beta1", 0.9), cfg.get("beta2", 0.998)),
        weight_decay = cfg.get("weight_decay", 1e-3),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = cfg.get("lr_decay", 0.7),
        patience = cfg.get("patience", 8),
        min_lr   = 1e-6,
    )
 
    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
 
    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch      = 0
    best_dev_loss    = math.inf
    patience_counter = 0
    best_epoch       = 0
    history          = []
 
    # Auto-resume: if no explicit path given, look for latest checkpoint
    if resume_path is None and cfg.get("auto_resume", True):
        resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f"[resume] Auto-detected checkpoint: {resume_path}")
 
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_dev_loss, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        best_epoch = start_epoch
        # Reconstruct patience counter from history
        if history:
            patience_counter = 0
            for r in reversed(history):
                if r["dev_loss"] <= best_dev_loss:
                    break
                patience_counter += 1
            # cap so we don't immediately early-stop on resume
            patience_counter = min(patience_counter, cfg.get("early_stop_patience", 20) - 1)
        print(f"[resume] patience_counter restored to {patience_counter}")
    elif resume_path:
        print(f"[resume] WARNING: {resume_path} not found — starting fresh.")
 
    # ── Training loop ─────────────────────────────────────────────────────
    max_epochs = cfg.get("max_epochs", 200)
    eval_every = cfg.get("eval_every", 1)
    save_every = cfg.get("save_every", 5)
    early_stop = cfg.get("early_stop_patience", 20)
 
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Dev Loss':>10}  {'BLEU-4':>8}  {'LR':>10}")
    print("-" * 60)
 
    for epoch in range(start_epoch + 1, max_epochs + 1):
        t0 = time.time()
 
        train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            train=True, accumulation_steps=accum_steps,
        )
        dev_metrics = run_epoch(
            model, dev_loader, None, device,
            train=False, accumulation_steps=1,
        )
        dev_loss = dev_metrics["loss"]
 
        bleu4 = 0.0
        if epoch % eval_every == 0:
            trans_metrics = evaluate_translations(
                model, dev_loader, word_vocab, device,
                beam_size=cfg.get("beam_size", 5),
            )
            bleu4 = trans_metrics["bleu4"]
 
        scheduler.step(dev_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
 
        print(
            f"{epoch:>6}  {train_metrics['loss']:>12.4f}  {dev_loss:>10.4f}"
            f"  {bleu4:>8.2f}  {current_lr:>10.2e}  ({dt:.1f}s)"
        )
 
        row = {
            "epoch":        epoch,
            "train_loss":   train_metrics["loss"],
            "train_loss_r": train_metrics["loss_r"],
            "train_loss_t": train_metrics["loss_t"],
            "dev_loss":     dev_loss,
            "bleu4":        bleu4,
            "lr":           current_lr,
        }
        history.append(row)
 
        # ── Save history every epoch (crash-safe) ─────────────────────────
        hist_path = os.path.join(ckpt_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
 
        # ── Save best checkpoint ───────────────────────────────────────────
        if dev_loss < best_dev_loss:
            best_dev_loss    = dev_loss
            best_epoch       = epoch
            patience_counter = 0
            save_checkpoint(
                os.path.join(ckpt_dir, "checkpoint_best.pt"),
                model, optimizer, scheduler, epoch, row, cfg,
                word_vocab, gloss_vocab, history,
            )
            print(f"          ↑ new best (dev_loss={dev_loss:.4f})  [saved checkpoint_best.pt]")
        else:
            patience_counter += 1
 
        # ── Save periodic checkpoint ───────────────────────────────────────
        if epoch % save_every == 0:
            tag  = f"checkpoint_epoch{epoch:04d}.pt"
            path = os.path.join(ckpt_dir, tag)
            save_checkpoint(
                path,
                model, optimizer, scheduler, epoch, row, cfg,
                word_vocab, gloss_vocab, history,
            )
            print(f"          [saved {tag}]")
 
            # Keep only last 3 periodic checkpoints to save disk
            epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pt")))
            for old in epoch_ckpts[:-3]:
                os.remove(old)
                print(f"          [deleted old: {os.path.basename(old)}]")
 
        # ── Early stopping ────────────────────────────────────────────────
        if patience_counter >= early_stop:
            print(f"\n[train] Early stopping at epoch {epoch} "
                  f"(no improvement for {early_stop} epochs)")
            break
 
        if current_lr <= 1e-6:
            print(f"\n[train] Learning rate at minimum — stopping.")
            break
 
    # ── Export best model ─────────────────────────────────────────────────
    best_ckpt = os.path.join(ckpt_dir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt):
        print(f"\n[train] Exporting best model …")
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        export_model(
            model, word_vocab, gloss_vocab, cfg,
            cfg.get("export_dir", "exports"), device,
        )
 
    print(f"\n[train] Done. Best epoch={best_epoch}  best_dev_loss={best_dev_loss:.4f}")
    print(f"[train] History saved → {hist_path}")
    return history
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args():
    p = argparse.ArgumentParser(description="Train Sign Language Transformer")
    p.add_argument("--config",      type=str,  default=None)
    p.add_argument("--resume",      action="store_true",
                   help="Auto-resume from latest checkpoint in checkpoint_dir")
    p.add_argument("--resume_from", type=str,  default=None,
                   help="Explicit path to a .pt checkpoint to resume from")
    p.add_argument("--dummy_data",  action="store_true")
    p.add_argument("--dummy_n",     type=int,  default=200)
    p.add_argument("--pose_dim",    type=int,  default=1728)
    p.add_argument("--d_model",     type=int,  default=512)
    p.add_argument("--nhead",       type=int,  default=8)
    p.add_argument("--num_encoder_layers", type=int, default=3)
    p.add_argument("--num_decoder_layers", type=int, default=3)
    p.add_argument("--batch_size",  type=int,  default=8)
    p.add_argument("--max_epochs",  type=int,  default=200)
    p.add_argument("--lr",          type=float,default=1e-3)
    p.add_argument("--lambda_recognition", type=float, default=0.0)
    p.add_argument("--lambda_translation", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--export_dir",  type=str,  default="exports")
    p.add_argument("--cpu",         action="store_true")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    cfg  = json.load(open(args.config)) if args.config else vars(args)
 
    if args.dummy_data: cfg["dummy_data"] = True
    if args.dummy_n:    cfg["dummy_n"]    = args.dummy_n
 
    # Resolve resume path
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = find_latest_checkpoint(cfg.get("checkpoint_dir", "checkpoints"))
        if resume_path:
            print(f"[cli] --resume: will load {resume_path}")
        else:
            print("[cli] --resume: no checkpoint found, starting fresh.")
 
    train(cfg, resume_path=resume_path)