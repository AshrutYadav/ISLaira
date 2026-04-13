"""
Sign Language Transformer: Joint End-to-end Sign Language Recognition and Translation
Based on: "Sign Language Transformers: Joint End-to-end Sign Language Recognition and
Translation" by Camgöz et al. (2020) - arXiv:2003.13830

Architecture:
  - SLRT (Sign Language Recognition Transformer): Encoder using CTC loss
  - SLTT (Sign Language Translation Transformer): Autoregressive decoder
  - Joint training with weighted loss: L = λR * LR + λT * LT

Input: Pose keypoint sequences (from pose_format / mediapipe)
Output: English text sentences

Python 3.9 compatible.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from Vaswani et al. (2017).
    Adds temporal ordering information to embedded representations.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional information added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Spatial (Pose) Embedding
# ---------------------------------------------------------------------------

class PoseSpatialEmbedding(nn.Module):
    """
    Embeds raw pose keypoints into a dense d_model space.

    Replaces the CNN-based SpatialEmbedding from the paper (which was used
    for RGB video). For pose data we use a 2-layer MLP with BatchNorm and
    ReLU — matching the BN+ReLU trick that gave a 7% WER improvement in
    Table 2 of the paper.

    Input:  (batch, T, num_keypoints * coords_per_keypoint)
    Output: (batch, T, d_model)
    """

    def __init__(
        self,
        input_dim: int,       # num_keypoints * coords_per_keypoint
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = d_model * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            (batch, T, d_model)
        """
        B, T, C = x.shape
        # BatchNorm1d expects (N, C) or (N, C, L)
        x = x.reshape(B * T, C)
        x = self.net(x)
        return x.reshape(B, T, -1)


# ---------------------------------------------------------------------------
# SLRT – Sign Language Recognition Transformer (Encoder + CTC)
# ---------------------------------------------------------------------------

class SLRT(nn.Module):
    """
    Sign Language Recognition Transformer.

    A standard TransformerEncoder trained with CTC loss to recognise
    sign glosses from pose sequences. Its spatio-temporal representations
    are shared with the SLTT decoder.

    Paper §3.2 — equations (2)(3)(4).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        gloss_vocab_size: int = 1066,  # PHOENIX14T default
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # CTC projection: d_model → gloss_vocab_size + 1 (blank)
        self.ctc_head = nn.Linear(d_model, gloss_vocab_size + 1)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src:  (batch, T, d_model) — positionally-encoded pose embeddings
            src_key_padding_mask: (batch, T) bool mask (True = ignore)
        Returns:
            z:          (batch, T, d_model)  spatio-temporal representations
            gloss_logp: (T, batch, vocab+1)  log-softmax for CTC
        """
        z = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        # CTC expects (T, N, C)
        gloss_logits = self.ctc_head(z).permute(1, 0, 2)
        gloss_logp = F.log_softmax(gloss_logits, dim=-1)
        return z, gloss_logp


# ---------------------------------------------------------------------------
# SLTT – Sign Language Translation Transformer (Decoder)
# ---------------------------------------------------------------------------

class SLTT(nn.Module):
    """
    Sign Language Translation Transformer.

    Autoregressive TransformerDecoder that attends to SLRT encoder
    representations (z) and generates spoken-language words one at a time.

    Paper §3.3 — equations (5)(6)(7).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        word_vocab_size: int = 2887,  # PHOENIX14T German default
    ):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.word_proj = nn.Linear(d_model, word_vocab_size)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to enforce causality (autoregressive)."""
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:   (batch, U, d_model) — word embeddings with positional enc.
            memory:(batch, T, d_model) — SLRT encoder output z
            tgt_key_padding_mask:    (batch, U) bool
            memory_key_padding_mask: (batch, T) bool
        Returns:
            logits: (batch, U, word_vocab_size)
        """
        causal = self._causal_mask(tgt.size(1), tgt.device)
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.word_proj(out)


# ---------------------------------------------------------------------------
# Full Sign Language Transformer (SLRT + SLTT)
# ---------------------------------------------------------------------------

class SignLanguageTransformer(nn.Module):
    """
    Joint Sign Language Recognition and Translation Transformer.

    Implements Sign2(Gloss+Text) from the paper:
      L = λR * LR + λT * LT

    where:
      LR = CTC loss on gloss recognition
      LT = Cross-entropy loss on word translation

    The model is fully exportable via:
      - torch.save / torch.load  (training checkpoints)
      - torch.onnx.export        (ONNX for cross-platform deployment)
      - torch.jit.script         (TorchScript for mobile / C++)
    """

    def __init__(
        self,
        pose_input_dim: int,
        gloss_vocab_size: int,
    
        word_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        blank_idx: int = 0,          # CTC blank token index
        lambda_recognition: float = 5.0,
        lambda_translation: float = 1.0,
    ):
        super().__init__()

        # ── Hyper-parameters ─────────────────────────────────────────────
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.blank_idx = blank_idx
        self.lambda_recognition = lambda_recognition
        self.lambda_translation = lambda_translation
        self.gloss_vocab_size = gloss_vocab_size
        self.word_vocab_size = word_vocab_size

        # ── Pose (spatial) embedding  ─────────────────────────────────────
        self.pose_embedding = PoseSpatialEmbedding(pose_input_dim, d_model, dropout)

        # ── Positional encodings  ─────────────────────────────────────────
        self.pose_pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)
        self.word_pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)

        # ── Word embedding (target side) ──────────────────────────────────
        self.word_embedding = nn.Embedding(word_vocab_size, d_model, padding_idx=pad_idx)


        # ── SLRT encoder ─────────────────────────────────────────────────
        self.slrt = SLRT(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            gloss_vocab_size=gloss_vocab_size,
        )

        # ── SLTT decoder ─────────────────────────────────────────────────
        self.sltt = SLTT(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            word_vocab_size=word_vocab_size,
        )

        # ── Losses ───────────────────────────────────────────────────────
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_idx
        )

        # ── Weight initialisation (Xavier uniform, paper §5.1) ───────────
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Forward ──────────────────────────────────────────────────────────

    def encode(
        self,
        pose: torch.Tensor,
        pose_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a batch of pose sequences.

        Args:
            pose:         (B, T, pose_input_dim) raw keypoints
            pose_lengths: (B,) actual frame counts (before padding)
        Returns:
            z:             (B, T, d_model)
            gloss_logp:    (T, B, gloss_vocab+1)  for CTC
            src_pad_mask:  (B, T) bool (True = padding frame)
        """
        B, T, _ = pose.shape
        src_pad_mask = self._length_mask(T, pose_lengths, pose.device)  # (B, T)

        f = self.pose_embedding(pose)                     # (B, T, d_model)
        f_hat = self.pose_pos_enc(f)                      # + positional enc
        z, gloss_logp = self.slrt(f_hat, src_pad_mask)
        return z, gloss_logp, src_pad_mask

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_lengths: Optional[torch.Tensor] = None,
        memory_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target word sequence given encoder memory.

        Args:
            tgt_ids:   (B, U) word token ids (teacher-forced)
            memory:    (B, T, d_model)
            tgt_lengths: (B,) actual target lengths
            memory_pad_mask: (B, T) bool
        Returns:
            logits: (B, U, word_vocab_size)
        """
        B, U = tgt_ids.shape
        # One-hot → linear embedding (paper uses a linear layer for word emb)
        m = self.word_embedding(tgt_ids)
        m_hat = self.word_pos_enc(m)

        tgt_pad_mask = None
        if tgt_lengths is not None:
            tgt_pad_mask = self._length_mask(U, tgt_lengths, tgt_ids.device)

        logits = self.sltt(m_hat, memory, tgt_pad_mask, memory_pad_mask)
        return logits

    def forward(
        self,
        pose: torch.Tensor,
        pose_lengths: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_lengths: torch.Tensor,
        gloss_ids: torch.Tensor,
        gloss_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (training).

        Args:
            pose:          (B, T, pose_input_dim)
            pose_lengths:  (B,)
            tgt_ids:       (B, U+1) includes <bos> prefix; labels shift by 1
            tgt_lengths:   (B,)
            gloss_ids:     (B, N)  ground-truth gloss token ids (padded)
            gloss_lengths: (B,)

        Returns:
            total_loss:  scalar
            loss_r:      recognition (CTC) loss — scalar
            loss_t:      translation (CE) loss  — scalar
        """
        # ── Encode ────────────────────────────────────────────────────────
        z, gloss_logp, src_pad_mask = self.encode(pose, pose_lengths)

        # ── CTC recognition loss (LR) ─────────────────────────────────────
        # Input lengths for CTC: number of valid encoder frames
        input_lengths = pose_lengths.clamp(max=z.size(1))
        loss_r = self.ctc_loss(gloss_logp, gloss_ids, input_lengths, gloss_lengths)

        # ── Autoregressive translation loss (LT) ──────────────────────────
        # Teacher forcing: input = tgt[:, :-1], label = tgt[:, 1:]
        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]
        dec_lengths = (tgt_lengths - 1).clamp(min=1)

        logits = self.decode(tgt_in, z, dec_lengths, src_pad_mask)  # (B, U, V)
        B, U, V = logits.shape
        loss_t = self.ce_loss(logits.reshape(B * U, V), tgt_out.reshape(B * U))

        # ── Joint loss (equation 8) ───────────────────────────────────────
        total_loss = self.lambda_recognition * loss_r + self.lambda_translation * loss_t
        return total_loss, loss_r, loss_t

    # ── Inference (beam search) ───────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        pose: torch.Tensor,
        pose_lengths: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 100,
        length_penalty: float = 1.0,
    ) -> List[List[int]]:
        """
        Generate English token ids from a pose sequence using beam search.

        Args:
            pose:         (B, T, pose_input_dim)  — B=1 recommended
            pose_lengths: (B,)
            beam_size:    width of beam (0 = greedy)
            max_len:      maximum output tokens
            length_penalty: α from Wu et al. (2016), applied as len**α

        Returns:
            List[List[int]]  — one token-id list per batch item
        """
        self.eval()
        z, _, src_pad_mask = self.encode(pose, pose_lengths)

        if beam_size <= 1:
            return self._greedy_decode(z, src_pad_mask, max_len)
        return self._beam_decode(z, src_pad_mask, beam_size, max_len, length_penalty)

    def _greedy_decode(
        self,
        memory: torch.Tensor,
        mem_mask: torch.Tensor,
        max_len: int,
    ) -> List[List[int]]:
        B = memory.size(0)
        ys = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=memory.device)
        finished = [False] * B

        for _ in range(max_len):
            logits = self.decode(ys, memory, memory_pad_mask=mem_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1)  # (B,)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            for i in range(B):
                if next_tok[i].item() == self.eos_idx:
                    finished[i] = True
            if all(finished):
                break

        results = []
        for i in range(B):
            seq = ys[i, 1:].tolist()  # strip <bos>
            if self.eos_idx in seq:
                seq = seq[: seq.index(self.eos_idx)]
            results.append(seq)
        return results

    def _beam_decode(
        self,
        memory: torch.Tensor,
        mem_mask: torch.Tensor,
        beam_size: int,
        max_len: int,
        alpha: float,
    ) -> List[List[int]]:
        """Beam search (single-batch item for clarity; loops over batch)."""
        B = memory.size(0)
        results = []
        for b in range(B):
            mem_b = memory[b: b + 1]                # (1, T, d)
            msk_b = mem_mask[b: b + 1] if mem_mask is not None else None
            best = self._beam_single(mem_b, msk_b, beam_size, max_len, alpha)
            results.append(best)
        return results

    def _beam_single(self, mem, msk, beam_size, max_len, alpha):
        device = mem.device
        # beams: list of (score, token_ids)
        beams = [(0.0, [self.bos_idx])]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == self.eos_idx:
                    completed.append((score, seq))
                    continue
                tgt = torch.tensor([seq], device=device)          # (1, L)
                logits = self.decode(tgt, mem, memory_pad_mask=msk)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)  # (V,)
                topk_log, topk_idx = log_probs.topk(beam_size)
                for lp, idx in zip(topk_log.tolist(), topk_idx.tolist()):
                    candidates.append((score + lp, seq + [idx]))

            if not candidates:
                break

            # length-normalised score
            def normed(item):
                s, seq = item
                length = max(len(seq) - 1, 1)
                return s / (length ** alpha)

            candidates.sort(key=normed, reverse=True)
            beams = candidates[:beam_size]

            if all(s[-1] == self.eos_idx for _, s in beams):
                completed.extend(beams)
                break

        if not completed:
            completed = beams

        best_score, best_seq = max(completed, key=lambda x: x[0] / max(len(x[1]) - 1, 1) ** alpha)
        # strip bos / eos
        if best_seq and best_seq[0] == self.bos_idx:
            best_seq = best_seq[1:]
        if best_seq and best_seq[-1] == self.eos_idx:
            best_seq = best_seq[:-1]
        return best_seq

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _length_mask(max_len: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Build a boolean padding mask (True = padding position).
        Shape: (batch, max_len)
        """
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, L)
        return idx >= lengths.unsqueeze(1)                         # (B, L)

    # ── Model config (for saving/loading) ────────────────────────────────

    def get_config(self) -> dict:
        """Return all constructor arguments for reproducible export."""
        return {
            "pose_input_dim": self.pose_embedding.net[0].in_features,
            "gloss_vocab_size": self.gloss_vocab_size,
            "word_vocab_size": self.word_vocab_size,
            "d_model": self.d_model,
            "nhead": self.slrt.encoder.layers[0].self_attn.num_heads,
            "num_encoder_layers": len(self.slrt.encoder.layers),
            "num_decoder_layers": len(self.sltt.decoder.layers),
            "dim_feedforward": self.slrt.encoder.layers[0].linear1.out_features,
            "dropout": self.slrt.encoder.layers[0].dropout.p,
            "pad_idx": self.pad_idx,
            "bos_idx": self.bos_idx,
            "eos_idx": self.eos_idx,
            "blank_idx": self.blank_idx,
            "lambda_recognition": self.lambda_recognition,
            "lambda_translation": self.lambda_translation,
        }
