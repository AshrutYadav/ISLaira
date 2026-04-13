"""
Real-Time ISL Translator
=========================
Combines MediaPipe pose extraction with the trained Sign Language Transformer
for live Indian Sign Language → English translation.

How it works:
    1. Captures webcam frames continuously
    2. Extracts 576 keypoint × 3 coords = 1728 dims per frame
    3. Every N seconds, runs translation on the buffered frames
    4. Displays the English translation on screen

Usage:
    python realtime_translator.py \
        --checkpoint checkpoints/checkpoint_best.pt \
        --vocab      data/word_vocab.json

Controls:
    SPACE  — translate current buffer now
    R      — clear buffer and start fresh
    Q      — quit

Requirements:
    pip install mediapipe opencv-python torch numpy
"""

import sys
import os
import time
import argparse
import threading
import collections
import numpy as np
import cv2

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: pip install mediapipe")
    sys.exit(1)


# ── Pose extraction constants ─────────────────────────────────────────────────
N_FACE       = 468
N_POSE       = 33
N_LEFT_HAND  = 21
N_RIGHT_HAND = 21
N_POSE_WORLD = 33
POSE_DIM     = (N_FACE + N_POSE + N_LEFT_HAND + N_RIGHT_HAND + N_POSE_WORLD) * 3
# = 576 × 3 = 1728


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model_and_vocab(checkpoint_path: str, vocab_path: str, device: torch.device):
    """Load trained model and vocabulary."""
    from models.sign_language_transformer import SignLanguageTransformer
    from utils.vocabulary import Vocabulary

    print(f"[model] Loading vocab: {vocab_path}")
    word_vocab = Vocabulary.load(vocab_path)

    print(f"[model] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    cfg   = state["model_config"]

    model = SignLanguageTransformer(
        pose_input_dim     = cfg["pose_input_dim"],
        gloss_vocab_size   = cfg["gloss_vocab_size"],
        word_vocab_size    = cfg["word_vocab_size"],
        d_model            = cfg["d_model"],
        nhead              = cfg["nhead"],
        num_encoder_layers = cfg["num_encoder_layers"],
        num_decoder_layers = cfg["num_decoder_layers"],
        dim_feedforward    = cfg.get("dim_feedforward", 2048),
        dropout            = 0.0,   # no dropout at inference
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
    print(f"[model] Loaded  epoch={trained_epoch}  dev_loss={dev_loss}")
    print(f"[model] Vocab size: {len(word_vocab)}  pose_dim: {cfg['pose_input_dim']}")
    return model, word_vocab


# ── Inference ─────────────────────────────────────────────────────────────────

def translate(
    model,
    word_vocab,
    frames: np.ndarray,
    device: torch.device,
    beam_size: int = 5,
) -> str:
    """
    Translate a sequence of pose frames to English.
    frames: (T, 1728) numpy array, already z-score normalized
    """
    if len(frames) < 5:
        return "(too few frames)"

    with torch.no_grad():
        pose   = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        length = torch.tensor([frames.shape[0]], dtype=torch.long).to(device)

        pred_ids = model.generate(
            pose, length,
            beam_size=beam_size,
            max_len=50,
        )

    tokens = word_vocab.decode(pred_ids[0], skip_special=True)
    text   = " ".join(tokens) if tokens else "(no output)"
    return text


# ── Pose extraction helpers ───────────────────────────────────────────────────

def landmarks_to_array(landmark_list, n: int) -> np.ndarray:
    if landmark_list is None:
        return np.zeros(n * 3, dtype=np.float32)
    out = np.zeros(n * 3, dtype=np.float32)
    for i, lm in enumerate(landmark_list.landmark[:n]):
        out[i*3]   = lm.x
        out[i*3+1] = lm.y
        out[i*3+2] = lm.z
    return out


def extract_frame_keypoints(results) -> np.ndarray:
    """Extract (1728,) keypoint vector from MediaPipe Holistic results."""
    return np.concatenate([
        landmarks_to_array(results.face_landmarks,       N_FACE),
        landmarks_to_array(results.pose_landmarks,       N_POSE),
        landmarks_to_array(results.left_hand_landmarks,  N_LEFT_HAND),
        landmarks_to_array(results.right_hand_landmarks, N_RIGHT_HAND),
        landmarks_to_array(results.pose_world_landmarks, N_POSE_WORLD),
    ])


def zscore_normalize(frames: np.ndarray) -> np.ndarray:
    """Z-score normalize (T, 1728) array — matches training."""
    mean = frames.mean(axis=0, keepdims=True)
    std  = frames.std(axis=0,  keepdims=True) + 1e-8
    return ((frames - mean) / std).astype(np.float32)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_ui(
    frame: np.ndarray,
    mp_holistic,
    mp_drawing,
    results,
    recording: bool,
    n_frames: int,
    translation: str,
    processing: bool,
    fps: float,
) -> np.ndarray:
    """Draw landmarks and UI overlay on frame."""
    h, w = frame.shape[:2]

    # Draw landmarks
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,200,0), thickness=2),
    )
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255,100,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(200,80,0),  thickness=2),
    )
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,80,200),  thickness=2),
    )

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    status = "● RECORDING" if recording else "○ PAUSED"
    color  = (0, 80, 255) if recording else (80, 80, 80)
    cv2.putText(frame, status, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"frames={n_frames}  fps={fps:.0f}",
                (220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

    # Translation box at bottom
    box_h = 80
    cv2.rectangle(frame, (0, h-box_h), (w, h), (15, 15, 15), -1)

    if processing:
        label = "Translating..."
        cv2.putText(frame, label, (10, h-box_h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    else:
        # Word-wrap translation text
        words  = translation.split()
        line   = ""
        lines  = []
        for w_tok in words:
            if len(line) + len(w_tok) + 1 < 60:
                line += (" " if line else "") + w_tok
            else:
                lines.append(line)
                line = w_tok
        if line:
            lines.append(line)

        for i, ln in enumerate(lines[:2]):   # max 2 lines
            cv2.putText(frame, ln,
                        (10, h-box_h+28+i*28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)

    # Controls hint
    cv2.putText(frame, "SPACE=translate  R=clear  Q=quit",
                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (120, 120, 120), 1)

    return frame


# ── Main real-time loop ───────────────────────────────────────────────────────

def run_realtime(
    checkpoint_path: str,
    vocab_path:      str,
    camera_index:    int  = 0,
    beam_size:       int  = 5,
    auto_translate:  bool = True,
    auto_interval:   int  = 5,    # seconds between auto-translations
    max_buffer:      int  = 1500, # max frames to keep in buffer
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[translator] Device: {device}")

    # Load model
    model, word_vocab = load_model_and_vocab(checkpoint_path, vocab_path, device)

    # MediaPipe setup
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    holistic    = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Webcam
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {camera_index}")

    print("\n[translator] Controls:")
    print("  SPACE — translate current buffer")
    print("  R     — clear buffer")
    print("  Q     — quit\n")

    # State
    frame_buffer  = []
    translation   = "Sign and press SPACE to translate"
    recording     = True
    processing    = False
    last_auto_t   = time.time()

    # FPS calculation
    fps_counter   = collections.deque(maxlen=30)
    prev_time     = time.time()

    # Translation runs in background thread so UI doesn't freeze
    translate_lock   = threading.Lock()
    translate_result = [translation]

    def run_translation(frames_copy):
        normalized = zscore_normalize(np.array(frames_copy))
        result     = translate(model, word_vocab, normalized, device, beam_size)
        with translate_lock:
            translate_result[0] = result

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror

        # FPS
        now = time.time()
        fps_counter.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_counter) / len(fps_counter)

        # Extract keypoints
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        if recording:
            kp = extract_frame_keypoints(results)
            frame_buffer.append(kp)
            # Keep buffer bounded
            if len(frame_buffer) > max_buffer:
                frame_buffer = frame_buffer[-max_buffer:]

        # Auto-translate
        if auto_translate and recording and not processing:
            if now - last_auto_t >= auto_interval and len(frame_buffer) >= 10:
                last_auto_t = now
                buf_copy    = list(frame_buffer)
                processing  = True
                def _translate_done(buf=buf_copy):
                    run_translation(buf)
                    nonlocal processing
                    processing = False
                threading.Thread(target=_translate_done, daemon=True).start()

        # Get latest translation
        with translate_lock:
            translation = translate_result[0]

        # Draw UI
        frame = draw_ui(
            frame, mp_holistic, mp_drawing, results,
            recording, len(frame_buffer),
            translation, processing, fps,
        )

        cv2.imshow("ISL → English Translator", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord(" "):
            # Manual translate
            if len(frame_buffer) >= 5 and not processing:
                buf_copy   = list(frame_buffer)
                processing = True
                def _manual_translate(buf=buf_copy):
                    run_translation(buf)
                    nonlocal processing
                    processing = False
                threading.Thread(target=_manual_translate, daemon=True).start()
                print(f"[translator] Translating {len(buf_copy)} frames...")

        elif key == ord("r") or key == ord("R"):
            frame_buffer = []
            with translate_lock:
                translate_result[0] = "Buffer cleared — sign again"
            print("[translator] Buffer cleared")

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("[translator] Done")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Real-time ISL → English translator")
    p.add_argument("--checkpoint",    type=str,
                   default="checkpoints/checkpoint_best.pt")
    p.add_argument("--vocab",         type=str,
                   default="data/word_vocab.json")
    p.add_argument("--camera",        type=int,  default=0)
    p.add_argument("--beam_size",     type=int,  default=5)
    p.add_argument("--auto_interval", type=int,  default=5,
                   help="Seconds between automatic translations (default: 5)")
    p.add_argument("--no_auto",       action="store_true",
                   help="Disable auto-translate (manual SPACE only)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_realtime(
        checkpoint_path = args.checkpoint,
        vocab_path      = args.vocab,
        camera_index    = args.camera,
        beam_size       = args.beam_size,
        auto_translate  = not args.no_auto,
        auto_interval   = args.auto_interval,
    )
