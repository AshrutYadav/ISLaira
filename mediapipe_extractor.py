"""
MediaPipe Pose Extractor — iSign Format
========================================
Extracts pose keypoints in EXACTLY the same format as the iSign .pose files
used during training.

Verified format (from pose_format inspection):
    Shape per file: (T, 576, 3)
    Flattened:      (T, 1728)
    Components:     Full MediaPipe Holistic — 576 keypoints × 3 coords
                    468 face + 33 pose + 21 left_hand + 21 right_hand
                    + 33 world_landmarks = 576 total

Normalization: z-score per sequence (matches dataset.py normalise="z-score")

Install:
    pip install mediapipe opencv-python numpy

Usage:
    # From video file:
    python mediapipe_extractor.py --input sign_video.mp4 --out pose.npy

    # From webcam (press R to start/stop recording, Q to quit):
    python mediapipe_extractor.py --webcam --out pose.npy

    # Extract + translate immediately:
    python mediapipe_extractor.py --webcam --out pose.npy --translate \
        --checkpoint checkpoints/checkpoint_best.pt \
        --vocab data/word_vocab.json
"""

import argparse
import os
import sys
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed.")
    print("Run: pip install mediapipe")
    sys.exit(1)


# ── Constants — must match training data exactly ──────────────────────────────
# Verified from: Pose.read(file).body.data[:, 0, :, :] → shape (T, 576, 3)

N_FACE       = 468   # face mesh
N_POSE       = 33    # body pose
N_LEFT_HAND  = 21   # left hand
N_RIGHT_HAND = 21   # right hand
N_POSE_WORLD = 33   # world pose landmarks

TOTAL_KEYPOINTS = N_FACE + N_POSE + N_LEFT_HAND + N_RIGHT_HAND + N_POSE_WORLD
# 468 + 33 + 21 + 21 + 33 = 576  ✓
POSE_DIM = TOTAL_KEYPOINTS * 3  # 576 × 3 = 1728  ✓


# ── Extractor class ───────────────────────────────────────────────────────────

class ISLPoseExtractor:
    """
    Extracts full MediaPipe Holistic keypoints matching iSign .pose format.
    Output shape per frame: (1728,) — zeros for missing/undetected landmarks.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ):
        self.mp_holistic  = mp.solutions.holistic
        self.mp_drawing   = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=True,    # needed for all 468 face points
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self.holistic.close()

    def _landmarks_to_array(self, landmark_list, n: int) -> np.ndarray:
        """Convert a MediaPipe landmark list to a flat (n*3,) array."""
        if landmark_list is None:
            return np.zeros(n * 3, dtype=np.float32)
        out = np.zeros(n * 3, dtype=np.float32)
        for i, lm in enumerate(landmark_list.landmark[:n]):
            out[i*3]   = lm.x
            out[i*3+1] = lm.y
            out[i*3+2] = lm.z
        return out

    def extract_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Process a single BGR frame and return (1728,) keypoint vector.
        Order matches iSign .pose file layout exactly.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        # Build in the same order as pose_format stores them
        parts = [
            self._landmarks_to_array(results.face_landmarks,       N_FACE),
            self._landmarks_to_array(results.pose_landmarks,       N_POSE),
            self._landmarks_to_array(results.left_hand_landmarks,  N_LEFT_HAND),
            self._landmarks_to_array(results.right_hand_landmarks, N_RIGHT_HAND),
            self._landmarks_to_array(results.pose_world_landmarks, N_POSE_WORLD),
        ]
        frame_kp = np.concatenate(parts)   # (1728,)

        assert frame_kp.shape[0] == POSE_DIM, \
            f"Dimension mismatch: got {frame_kp.shape[0]}, expected {POSE_DIM}"

        return frame_kp, results   # return results for drawing

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw pose/hand/face landmarks on frame for preview."""
        # Pose
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 200, 0), thickness=2),
        )
        # Left hand
        self.mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 100, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(200, 80, 0), thickness=2),
        )
        # Right hand
        self.mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 100, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 80, 200), thickness=2),
        )
        return frame

    @staticmethod
    def normalize_zscore(keypoints: np.ndarray) -> np.ndarray:
        """
        Z-score normalize — matches dataset.py normalise='z-score'.
        Applied per sequence across all frames.
        Input/output shape: (T, 1728)
        """
        mean = keypoints.mean(axis=0, keepdims=True)
        std  = keypoints.std(axis=0,  keepdims=True) + 1e-8
        return ((keypoints - mean) / std).astype(np.float32)

    def extract_from_video(
        self,
        video_path: str,
        max_frames: int = 2000,
        show_preview: bool = True,
    ) -> np.ndarray:
        """
        Extract keypoints from a video file.
        Returns normalized (T, 1728) array.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        print(f"[extractor] Video: {video_path}")
        print(f"[extractor] Frames: {total}  FPS: {fps:.1f}")

        frames = []
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            kp, results = self.extract_frame(frame)
            frames.append(kp)

            if show_preview:
                display = frame.copy()
                self.draw_landmarks(display, results)
                cv2.putText(display,
                    f"Extracting frame {len(frames)}/{min(total, max_frames)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("Extracting Pose", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

        if not frames:
            raise RuntimeError("No frames extracted from video.")

        raw = np.stack(frames, axis=0)          # (T, 1728)
        normalized = self.normalize_zscore(raw) # z-score normalize
        print(f"[extractor] Extracted {raw.shape[0]} frames  shape={raw.shape}")
        return normalized

    def extract_from_webcam(
        self,
        camera_index: int = 0,
        max_frames: int = 2000,
    ) -> np.ndarray:
        """
        Interactive webcam recording.
        Press R to start/stop recording.
        Press Q to quit and save.
        Returns normalized (T, 1728) array.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {camera_index}")

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        recording = False
        frames    = []

        print("\n[webcam] Controls:")
        print("  R — start / stop recording")
        print("  Q — quit and save")
        print("  Make sure your full upper body is visible\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)   # mirror for natural feel
            kp, results = self.extract_frame(frame)

            if recording:
                frames.append(kp)
                if len(frames) >= max_frames:
                    print("[webcam] Max frames reached — stopping recording.")
                    break

            # Draw
            display = frame.copy()
            self.draw_landmarks(display, results)

            # Status bar
            status = "● REC" if recording else "○ READY"
            color  = (0, 0, 255) if recording else (0, 255, 0)
            cv2.rectangle(display, (0, 0), (400, 45), (0, 0, 0), -1)
            cv2.putText(display, f"{status}  frames={len(frames)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, "R=record  Q=quit+save",
                        (10, display.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("ISL Pose Extractor — Webcam", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("r") or key == ord("R"):
                recording = not recording
                state = "STARTED" if recording else "PAUSED"
                print(f"[webcam] Recording {state}  (frames so far: {len(frames)})")

        cap.release()
        cv2.destroyAllWindows()
        self.close()

        if not frames:
            raise RuntimeError("No frames recorded.")

        raw = np.stack(frames, axis=0)
        normalized = self.normalize_zscore(raw)
        print(f"[extractor] Recorded {raw.shape[0]} frames  shape={raw.shape}")
        return normalized


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract ISL pose keypoints for iSign translator"
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",  type=str, help="Path to input video file")
    source.add_argument("--webcam", action="store_true", help="Use webcam")

    p.add_argument("--out",        type=str, required=True,
                   help="Output .npy file path (e.g. pose.npy)")
    p.add_argument("--camera",     type=int, default=0,
                   help="Webcam camera index (default: 0)")
    p.add_argument("--max_frames", type=int, default=2000)
    p.add_argument("--no_preview", action="store_true",
                   help="Disable preview window (faster for long videos)")

    # Optional: translate immediately after extraction
    p.add_argument("--translate",   action="store_true",
                   help="Run translation immediately after extraction")
    p.add_argument("--checkpoint",  type=str, default="checkpoints/checkpoint_best.pt")
    p.add_argument("--vocab",       type=str, default="data/word_vocab.json")
    p.add_argument("--beam_size",   type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extractor = ISLPoseExtractor()

    # Extract
    if args.webcam:
        keypoints = extractor.extract_from_webcam(
            camera_index=args.camera,
            max_frames=args.max_frames,
        )
    else:
        keypoints = extractor.extract_from_video(
            video_path=args.input,
            max_frames=args.max_frames,
            show_preview=not args.no_preview,
        )

    # Save
    np.save(args.out, keypoints)
    print(f"[extractor] Saved → {args.out}  shape={keypoints.shape}")
    print(f"[extractor] pose_dim={keypoints.shape[1]}  "
          f"({'✓ matches model' if keypoints.shape[1] == 1728 else '✗ MISMATCH — check model config'})")

    # Optionally translate immediately
    if args.translate:
        print("\n[extractor] Running translation...")
        import subprocess
        subprocess.run([
            sys.executable, "scripts/predict.py",
            "--checkpoint", args.checkpoint,
            "--vocab",      args.vocab,
            "--pose",       args.out,
            "--beam_size",  str(args.beam_size),
        ])
