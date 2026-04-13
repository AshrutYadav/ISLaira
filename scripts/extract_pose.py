"""
Pose extractor using MediaPipe Holistic.

Converts video files or webcam streams into .npy pose files
compatible with the Sign Language Transformer.

Output shape: (T, 1629)
  - Face:       468 landmarks × 3 coords = 1404
  - Pose body:   33 landmarks × 3 coords =   99
  - Left hand:   21 landmarks × 3 coords =   63
  - Right hand:  21 landmarks × 3 coords =   63
  Total: 543 × 3 = 1629

Usage
-----
    # Extract from a video file:
    python extract_pose.py --video input.mp4 --output poses/sample.npy

    # Extract from webcam (live):
    python extract_pose.py --webcam --output poses/live.npy

    # Batch extract from a folder of videos:
    python extract_pose.py --video_dir videos/ --output_dir poses/

Dependencies:
    mediapipe, opencv-python (both in requirements)
"""

import argparse
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Holistic extractor
# ---------------------------------------------------------------------------

def extract_pose_from_video(
    video_path: str,
    output_path: str,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    static_image_mode: bool = False,
) -> int:
    """
    Extract holistic pose landmarks from a video file.

    Args:
        video_path:  input video (.mp4, .avi, etc.)
        output_path: output .npy file
        min_detection_confidence: MediaPipe detection threshold
        min_tracking_confidence:  MediaPipe tracking threshold
        static_image_mode: treat each frame independently (slower but more accurate)

    Returns:
        Number of frames extracted.
    """
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        raise ImportError(
            "mediapipe and opencv-python are required.\n"
            "Install: pip install mediapipe opencv-contrib-python"
        )

    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    holistic = mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = _extract_landmarks(results)
        frames.append(keypoints)

    cap.release()
    holistic.close()

    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    arr = np.stack(frames, axis=0).astype(np.float32)  # (T, 1629)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, arr)
    print(f"[extract] {os.path.basename(video_path)} → {output_path}  shape={arr.shape}")
    return len(frames)


def _extract_landmarks(results) -> np.ndarray:
    """
    Flatten all MediaPipe Holistic landmarks into a single 1629-dim vector.
    Missing landmarks are filled with zeros.
    """
    def _lm_to_array(landmark_list, n: int) -> np.ndarray:
        arr = np.zeros((n, 3), dtype=np.float32)
        if landmark_list:
            for i, lm in enumerate(landmark_list.landmark[:n]):
                arr[i] = [lm.x, lm.y, lm.z]
        return arr.flatten()

    face      = _lm_to_array(results.face_landmarks,       468)   # 1404
    body      = _lm_to_array(results.pose_landmarks,        33)   #   99
    left_hand = _lm_to_array(results.left_hand_landmarks,   21)   #   63
    right_hand= _lm_to_array(results.right_hand_landmarks,  21)   #   63

    return np.concatenate([face, body, left_hand, right_hand])     # 1629


# ---------------------------------------------------------------------------
# Webcam live extraction
# ---------------------------------------------------------------------------

def extract_pose_from_webcam(
    output_path: str,
    camera_id: int = 0,
    fps_limit: int = 30,
) -> int:
    """
    Capture pose landmarks from webcam. Press 'q' to stop recording.

    Returns:
        Number of frames captured.
    """
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        raise ImportError("Install: pip install mediapipe opencv-contrib-python")

    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_id)
    frames = []
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("[webcam] Press 'q' to stop recording …")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = _extract_landmarks(results)
        frames.append(keypoints)

        # Draw landmarks for visual feedback
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        cv2.imshow("Sign Language — Recording (press q to stop)", frame)
        if cv2.waitKey(1000 // fps_limit) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

    arr = np.stack(frames, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, arr)
    print(f"[webcam] Saved {arr.shape[0]} frames → {output_path}")
    return len(frames)


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def batch_extract(
    video_dir: str,
    output_dir: str,
    extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
    **kwargs,
):
    """
    Extract poses from all videos in a directory.

    Creates output_dir/{basename}.npy for each video.
    """
    os.makedirs(output_dir, exist_ok=True)
    videos = [
        f for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]
    print(f"[batch_extract] {len(videos)} videos in {video_dir}")

    success = errors = 0
    for fname in sorted(videos):
        base = os.path.splitext(fname)[0]
        in_path  = os.path.join(video_dir, fname)
        out_path = os.path.join(output_dir, base + ".npy")
        if os.path.exists(out_path):
            print(f"  [skip] {out_path} already exists")
            continue
        try:
            extract_pose_from_video(in_path, out_path, **kwargs)
            success += 1
        except Exception as e:
            print(f"  [error] {fname}: {e}")
            errors += 1

    print(f"[batch_extract] Done. Success={success} Errors={errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe Holistic Pose Extractor")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video",     help="Input video file")
    g.add_argument("--video_dir", help="Input directory of videos")
    g.add_argument("--webcam",    action="store_true", help="Use webcam")

    p.add_argument("--output",     default=None, help="Output .npy file (single video/webcam)")
    p.add_argument("--output_dir", default="poses", help="Output directory (batch mode)")
    p.add_argument("--camera_id",  type=int, default=0)
    p.add_argument("--detection_conf",  type=float, default=0.5)
    p.add_argument("--tracking_conf",   type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.webcam:
        out = args.output or "poses/webcam.npy"
        extract_pose_from_webcam(out, camera_id=args.camera_id)

    elif args.video:
        out = args.output or os.path.splitext(args.video)[0] + ".npy"
        extract_pose_from_video(
            args.video, out,
            min_detection_confidence=args.detection_conf,
            min_tracking_confidence=args.tracking_conf,
        )

    elif args.video_dir:
        batch_extract(
            args.video_dir,
            args.output_dir,
            min_detection_confidence=args.detection_conf,
            min_tracking_confidence=args.tracking_conf,
        )
