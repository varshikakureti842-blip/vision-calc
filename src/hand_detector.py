"""Hand Detector Module
Powered by MediaPipe HandLandmarker & OpenCV.
Detects 21 3D landmarks, calculates finger extension states, and renders neon HUD skeletons.
"""

from __future__ import annotations
import os
import math
import urllib.request
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Default path for the pre-trained MediaPipe hand landmarker model
DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

# Hand bone connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

# Color palette for neon joints (BGR format)
COLOR_THUMB = (255, 0, 255)   # Magenta
COLOR_INDEX = (255, 255, 0)   # Cyan
COLOR_MIDDLE = (0, 255, 0)    # Lime Green
COLOR_RING = (0, 215, 255)    # Amber / Gold
COLOR_PINKY = (0, 128, 255)   # Orange
COLOR_WRIST = (200, 200, 200) # Silver
COLOR_BONE = (80, 80, 80)     # Dark Gray bone lines


class HandData:
    """Encapsulates detected hand information for a single hand."""
    def __init__(self, landmarks: List[Any], handedness: str, score: float, img_w: int, img_h: int):
        self.raw_landmarks = landmarks
        self.handedness = handedness  # 'Left' or 'Right'
        self.score = score
        self.img_w = img_w
        self.img_h = img_h

        # Pixel coordinates: list of (x, y)
        self.pixel_coords: List[Tuple[int, int]] = [
            (int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks
        ]

        # Calculate finger extension states
        self.finger_states: Dict[str, bool] = self._compute_finger_states()
        self.extended_count: int = sum(1 for v in self.finger_states.values() if v)

    def _dist_2d(self, idx1: int, idx2: int) -> float:
        x1, y1 = self.raw_landmarks[idx1].x, self.raw_landmarks[idx1].y
        x2, y2 = self.raw_landmarks[idx2].x, self.raw_landmarks[idx2].y
        return math.hypot(x1 - x2, y1 - y2)

    def _compute_finger_states(self) -> Dict[str, bool]:
        """Calculates boolean extension state for each of the 5 fingers.
        A finger is considered UP/Extended if its tip is extended away from knuckles and wrist.
        """
        lms = self.raw_landmarks
        wrist = 0

        # --- FINGERS 1-4: Index, Middle, Ring, Pinky ---
        # Tip must be higher than PIP (smaller y) AND distance from wrist must exceed PIP
        def is_regular_finger_up(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            # Vertical height check (in image space, top is y=0)
            is_above_pip = lms[tip_idx].y < lms[pip_idx].y
            # Radial distance check (prevents false positives when hand is tilted)
            dist_tip = self._dist_2d(wrist, tip_idx)
            dist_pip = self._dist_2d(wrist, pip_idx)
            dist_mcp = self._dist_2d(wrist, mcp_idx)
            is_extended_radially = dist_tip > (dist_pip * 1.08) and dist_tip > (dist_mcp * 1.25)
            return is_above_pip and is_extended_radially

        index_up = is_regular_finger_up(8, 6, 5)
        middle_up = is_regular_finger_up(12, 10, 9)
        ring_up = is_regular_finger_up(16, 14, 13)
        pinky_up = is_regular_finger_up(20, 18, 17)

        # --- THUMB ---
        # Thumb moves laterally and vertically.
        # Check 1: Thumbs-up posture (tip higher than IP and MCP)
        thumb_vertical = (lms[4].y < lms[3].y) and (lms[4].y < lms[2].y)

        # Check 2: Lateral extension away from palm knuckles (landmark 5 and 17)
        dist_thumb_tip_wrist = self._dist_2d(0, 4)
        dist_thumb_mcp_wrist = self._dist_2d(0, 2)
        dist_thumb_to_index_mcp = self._dist_2d(4, 5)
        dist_ip_to_index_mcp = self._dist_2d(3, 5)
        
        # When thumb is tucked into fist, tip is close to knuckles
        thumb_lateral = (
            dist_thumb_tip_wrist > (dist_thumb_mcp_wrist * 1.15) and
            dist_thumb_to_index_mcp > (dist_ip_to_index_mcp * 1.05)
        )

        thumb_up = thumb_vertical or thumb_lateral

        return {
            "thumb": bool(thumb_up),
            "index": bool(index_up),
            "middle": bool(middle_up),
            "ring": bool(ring_up),
            "pinky": bool(pinky_up),
        }


class HandDetector:
    """Wrapper around MediaPipe HandLandmarker for real-time tracking."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.60,
    ):
        self.model_path = model_path
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        """Ensures the hand landmarker .task model file is downloaded locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[HandDetector] Downloading MediaPipe model to {self.model_path}...")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, self.model_path)
            print("[HandDetector] Model download complete.")

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        """Processes an OpenCV BGR frame and returns a list of HandData objects."""
        h, w, _ = frame_bgr.shape
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.detector.detect(mp_image)
        hands_list: List[HandData] = []

        if not result.hand_landmarks:
            return hands_list

        for idx, landmarks in enumerate(result.hand_landmarks):
            handedness_str = "Right"
            score = 1.0
            if result.handedness and idx < len(result.handedness):
                handedness_cat = result.handedness[idx][0]
                handedness_str = handedness_cat.category_name
                score = handedness_cat.score

            hand_data = HandData(
                landmarks=landmarks,
                handedness=handedness_str,
                score=score,
                img_w=w,
                img_h=h,
            )
            hands_list.append(hand_data)

        return hands_list

    def draw_skeleton(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        show_labels: bool = True,
    ) -> np.ndarray:
        """Renders high-tech neon bone connections and glowing fingertip joints."""
        for hand in hands:
            coords = hand.pixel_coords
            states = hand.finger_states

            # 1. Draw connecting bones
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                pt1 = coords[p1_idx]
                pt2 = coords[p2_idx]
                cv2.line(frame, pt1, pt2, COLOR_BONE, 2, cv2.LINE_AA)

            # 2. Draw finger joints with custom colors
            tip_map = {
                4: ("T", COLOR_THUMB, states["thumb"]),
                8: ("I", COLOR_INDEX, states["index"]),
                12: ("M", COLOR_MIDDLE, states["middle"]),
                16: ("R", COLOR_RING, states["ring"]),
                20: ("P", COLOR_PINKY, states["pinky"]),
            }

            # Draw non-tip knuckle dots
            for idx, pt in enumerate(coords):
                if idx not in tip_map:
                    cv2.circle(frame, pt, 4, (180, 180, 180), -1, cv2.LINE_AA)

            # Draw glowing fingertips with state glow
            for tip_idx, (letter, color, is_up) in tip_map.items():
                tip_pt = coords[tip_idx]
                if is_up:
                    # Outer neon halo
                    cv2.circle(frame, tip_pt, 14, color, 2, cv2.LINE_AA)
                    # Filled vibrant center
                    cv2.circle(frame, tip_pt, 8, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, tip_pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
                else:
                    # Dim closed state
                    cv2.circle(frame, tip_pt, 6, (70, 70, 70), -1, cv2.LINE_AA)

                if show_labels:
                    cv2.putText(
                        frame,
                        letter,
                        (tip_pt[0] - 5, tip_pt[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color if is_up else (140, 140, 140),
                        1,
                        cv2.LINE_AA,
                    )

            # Draw Hand label & Count badge above wrist
            wrist_pt = coords[0]
            hand_tag = f"{hand.handedness}: {hand.extended_count} Up"
            cv2.putText(
                frame,
                hand_tag,
                (wrist_pt[0] - 40, wrist_pt[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame
