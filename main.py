"""Main Application Entry Point.
FinMath: Real-Time Hand Gesture Computer Vision Math Calculator.
Runs webcam feed with MediaPipe tracking, gesture classification, and interactive Sci-Fi HUD.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from typing import Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_detector import HandDetector, HandData
from src.gesture_engine import GestureEngine, ClassifiedGesture, GestureType
from src.calculator_logic import GestureCalculator
from src.ui_overlay import UIOverlayRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="FinMath: AI Hand Gesture Math Calculator")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Target frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target frame height (default: 720)")
    parser.add_argument("--demo", action="store_true", help="Run in automated simulation/demo mode")
    parser.add_argument("--hold-time", type=float, default=0.85, help="Seconds to hold gesture to confirm (default: 0.85)")
    return parser.parse_args()


class SyntheticHandSimulator:
    """Simulates realistic hand gestures for headless testing and demos."""
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.start_time = time.time()

    def get_simulated_gesture(self) -> Tuple[ClassifiedGesture, str]:
        elapsed = time.time() - self.start_time
        cycle = elapsed % 18.0

        if cycle < 3.0:
            # Thumb Only -> Multiply
            g = ClassifiedGesture(
                gesture_type=GestureType.OP_MULTIPLY,
                operator_symbol="*",
                description="Simulated: Thumb (Multiply *)",
                raw_finger_count=1,
            )
            return g, "Thumb Extended (Multiply)"
        elif cycle < 6.5:
            # 4 Fingers -> Operand 1
            g = ClassifiedGesture(
                gesture_type=GestureType.NUMBER,
                value=4,
                description="Simulated: 4 Fingers (Four)",
                raw_finger_count=4,
            )
            return g, "4 Fingers (Operand 1 = 4)"
        elif cycle < 10.0:
            # 3 Fingers -> Operand 2
            g = ClassifiedGesture(
                gesture_type=GestureType.NUMBER,
                value=3,
                description="Simulated: 3 Fingers (Three)",
                raw_finger_count=3,
            )
            return g, "3 Fingers (Operand 2 = 3 -> Result: 4 * 3 = 12)"
        elif cycle < 13.5:
            # 2 Fingers -> Successive Chaining
            g = ClassifiedGesture(
                gesture_type=GestureType.NUMBER,
                value=2,
                description="Simulated: 2 Fingers (Successive)",
                raw_finger_count=2,
            )
            return g, "2 Fingers (Successive: 12 * 2 = 24)"
        else:
            # Fist -> Reset
            g = ClassifiedGesture(
                gesture_type=GestureType.RESET,
                value=0,
                description="Simulated: Closed Fist (Reset)",
                raw_finger_count=0,
            )
            return g, "Closed Fist (Hold to Reset)"


def run_application():
    args = parse_args()
    print("=" * 70)
    print(" 🚀 FinMath - OpenCV Real-Time Gesture Calculator")
    print("=" * 70)
    print(" • Thumb only         -> MULTIPLY (*)")
    print(" • Index only         -> ADD (+)")
    print(" • Pinky only         -> SUBTRACT (-)")
    print(" • Index + Middle (V) -> DIVIDE (/)")
    print(" • Finger Count       -> Numerical Value (1 to 5, or up to 10)")
    print(" • Closed Fist        -> Hold 1.5s to Clear / Reset")
    print("-" * 70)
    print(" Controls: [Q] Quit  |  [C] Reset  |  [S] Screenshot  |  [H] Toggle HUD")
    print("=" * 70)

    # Initialize components
    detector = HandDetector()
    calculator = GestureCalculator(hold_threshold_seconds=args.hold_time)
    simulator = SyntheticHandSimulator(args.width, args.height) if args.demo else None

    # Open Camera (if not demo mode)
    cap = None
    if not args.demo:
        print(f"[FinMath] Connecting to Camera (ID: {args.camera})...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"⚠️ Warning: Could not open camera {args.camera}. Switching to Demo Simulation Mode!")
            simulator = SyntheticHandSimulator(args.width, args.height)
        else:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            except Exception:
                pass

    window_name = "FinMath | AI Gesture Calculator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    show_hud = True
    prev_time = time.time()
    fps = 30.0

    os.makedirs(os.path.join(PROJECT_ROOT, "screenshots"), exist_ok=True)

    try:
        while True:
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = (0.9 * fps) + (0.1 * (1.0 / dt))

            if simulator is not None:
                # Synthetic canvas
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                # Subtle grid pattern
                for y in range(0, args.height, 40):
                    cv2.line(frame, (0, y), (args.width, y), (24, 28, 34), 1)
                for x in range(0, args.width, 40):
                    cv2.line(frame, (x, 0), (x, args.height), (24, 28, 34), 1)

                classified, sim_desc = simulator.get_simulated_gesture()
                calc_status = calculator.update(classified)

                # Render simulation banner
                cv2.putText(
                    frame,
                    f"[SIMULATION DEMO] {sim_desc}",
                    (args.width // 2 - 250, args.height // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab camera frame. Exiting...")
                    break

                # Flip horizontally for natural mirror feel
                frame = cv2.flip(frame, 1)

                # Detect hands and finger states
                hands = detector.process(frame)
                detector.draw_skeleton(frame, hands)

                # Classify dominant gesture
                classified = GestureEngine.classify(hands)

                # Update calculator state machine
                calc_status = calculator.update(classified)

            # Render Sci-Fi HUD Overlay
            if show_hud:
                frame = UIOverlayRenderer.render(frame, calc_status, fps=fps)

            # Display frame
            cv2.imshow(window_name, frame)

            # Handle user keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                print("[FinMath] Terminating...")
                break
            elif key in (ord('c'), ord('C'), ord('r'), ord('R')):
                calculator.reset()
                print("[FinMath] Calculator manually reset!")
            elif key in (ord('h'), ord('H')):
                show_hud = not show_hud
            elif key in (ord('s'), ord('S')):
                shot_path = os.path.join(PROJECT_ROOT, "screenshots", f"capture_{int(time.time())}.png")
                cv2.imwrite(shot_path, frame)
                print(f"[FinMath] Saved screenshot to {shot_path}")

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[FinMath] Shutdown complete.")


if __name__ == "__main__":
    run_application()
