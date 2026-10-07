"""UI Overlay Module.
Renders a modern, semi-transparent Sci-Fi / Cyberpunk HUD over OpenCV webcam frames:
  - Header with large active mathematical equation
  - Dynamic step-by-step guidance banner
  - Operator status cards with active glow
  - Circular progress ring for hold-to-confirm feedback
  - Finger telemetry badge
  - Reset indicator and shortcut legend
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

# Color Palette (BGR)
BG_DARK = (18, 22, 28)
ACCENT_CYAN = (255, 220, 0)
ACCENT_GREEN = (50, 220, 50)
ACCENT_ORANGE = (0, 140, 255)
ACCENT_MAGENTA = (220, 50, 220)
ACCENT_RED = (50, 50, 240)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_DIM = (160, 160, 160)
COLOR_CARD_BG = (35, 40, 50)
COLOR_ACTIVE_CARD = (80, 140, 40)


def draw_alpha_rect(
    frame: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.65,
) -> None:
    """Draws a semi-transparent filled rectangle using alpha blending."""
    x1, y1 = max(0, top_left[0]), max(0, top_left[1])
    x2, y2 = min(frame.shape[1], bottom_right[0]), min(frame.shape[0], bottom_right[1])
    if x1 >= x2 or y1 >= y2:
        return
    sub_img = frame[y1:y2, x1:x2]
    rect = np.full(sub_img.shape, color, dtype=np.uint8)
    blended = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 0)
    frame[y1:y2, x1:x2] = blended


def draw_circular_progress(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    progress: float,
    color: Tuple[int, int, int] = ACCENT_GREEN,
    thickness: int = 4,
) -> None:
    """Draws an animated circular countdown/confirmation ring."""
    # Background circle (dim gray)
    cv2.circle(frame, center, radius, (80, 80, 80), thickness, cv2.LINE_AA)
    if progress > 0.0:
        angle_sweep = int(progress * 360)
        # Draw arc from -90 degrees (top) clockwise
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            0,
            angle_sweep,
            color,
            thickness + 2,
            cv2.LINE_AA,
        )


class UIOverlayRenderer:
    """Renders the comprehensive HUD onto camera frames."""

    @staticmethod
    def render(
        frame: np.ndarray,
        calc_status: Dict[str, Any],
        fps: float = 0.0,
        show_telemetry: bool = True,
    ) -> np.ndarray:
        h, w, _ = frame.shape

        # ----------------------------------------------------------------------
        # 1. TOP HEADER BANNER (Semi-transparent dashboard)
        # ----------------------------------------------------------------------
        header_h = 110
        draw_alpha_rect(frame, (0, 0), (w, header_h), BG_DARK, alpha=0.75)
        # Subtle glowing bottom accent border
        cv2.line(frame, (0, header_h), (w, header_h), (60, 70, 85), 2, cv2.LINE_AA)

        # Title & Subtitle
        cv2.putText(
            frame,
            "FINMATH | GESTURE CALCULATOR",
            (20, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            ACCENT_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Active Expression Display (Large Centered Formula)
        expr_text = calc_status.get("expression", "0")
        is_result_state = calc_status.get("state") == "RESULT_DISPLAY"
        expr_color = ACCENT_GREEN if is_result_state else (255, 255, 255)
        cv2.putText(
            frame,
            f"EQUATION: {expr_text}",
            (20, 75),
            cv2.FONT_HERSHEY_DUPLEX,
            1.05,
            expr_color,
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------------------------
        # 2. OPERATOR BADGES (Top Right)
        # ----------------------------------------------------------------------
        active_op = calc_status.get("operator")
        op_cards = [
            ("Thumb: *", "*", (220, 50, 220)),
            ("Index: +", "+", (255, 220, 0)),
            ("Pinky: -", "-", (0, 140, 255)),
            ("Peace: /", "/", (50, 220, 50)),
        ]
        card_w, card_h = 92, 42
        start_x = w - (card_w * 4) - 30
        for i, (label, symbol, color) in enumerate(op_cards):
            cx = start_x + (i * (card_w + 6))
            cy = 18
            is_active = (active_op == symbol)
            bg_c = color if is_active else COLOR_CARD_BG
            alpha_val = 0.85 if is_active else 0.45

            draw_alpha_rect(frame, (cx, cy), (cx + card_w, cy + card_h), bg_c, alpha=alpha_val)
            border_c = (255, 255, 255) if is_active else (80, 90, 105)
            cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), border_c, 1, cv2.LINE_AA)

            text_color = (255, 255, 255) if is_active else COLOR_TEXT_DIM
            cv2.putText(
                frame,
                label,
                (cx + 8, cy + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                text_color,
                1,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------------------------
        # 3. DYNAMIC INSTRUCTION BANNER (Below Header)
        # ----------------------------------------------------------------------
        guide_y = header_h + 35
        state = calc_status.get("state", "")
        message = calc_status.get("message", "")

        if state == "WAITING_FOR_OPERATOR":
            guide_title = "STEP 1: SELECT OPERATOR"
            guide_detail = "Show Thumb (*), Index (+), Pinky (-), or Index+Middle (/)"
            guide_color = ACCENT_CYAN
        elif state == "WAITING_FOR_FIRST_OPERAND":
            guide_title = "STEP 2: SHOW FIRST NUMBER"
            guide_detail = "Hold fingers (1 to 5) steady to set first operand"
            guide_color = ACCENT_ORANGE
        elif state == "WAITING_FOR_NEXT_OPERAND":
            guide_title = "STEP 3: SHOW NEXT NUMBER"
            guide_detail = "Hold fingers (1 to 5) steady to compute result"
            guide_color = ACCENT_MAGENTA
        elif state == "RESULT_DISPLAY":
            guide_title = "RESULT READY"
            guide_detail = "Show fingers to multiply/add again, switch operator, or Fist to clear"
            guide_color = ACCENT_GREEN
        else:
            guide_title = "STATUS"
            guide_detail = message
            guide_color = COLOR_TEXT_WHITE

        # Draw guidance badge
        badge_w = min(680, w - 40)
        draw_alpha_rect(frame, (20, header_h + 8), (20 + badge_w, guide_y + 10), BG_DARK, alpha=0.7)
        cv2.putText(
            frame,
            f"{guide_title}: {guide_detail}",
            (30, guide_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            guide_color,
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------------------------
        # 4. HOLD-TO-CONFIRM PROGRESS RING (Center Top)
        # ----------------------------------------------------------------------
        hold_prog = calc_status.get("hold_progress", 0.0)
        if hold_prog > 0.05:
            center_x, center_y = w // 2, header_h + 90
            ring_color = ACCENT_GREEN if hold_prog >= 0.9 else ACCENT_CYAN
            draw_circular_progress(frame, (center_x, center_y), radius=32, progress=hold_prog, color=ring_color, thickness=4)
            pct_text = f"{int(hold_prog * 100)}%"
            cv2.putText(
                frame,
                pct_text,
                (center_x - 16, center_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "HOLD STEADY",
                (center_x - 42, center_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                ring_color,
                1,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------------------------
        # 5. RESET PROGRESS BAR (If Fist is held)
        # ----------------------------------------------------------------------
        reset_prog = calc_status.get("reset_progress", 0.0)
        if reset_prog > 0.05:
            bar_w, bar_h = 320, 22
            bar_x = (w - bar_w) // 2
            bar_y = h // 2
            draw_alpha_rect(frame, (bar_x - 10, bar_y - 25), (bar_x + bar_w + 10, bar_y + bar_h + 10), BG_DARK, alpha=0.85)
            cv2.putText(
                frame,
                "RESETTING / CLEARING...",
                (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                ACCENT_RED,
                1,
                cv2.LINE_AA,
            )
            # Filled portion
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * reset_prog)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ACCENT_RED, -1)

        # ----------------------------------------------------------------------
        # 6. BOTTOM TELEMETRY & CONTROLS FOOTER
        # ----------------------------------------------------------------------
        footer_h = 42
        draw_alpha_rect(frame, (0, h - footer_h), (w, h), BG_DARK, alpha=0.80)
        cv2.line(frame, (0, h - footer_h), (w, h - footer_h), (50, 60, 75), 1, cv2.LINE_AA)

        # Keyboard shortcuts
        shortcuts_text = "[Q] Quit  |  [C] Reset / Clear  |  [S] Screenshot  |  [H] Toggle HUD"
        cv2.putText(
            frame,
            shortcuts_text,
            (20, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

        # FPS & Performance Badge
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (w - 110, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            ACCENT_CYAN,
            1,
            cv2.LINE_AA,
        )

        return frame
