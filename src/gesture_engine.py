"""Gesture Classification Engine.
Translates detected hand finger states into mathematical operators and numerical operands
according to user-specified rules:
  - Thumb only         -> Multiply (*)
  - Index only         -> Add (+)
  - Pinky only         -> Subtract (-)
  - Index + Middle (V) -> Divide (/)
  - Finger Count       -> Numerical Operand (0 to 10)
  - Fist held          -> Reset / Clear (C)
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Tuple, Dict
from src.hand_detector import HandData


class GestureType(Enum):
    NONE = "NONE"
    OP_MULTIPLY = "*"
    OP_ADD = "+"
    OP_SUBTRACT = "-"
    OP_DIVIDE = "/"
    RESET = "RESET"
    NUMBER = "NUMBER"


class ClassifiedGesture:
    """Represents a classified gesture along with numerical and metadata properties."""
    def __init__(
        self,
        gesture_type: GestureType,
        value: Optional[int] = None,
        operator_symbol: Optional[str] = None,
        description: str = "",
        raw_finger_count: int = 0,
    ):
        self.gesture_type = gesture_type
        self.value = value
        self.operator_symbol = operator_symbol
        self.description = description
        self.raw_finger_count = raw_finger_count

    def __repr__(self) -> str:
        return f"<Gesture {self.gesture_type.name} val={self.value} op={self.operator_symbol} desc='{self.description}'>"


class GestureEngine:
    """Classifies hand states into operators and operands."""

    @staticmethod
    def classify(hands: List[HandData]) -> ClassifiedGesture:
        """Evaluates detected hands and returns the dominant classified gesture."""
        if not hands:
            return ClassifiedGesture(
                gesture_type=GestureType.NONE,
                description="No hands detected",
                raw_finger_count=0,
            )

        # Sum of extended fingers across all visible hands
        total_fingers = sum(h.extended_count for h in hands)

        # Primary hand (usually hand with highest detection score or first detected)
        primary_hand = hands[0]
        f = primary_hand.finger_states
        thumb, index, middle, ring, pinky = f["thumb"], f["index"], f["middle"], f["ring"], f["pinky"]
        count = primary_hand.extended_count

        # ----------------------------------------------------------------------
        # 1. SPECIFIC OPERATOR GESTURE RULES
        # ----------------------------------------------------------------------
        # Rule A: Thumb ONLY is extended -> MULTIPLICATION (*)
        if thumb and not index and not middle and not ring and not pinky:
            return ClassifiedGesture(
                gesture_type=GestureType.OP_MULTIPLY,
                operator_symbol="*",
                description="Thumb Gesture -> Multiply (*)",
                raw_finger_count=total_fingers,
            )

        # Rule B: Pinky ONLY is extended -> SUBTRACTION (-)
        if pinky and not thumb and not index and not middle and not ring:
            return ClassifiedGesture(
                gesture_type=GestureType.OP_SUBTRACT,
                operator_symbol="-",
                description="Pinky Gesture -> Subtract (-)",
                raw_finger_count=total_fingers,
            )

        # Rule C: Index ONLY is extended -> ADDITION (+)
        if index and not middle and not ring and not pinky and not thumb:
            return ClassifiedGesture(
                gesture_type=GestureType.OP_ADD,
                operator_symbol="+",
                description="Index Finger Gesture -> Add (+)",
                raw_finger_count=total_fingers,
            )

        # Rule D: Two fingers side-by-side with Index (Index + Middle, Peace / V sign) -> DIVISION (/)
        if index and middle and not ring and not pinky and not thumb:
            return ClassifiedGesture(
                gesture_type=GestureType.OP_DIVIDE,
                operator_symbol="/",
                description="Index + Middle (Peace) -> Divide (/)",
                raw_finger_count=total_fingers,
            )

        # Rule E: Closed Fist (0 fingers) -> Reset candidate
        if total_fingers == 0:
            return ClassifiedGesture(
                gesture_type=GestureType.RESET,
                value=0,
                description="Closed Fist -> Hold to Reset / Zero (0)",
                raw_finger_count=0,
            )

        # ----------------------------------------------------------------------
        # 2. GENERAL NUMBER GESTURE (Finger Count: 1 to 10)
        # ----------------------------------------------------------------------
        return ClassifiedGesture(
            gesture_type=GestureType.NUMBER,
            value=total_fingers,
            description=f"{total_fingers} Finger{'s' if total_fingers != 1 else ''} Shown",
            raw_finger_count=total_fingers,
        )
