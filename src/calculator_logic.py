"""Calculator Logic and State Machine Engine.
Implements the multi-step gesture workflow:
  1. Operator selection (Thumb: *, Index: +, Pinky: -, Peace: /)
  2. First operand input
  3. Second / successive operands
  4. Out-of-sample chained evaluation & state resets
Includes temporal smoothing, hold-to-confirm progress, debounce cooldown, and audio feedback.
"""

from __future__ import annotations
import time
import threading
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from src.gesture_engine import GestureType, ClassifiedGesture

try:
    import winsound
    def _play_beep(freq: int, duration_ms: int):
        threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()
except ImportError:
    def _play_beep(freq: int, duration_ms: int):
        pass


class CalcState(Enum):
    WAITING_FOR_OPERATOR = "WAITING_FOR_OPERATOR"
    WAITING_FOR_FIRST_OPERAND = "WAITING_FOR_FIRST_OPERAND"
    WAITING_FOR_NEXT_OPERAND = "WAITING_FOR_NEXT_OPERAND"
    RESULT_DISPLAY = "RESULT_DISPLAY"


class CalculationRecord:
    """Stores a single evaluation equation."""
    def __init__(self, expression: str, result: float):
        self.expression = expression
        self.result = result
        self.timestamp = time.strftime("%H:%M:%S")

    def __repr__(self) -> str:
        return f"{self.expression} = {self.result}"


class GestureCalculator:
    """State machine governing math execution and gesture confirmation."""

    def __init__(
        self,
        hold_threshold_seconds: float = 0.9,
        cooldown_seconds: float = 0.8,
        reset_hold_seconds: float = 1.4,
    ):
        self.hold_threshold = hold_threshold_seconds
        self.cooldown_duration = cooldown_seconds
        self.reset_hold_threshold = reset_hold_seconds

        # Calculator memory
        self.state = CalcState.WAITING_FOR_OPERATOR
        self.active_operator: Optional[str] = None
        self.first_operand: Optional[float] = None
        self.current_operand: Optional[float] = None
        self.running_result: Optional[float] = None
        self.error_message: Optional[str] = None
        self.history: List[CalculationRecord] = []

        # Temporal stability tracking
        self.candidate_key: Optional[str] = None
        self.candidate_start_time: float = 0.0
        self.hold_progress: float = 0.0  # 0.0 to 1.0
        self.cooldown_until: float = 0.0

        # Reset gesture hold tracking
        self.reset_start_time: float = 0.0
        self.reset_progress: float = 0.0

    def reset(self, play_audio: bool = True):
        """Clears all registers and resets to initial operator selection state."""
        self.state = CalcState.WAITING_FOR_OPERATOR
        self.active_operator = None
        self.first_operand = None
        self.current_operand = None
        self.running_result = None
        self.error_message = None
        self.candidate_key = None
        self.candidate_start_time = 0.0
        self.hold_progress = 0.0
        self.reset_start_time = 0.0
        self.reset_progress = 0.0
        if play_audio:
            _play_beep(450, 200)

    def _get_gesture_key(self, g: ClassifiedGesture) -> Optional[str]:
        """Converts gesture to a unique stable key for temporal debounce."""
        if g.gesture_type in (GestureType.OP_MULTIPLY, GestureType.OP_ADD, GestureType.OP_SUBTRACT, GestureType.OP_DIVIDE):
            return f"OP_{g.operator_symbol}"
        elif g.gesture_type == GestureType.NUMBER:
            return f"NUM_{g.value}"
        elif g.gesture_type == GestureType.RESET:
            return "RESET"
        return None

    def update(self, classified: ClassifiedGesture) -> Dict[str, Any]:
        """Main frame-by-frame update step.
        Returns visual feedback dictionary for HUD rendering.
        """
        now = time.time()
        key = self._get_gesture_key(classified)

        # ----------------------------------------------------------------------
        # 1. HOLD-TO-RESET MONITOR (Closed Fist or 0 fingers held)
        # ----------------------------------------------------------------------
        if key == "RESET" and (self.first_operand is not None or self.active_operator is not None):
            if self.reset_start_time == 0.0:
                self.reset_start_time = now
            elapsed_reset = now - self.reset_start_time
            self.reset_progress = min(1.0, elapsed_reset / self.reset_hold_threshold)
            if self.reset_progress >= 1.0:
                self.reset()
                self.cooldown_until = now + self.cooldown_duration
                return self._status_payload("Calculator Reset to Zero!", is_confirmed=True)
        else:
            self.reset_start_time = 0.0
            self.reset_progress = 0.0

        # ----------------------------------------------------------------------
        # 2. COOLDOWN GATE (Prevents duplicate continuous inputs)
        # ----------------------------------------------------------------------
        if now < self.cooldown_until:
            self.candidate_key = None
            self.candidate_start_time = 0.0
            self.hold_progress = 0.0
            return self._status_payload("Cooldown...", is_confirmed=False)

        # ----------------------------------------------------------------------
        # 3. GESTURE TEMPORAL SMOOTHING & HOLD ACCUMULATION
        # ----------------------------------------------------------------------
        if key is None or key == "RESET":
            self.candidate_key = None
            self.candidate_start_time = 0.0
            self.hold_progress = 0.0
            return self._status_payload("Awaiting steady gesture...", is_confirmed=False)

        # If hold threshold is 0 (e.g., unit tests or instant response mode), commit immediately
        if self.hold_threshold <= 0.0:
            confirmed_action = self._commit_action(classified)
            self.cooldown_until = now + self.cooldown_duration
            self.candidate_key = None
            self.candidate_start_time = 0.0
            self.hold_progress = 0.0
            return self._status_payload(confirmed_action, is_confirmed=True)

        if key != self.candidate_key:
            # New gesture detected; start timer
            self.candidate_key = key
            self.candidate_start_time = now
            self.hold_progress = 0.0
        else:
            # Same gesture held steady
            elapsed = now - self.candidate_start_time
            self.hold_progress = min(1.0, elapsed / self.hold_threshold)

            # If held for threshold, commit state transition!
            if self.hold_progress >= 1.0:
                confirmed_action = self._commit_action(classified)
                self.cooldown_until = now + self.cooldown_duration
                self.candidate_key = None
                self.candidate_start_time = 0.0
                self.hold_progress = 0.0
                return self._status_payload(confirmed_action, is_confirmed=True)

        return self._status_payload(f"Holding: {classified.description}", is_confirmed=False)

    def _commit_action(self, g: ClassifiedGesture) -> str:
        """Executes the action once gesture hold threshold is achieved."""
        # --- A. OPERATOR SELECTION GESTURES ---
        if g.gesture_type in (GestureType.OP_MULTIPLY, GestureType.OP_ADD, GestureType.OP_SUBTRACT, GestureType.OP_DIVIDE):
            op = g.operator_symbol
            self.active_operator = op
            _play_beep(650, 180)

            if self.running_result is not None:
                # Chaining from existing result
                self.first_operand = self.running_result
                self.state = CalcState.WAITING_FOR_NEXT_OPERAND
                return f"Chained: {self.running_result} {op} ?"
            elif self.first_operand is not None:
                self.state = CalcState.WAITING_FOR_NEXT_OPERAND
                return f"Operator set to: {op}"
            else:
                # Operator chosen first (as user described: thumb shown initially)
                self.state = CalcState.WAITING_FOR_FIRST_OPERAND
                return f"Operator set to: {op} -> Show First Number"

        # --- B. NUMBER SELECTION GESTURES ---
        elif g.gesture_type == GestureType.NUMBER and g.value is not None:
            val = float(g.value)
            _play_beep(880, 180)

            # State 1: We were waiting for first operand
            if self.state in (CalcState.WAITING_FOR_FIRST_OPERAND, CalcState.WAITING_FOR_OPERATOR):
                self.first_operand = val
                if self.active_operator is not None:
                    self.state = CalcState.WAITING_FOR_NEXT_OPERAND
                    return f"First Number: {int(val)} -> Show Next Number"
                else:
                    self.state = CalcState.WAITING_FOR_OPERATOR
                    return f"First Number: {int(val)} -> Show Operator"

            # State 2: We were waiting for second / successive operand
            elif self.state in (CalcState.WAITING_FOR_NEXT_OPERAND, CalcState.RESULT_DISPLAY):
                second_val = val
                calc_res, err = self._evaluate_math(self.first_operand or 0.0, self.active_operator or "*", second_val)

                if err:
                    self.error_message = err
                    _play_beep(300, 300)
                    return f"Error: {err}"
                else:
                    self.error_message = None
                    self.current_operand = second_val
                    self.running_result = calc_res
                    expr = f"{int(self.first_operand) if self.first_operand.is_integer() else self.first_operand} {self.active_operator} {int(second_val)}"
                    self.history.append(CalculationRecord(expr, calc_res))
                    # Pre-load running result for successive operations
                    self.first_operand = calc_res
                    self.state = CalcState.RESULT_DISPLAY
                    _play_beep(1200, 250)
                    return f"Evaluated: {expr} = {calc_res}"

        return "Gesture processed"

    def _evaluate_math(self, a: float, op: str, b: float) -> Tuple[Optional[float], Optional[str]]:
        """Calculates standard arithmetic with division-by-zero protection."""
        if op == "*":
            res = a * b
        elif op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        elif op == "/":
            if b == 0:
                return None, "Division by Zero!"
            res = a / b
        else:
            return None, f"Unknown Operator: {op}"

        # Clean round if practically integer
        if abs(res - round(res)) < 1e-6:
            res = float(round(res))
        else:
            res = round(res, 2)
        return res, None

    def _status_payload(self, message: str, is_confirmed: bool) -> Dict[str, Any]:
        """Prepares comprehensive status snapshot for UI rendering."""
        # Format current active expression
        if self.running_result is not None and self.state == CalcState.RESULT_DISPLAY:
            expr_str = f"{self.history[-1].expression} = {self.running_result}" if self.history else str(self.running_result)
        elif self.first_operand is not None and self.active_operator is not None:
            first_fmt = int(self.first_operand) if self.first_operand.is_integer() else self.first_operand
            expr_str = f"{first_fmt} {self.active_operator} ?"
        elif self.active_operator is not None:
            expr_str = f"? {self.active_operator} ?"
        elif self.first_operand is not None:
            first_fmt = int(self.first_operand) if self.first_operand.is_integer() else self.first_operand
            expr_str = f"{first_fmt}"
        else:
            expr_str = "0"

        return {
            "state": self.state.value,
            "operator": self.active_operator,
            "first_operand": self.first_operand,
            "running_result": self.running_result,
            "expression": expr_str,
            "message": message,
            "hold_progress": self.hold_progress,
            "reset_progress": self.reset_progress,
            "is_confirmed": is_confirmed,
            "error": self.error_message,
            "history_count": len(self.history),
        }
