"""Automated Unit Tests for Gesture Classification & Mathematical Calculator Logic."""

from __future__ import annotations
import unittest
import time
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gesture_engine import GestureEngine, GestureType, ClassifiedGesture
from src.calculator_logic import GestureCalculator, CalcState
from src.hand_detector import HandData


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MockHandData(HandData):
    """Generates mock hand states directly for deterministic unit testing."""
    def __init__(self, thumb: bool, index: bool, middle: bool, ring: bool, pinky: bool):
        self.finger_states = {
            "thumb": thumb,
            "index": index,
            "middle": middle,
            "ring": ring,
            "pinky": pinky,
        }
        self.extended_count = sum(1 for v in self.finger_states.values() if v)
        self.handedness = "Right"
        self.score = 0.99
        self.pixel_coords = [(100, 100)] * 21


class TestGestureEngine(unittest.TestCase):
    """Verifies that gestures map exactly to the user-specified rules."""

    def test_thumb_only_multiplies(self):
        """Rule: Thumb is shown initially -> Multiply (*)."""
        hand = MockHandData(thumb=True, index=False, middle=False, ring=False, pinky=False)
        gesture = GestureEngine.classify([hand])
        self.assertEqual(gesture.gesture_type, GestureType.OP_MULTIPLY)
        self.assertEqual(gesture.operator_symbol, "*")

    def test_pinky_only_subtracts(self):
        """Rule: Pinky is shown -> Subtract (-)."""
        hand = MockHandData(thumb=False, index=False, middle=False, ring=False, pinky=True)
        gesture = GestureEngine.classify([hand])
        self.assertEqual(gesture.gesture_type, GestureType.OP_SUBTRACT)
        self.assertEqual(gesture.operator_symbol, "-")

    def test_index_only_adds(self):
        """Rule: Index finger is shown -> Add (+)."""
        hand = MockHandData(thumb=False, index=True, middle=False, ring=False, pinky=False)
        gesture = GestureEngine.classify([hand])
        self.assertEqual(gesture.gesture_type, GestureType.OP_ADD)
        self.assertEqual(gesture.operator_symbol, "+")

    def test_index_and_middle_divides(self):
        """Rule: Any other two fingers shown side-by-side with index (Index+Middle) -> Divide (/)."""
        hand = MockHandData(thumb=False, index=True, middle=True, ring=False, pinky=False)
        gesture = GestureEngine.classify([hand])
        self.assertEqual(gesture.gesture_type, GestureType.OP_DIVIDE)
        self.assertEqual(gesture.operator_symbol, "/")

    def test_fist_resets(self):
        """Rule: Closed fist (0 fingers) maps to Reset candidate."""
        hand = MockHandData(thumb=False, index=False, middle=False, ring=False, pinky=False)
        gesture = GestureEngine.classify([hand])
        self.assertEqual(gesture.gesture_type, GestureType.RESET)
        self.assertEqual(gesture.value, 0)

    def test_finger_counts_for_numbers(self):
        """Verifies counts for 3, 4, and 5 fingers."""
        # 3 fingers (Index, Middle, Ring)
        hand3 = MockHandData(thumb=False, index=True, middle=True, ring=True, pinky=False)
        g3 = GestureEngine.classify([hand3])
        self.assertEqual(g3.gesture_type, GestureType.NUMBER)
        self.assertEqual(g3.value, 3)

        # 4 fingers (Index, Middle, Ring, Pinky)
        hand4 = MockHandData(thumb=False, index=True, middle=True, ring=True, pinky=True)
        g4 = GestureEngine.classify([hand4])
        self.assertEqual(g4.gesture_type, GestureType.NUMBER)
        self.assertEqual(g4.value, 4)

        # 5 fingers (Full hand)
        hand5 = MockHandData(thumb=True, index=True, middle=True, ring=True, pinky=True)
        g5 = GestureEngine.classify([hand5])
        self.assertEqual(g5.gesture_type, GestureType.NUMBER)
        self.assertEqual(g5.value, 5)


class TestCalculatorLogic(unittest.TestCase):
    """Verifies math evaluation, state transitions, and successive chaining."""

    def setUp(self):
        # Use zero hold time for instantaneous testing
        self.calc = GestureCalculator(hold_threshold_seconds=0.0, cooldown_seconds=0.0)

    def test_full_multiplication_workflow(self):
        """Tests the user's exact prompt scenario:
        Thumb shown initially (*), then first operand (4), then successive operand (3) -> 12, then (2) -> 24.
        """
        # Step 1: Show Thumb (*)
        g_op = ClassifiedGesture(GestureType.OP_MULTIPLY, operator_symbol="*")
        res1 = self.calc.update(g_op)
        self.assertEqual(self.calc.active_operator, "*")
        self.assertEqual(self.calc.state, CalcState.WAITING_FOR_FIRST_OPERAND)

        # Step 2: Show 4 fingers (First operand = 4)
        g_num1 = ClassifiedGesture(GestureType.NUMBER, value=4)
        res2 = self.calc.update(g_num1)
        self.assertEqual(self.calc.first_operand, 4.0)
        self.assertEqual(self.calc.state, CalcState.WAITING_FOR_NEXT_OPERAND)

        # Step 3: Show 3 fingers (Second operand = 3)
        g_num2 = ClassifiedGesture(GestureType.NUMBER, value=3)
        res3 = self.calc.update(g_num2)
        self.assertEqual(self.calc.running_result, 12.0)
        self.assertEqual(self.calc.state, CalcState.RESULT_DISPLAY)
        self.assertEqual(res3["expression"], "4 * 3 = 12.0")

        # Step 4: Successive chaining: Show 2 fingers -> (12 * 2 = 24)
        g_num3 = ClassifiedGesture(GestureType.NUMBER, value=2)
        res4 = self.calc.update(g_num3)
        self.assertEqual(self.calc.running_result, 24.0)
        self.assertEqual(res4["expression"], "12 * 2 = 24.0")

    def test_addition_workflow(self):
        """Index only (+) with 5 and 3 -> 8."""
        g_op = ClassifiedGesture(GestureType.OP_ADD, operator_symbol="+")
        self.calc.update(g_op)
        self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=5))
        res = self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=3))
        self.assertEqual(self.calc.running_result, 8.0)

    def test_subtraction_workflow(self):
        """Pinky only (-) with 5 and 2 -> 3."""
        g_op = ClassifiedGesture(GestureType.OP_SUBTRACT, operator_symbol="-")
        self.calc.update(g_op)
        self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=5))
        res = self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=2))
        self.assertEqual(self.calc.running_result, 3.0)

    def test_division_workflow_and_zero_protection(self):
        """Peace sign (/) with 8 and 2 -> 4; division by zero returns error without crashing."""
        g_op = ClassifiedGesture(GestureType.OP_DIVIDE, operator_symbol="/")
        self.calc.update(g_op)
        self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=8))
        res = self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=2))
        self.assertEqual(self.calc.running_result, 4.0)

        # Reset and test divide by zero
        self.calc.reset(play_audio=False)
        self.calc.update(g_op)
        self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=5))
        res_zero = self.calc.update(ClassifiedGesture(GestureType.NUMBER, value=0))
        self.assertIn("Division by Zero", self.calc.error_message)


if __name__ == "__main__":
    unittest.main()
