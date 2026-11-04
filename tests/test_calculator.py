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


class TestFingerStateGeometry(unittest.TestCase):
    """Verifies geometric calculations in HandData._compute_finger_states.
    Specifically tests that closed thumbs are correctly recognized as False across all postures.
    """

    def _make_hand(self, tip_x: float, tip_y: float, ip_x: float, ip_y: float, other_up: bool) -> HandData:
        lms = [MockLandmark(0.5, 0.5) for _ in range(21)]
        lms[0] = MockLandmark(0.50, 0.82) # Wrist
        lms[1] = MockLandmark(0.46, 0.73) # Thumb CMC
        lms[2] = MockLandmark(0.42, 0.65) # Thumb MCP
        lms[3] = MockLandmark(ip_x, ip_y)   # Thumb IP
        lms[4] = MockLandmark(tip_x, tip_y) # Thumb TIP

        # Index finger
        lms[5] = MockLandmark(0.43, 0.52)
        lms[6] = MockLandmark(0.43, 0.40 if other_up else 0.58)
        lms[7] = MockLandmark(0.43, 0.30 if other_up else 0.62)
        lms[8] = MockLandmark(0.43, 0.20 if other_up else 0.65)

        # Middle finger
        lms[9] = MockLandmark(0.49, 0.50)
        lms[10] = MockLandmark(0.49, 0.38 if other_up else 0.58)
        lms[11] = MockLandmark(0.49, 0.28 if other_up else 0.62)
        lms[12] = MockLandmark(0.49, 0.18 if other_up else 0.65)

        # Ring finger
        lms[13] = MockLandmark(0.55, 0.52)
        lms[14] = MockLandmark(0.55, 0.40 if other_up else 0.58)
        lms[15] = MockLandmark(0.55, 0.30 if other_up else 0.62)
        lms[16] = MockLandmark(0.55, 0.22 if other_up else 0.65)

        # Pinky finger
        lms[17] = MockLandmark(0.60, 0.56)
        lms[18] = MockLandmark(0.60, 0.45 if other_up else 0.60)
        lms[19] = MockLandmark(0.60, 0.36 if other_up else 0.64)
        lms[20] = MockLandmark(0.60, 0.28 if other_up else 0.67)

        return HandData(lms, "Right", 0.99, 1280, 720)

    def test_closed_fist_thumb_is_closed(self):
        """In a closed fist, thumb is curled over knuckles and must be False."""
        hand = self._make_hand(tip_x=0.45, tip_y=0.54, ip_x=0.45, ip_y=0.60, other_up=False)
        self.assertFalse(hand.finger_states["thumb"])
        self.assertEqual(hand.extended_count, 0)

    def test_thumbs_up_thumb_is_open(self):
        """In a thumbs up, thumb is extended upwards and must be True."""
        hand = self._make_hand(tip_x=0.41, tip_y=0.38, ip_x=0.42, ip_y=0.50, other_up=False)
        self.assertTrue(hand.finger_states["thumb"])
        self.assertEqual(hand.extended_count, 1)

    def test_four_fingers_folded_thumb_is_closed(self):
        """When showing 4 fingers, thumb is folded across palm and must be False."""
        hand = self._make_hand(tip_x=0.52, tip_y=0.60, ip_x=0.47, ip_y=0.62, other_up=True)
        self.assertFalse(hand.finger_states["thumb"])
        self.assertEqual(hand.extended_count, 4)

    def test_five_fingers_thumb_is_open(self):
        """When showing 5 fingers, thumb is abducted laterally and must be True."""
        hand = self._make_hand(tip_x=0.26, tip_y=0.58, ip_x=0.34, ip_y=0.60, other_up=True)
        self.assertTrue(hand.finger_states["thumb"])
        self.assertEqual(hand.extended_count, 5)


if __name__ == "__main__":
    unittest.main()
