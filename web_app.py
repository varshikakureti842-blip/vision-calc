"""FinMath: Streamlit Web Terminal for Gesture Computer Vision Calculator.
Enables real-time browser access to the webcam math gesture recognition engine.
"""

from __future__ import annotations
import os
import sys
import time
import cv2
import numpy as np
import streamlit as st
import pandas as pd

# Path configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_detector import HandDetector
from src.gesture_engine import GestureEngine, GestureType, ClassifiedGesture
from src.calculator_logic import GestureCalculator, CalcState
from src.ui_overlay import UIOverlayRenderer
from main import SyntheticHandSimulator

st.set_page_config(
    page_title="FinMath | AI Gesture Calculator",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyberpunk / Dark Terminal CSS
st.markdown("""
<style>
    .metric-box {
        background: #151821;
        border: 1px solid #283040;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 10px;
    }
    .big-eq {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0,255,204,0.3);
    }
    .op-tag {
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        background: #1f293d;
        color: #ff9900;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.title("🖐️ FinMath AI Terminal")
st.sidebar.caption("Computer Vision Hand Gesture Math Engine")

mode = st.sidebar.radio("Operating Mode", ["📷 Live Webcam", "🤖 Auto-Simulation Demo"], index=0)
camera_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
hold_time = st.sidebar.slider("Hold-to-Confirm Sensitivity (Seconds)", min_value=0.4, max_value=2.0, value=0.85, step=0.05)
draw_hud = st.sidebar.checkbox("Render Sci-Fi HUD Overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Gesture Guide")
st.sidebar.markdown("""
* **Thumb Only (👍):** **`*` Multiply**
* **Index Only (☝️):** **`+` Add**
* **Pinky Only (🤙):** **`-` Subtract**
* **Index + Middle (✌️):** **`/` Divide**
* **Fingers (1 - 5):** **Operand Number**
* **Closed Fist (✊):** **Hold to Clear (0)**
""")

if st.sidebar.button("🔄 Reset Calculator Memory"):
    st.session_state.calculator = GestureCalculator(hold_threshold_seconds=hold_time)
    st.rerun()

# Initialize session state for calculator
if "calculator" not in st.session_state:
    st.session_state.calculator = GestureCalculator(hold_threshold_seconds=hold_time)

calc: GestureCalculator = st.session_state.calculator
calc.hold_threshold = hold_time

# ------------------------------------------------------------------------------
# TOP DASHBOARD HEADER
# ------------------------------------------------------------------------------
st.title("🧮 FinMath: Real-Time Hand Gesture Math Calculator")
st.caption("Perform mathematical operations in mid-air using webcam gestures. Zero physical contact.")

# Live KPI columns
col_eq, col_op, col_res, col_state = st.columns([3, 1.2, 1.5, 2])

with col_eq:
    eq_display = calc.history[-1].expression if (calc.history and calc.state == CalcState.RESULT_DISPLAY) else (
        f"{calc.first_operand} {calc.active_operator} ?" if (calc.first_operand and calc.active_operator) else (
            f"? {calc.active_operator} ?" if calc.active_operator else "0"
        )
    )
    st.markdown(f'<div class="metric-box"><div style="font-size:0.85rem; color:#8892b0;">CURRENT EXPRESSION</div><div class="big-eq">{eq_display}</div></div>', unsafe_allow_html=True)

with col_op:
    op_str = calc.active_operator if calc.active_operator else "None"
    st.metric("Active Operator", op_str)

with col_res:
    res_str = f"{calc.running_result}" if calc.running_result is not None else "---"
    st.metric("Evaluated Result", res_str)

with col_state:
    st.metric("Engine State", calc.state.value.replace("_", " "))

# ------------------------------------------------------------------------------
# MAIN VIDEO FEED & CONTROLS
# ------------------------------------------------------------------------------
st.markdown("---")

run_feed = st.toggle("🟢 Start Camera & Gesture Processing", value=True)

feed_col, info_col = st.columns([3.5, 1.5])

with feed_col:
    video_placeholder = st.empty()
    status_banner = st.empty()

with info_col:
    st.subheader("📋 Calculation History")
    history_placeholder = st.empty()
    
    st.subheader("💡 Dynamic Step Guide")
    guide_placeholder = st.empty()

# Persistent Hand Detector
@st.cache_resource
def get_detector():
    return HandDetector()

detector = get_detector()
simulator = SyntheticHandSimulator(1280, 720) if mode == "🤖 Auto-Simulation Demo" else None

if run_feed:
    cap = None
    if mode == "📷 Live Webcam":
        # Release any previously active camera handle
        if "active_cap" in st.session_state and st.session_state.active_cap is not None:
            try:
                st.session_state.active_cap.release()
            except Exception:
                pass
            st.session_state.active_cap = None

        # Try default backend first (most reliable on Windows), then fallback to CAP_DSHOW
        cap = cv2.VideoCapture(int(camera_idx))
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(camera_idx), cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error(f"❌ Could not open camera {camera_idx}. Please verify webcam permissions, close any other app using the camera, or switch to Auto-Simulation Demo in the sidebar.")
            st.stop()

        # Safely attempt to set resolution without crashing on drivers that don't support custom props
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass

        st.session_state.active_cap = cap

    prev_time = time.time()
    fps = 30.0

    try:
        while run_feed:
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = (0.9 * fps) + (0.1 * (1.0 / dt))

            if mode == "🤖 Auto-Simulation Demo":
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                for y in range(0, 720, 40):
                    cv2.line(frame, (0, y), (1280, y), (25, 29, 36), 1)
                for x in range(0, 1280, 40):
                    cv2.line(frame, (x, 0), (x, 720), (25, 29, 36), 1)

                classified, sim_desc = simulator.get_simulated_gesture()
                calc_status = calc.update(classified)

                cv2.putText(
                    frame,
                    f"[SIMULATION DEMO] {sim_desc}",
                    (380, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Could not read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)
                hands = detector.process(frame)
                detector.draw_skeleton(frame, hands)
                classified = GestureEngine.classify(hands)
                calc_status = calc.update(classified)

            # Draw HUD
            if draw_hud:
                frame = UIOverlayRenderer.render(frame, calc_status, fps=fps)

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)

            # Update guidance text
            step_msg = calc_status.get("message", "")
            status_banner.info(f"**Live Engine Feedback:** {step_msg}")

            # Update history table
            if calc.history:
                df_hist = pd.DataFrame([
                    {"Time": r.timestamp, "Formula": r.expression, "Result": r.result}
                    for r in reversed(calc.history[-8:])
                ])
                history_placeholder.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                history_placeholder.caption("No calculations recorded yet.")

            # Update step guide
            cur_state = calc.state
            if cur_state == CalcState.WAITING_FOR_OPERATOR:
                guide_placeholder.markdown("""
                **👉 STEP 1: Select Operator**  
                Show your hand gesture for ~1 second:  
                * **Thumb only** ➔ Multiply `*`  
                * **Index only** ➔ Add `+`  
                * **Pinky only** ➔ Subtract `-`  
                * **Index + Middle** ➔ Divide `/`  
                """)
            elif cur_state == CalcState.WAITING_FOR_FIRST_OPERAND:
                guide_placeholder.markdown("""
                **👉 STEP 2: Show First Number**  
                Hold fingers (1 to 5) steady in front of the lens.
                """)
            elif cur_state == CalcState.WAITING_FOR_NEXT_OPERAND:
                guide_placeholder.markdown("""
                **👉 STEP 3: Show Next Number**  
                Hold successive fingers steady to compute!
                """)
            elif cur_state == CalcState.RESULT_DISPLAY:
                guide_placeholder.markdown(f"""
                **✅ Result Ready: `{calc.running_result}`**  
                * Show new fingers to continue chain!  
                * Show new operator to change operation.  
                * Hold a Closed Fist to reset to 0.  
                """)

            time.sleep(0.01)

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        st.session_state.active_cap = None
else:
    st.info("Click the toggle switch above to start the live video feed.")
