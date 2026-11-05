"""VisionCalc: Streamlit Web Terminal for Gesture Computer Vision Calculator.
Enables real-time browser access to the webcam math gesture recognition engine.
Supports:
  1. Live Browser Camera (WebRTC - Cloud & Local)
  2. Browser Snapshot Camera (HTML5 camera_input for cloud mobile/desktop access)
  3. Auto-Simulation Demo (Runs universally on Streamlit Cloud & local)
  4. Local Desktop Stream (Direct OpenCV for local machine execution)
"""

from __future__ import annotations
import os
import sys
import time
import threading
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

try:
    import av
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase,
    )
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

st.set_page_config(
    page_title="VisionCalc | AI Gesture Terminal",
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
st.sidebar.title("🖐️ VisionCalc AI Terminal")
st.sidebar.caption("Computer Vision Hand Gesture Math Engine")

MODES = [
    "🌐 Live Browser Camera (WebRTC - Cloud & Local)",
    "📸 Browser Camera Snapshot (Photo Input)",
    "🤖 Auto-Simulation Demo",
    "💻 Local Desktop Stream (Physical USB/Webcam)",
]

if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = MODES[0]

mode = st.sidebar.radio(
    "Operating Mode",
    MODES,
    key="app_mode",
)
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

if st.sidebar.button("🔄 Reset Calculator Memory", use_container_width=True):
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
st.title("🧮 VisionCalc: Real-Time Hand Gesture Math Calculator")
st.caption("Perform mathematical operations in mid-air using hand gestures. Zero physical contact.")

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


def render_dynamic_guide(cur_state: CalcState, running_res: float | None = None):
    """Renders contextual step-by-step guidance."""
    if cur_state == CalcState.WAITING_FOR_OPERATOR:
        st.markdown("""
        **👉 STEP 1: Select Operator**  
        Show your hand gesture for ~1 second:  
        * **Thumb only (👍)** ➔ Multiply `*`  
        * **Index only (☝️)** ➔ Add `+`  
        * **Pinky only (🤙)** ➔ Subtract `-`  
        * **Index + Middle (✌️)** ➔ Divide `/`  
        """)
    elif cur_state == CalcState.WAITING_FOR_FIRST_OPERAND:
        st.markdown("""
        **👉 STEP 2: Show First Number**  
        Hold fingers (1 to 5) steady in front of the lens.
        """)
    elif cur_state == CalcState.WAITING_FOR_NEXT_OPERAND:
        st.markdown("""
        **👉 STEP 3: Show Next Number**  
        Hold successive fingers steady to compute!
        """)
    elif cur_state == CalcState.RESULT_DISPLAY:
        st.markdown(f"""
        **✅ Result Ready: `{running_res}`**  
        * Show new fingers (1-5) to continue chain!  
        * Show new operator (👍/☝️/🤙/✌️) to change operation.  
        * Hold a Closed Fist (✊) to reset to 0.  
        """)


# ------------------------------------------------------------------------------
# HAND DETECTOR INITIALIZATION WITH RESILIENT CATCH
# ------------------------------------------------------------------------------
@st.cache_resource
def load_detector():
    try:
        return HandDetector(), None
    except Exception as exc:
        return None, str(exc)

detector, detector_err = load_detector()

if detector is None:
    st.warning(f"⚠️ **Note:** MediaPipe C++ engine initializing in cloud container fallback mode ({detector_err}).")
    st.info("💡 **Auto-Simulation Demo** is active and fully functional below!")

st.markdown("---")

# ------------------------------------------------------------------------------
# MODE 1: LIVE WEBRTC BROWSER CAMERA (Cloud & Local)
# ------------------------------------------------------------------------------
if mode == "🌐 Live Browser Camera (WebRTC - Cloud & Local)":
    st.subheader("🌐 Live Browser Camera Stream")
    st.caption("Streams your camera directly through your web browser with zero server hardware required (works on Streamlit Cloud & mobile).")

    if not WEBRTC_AVAILABLE:
        st.error("⚠️ `streamlit-webrtc` or `av` is not installed.")
        st.info("Run `pip install streamlit-webrtc av` to enable live WebRTC streaming.")
    else:
        feed_col, info_col = st.columns([3.5, 1.5])

        with feed_col:
            calc_lock = threading.Lock()

            class WebRTCVideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.fps = 30.0
                    self.prev_time = time.time()

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.flip(img, 1)

                    curr_time = time.time()
                    dt = curr_time - self.prev_time
                    self.prev_time = curr_time
                    if dt > 0:
                        self.fps = (0.9 * self.fps) + (0.1 * (1.0 / dt))

                    with calc_lock:
                        if detector is not None:
                            try:
                                hands = detector.process(img)
                                detector.draw_skeleton(img, hands)
                                classified = GestureEngine.classify(hands)
                            except Exception:
                                classified = ClassifiedGesture(GestureType.NONE, description="Processing error")
                        else:
                            classified = ClassifiedGesture(GestureType.NONE, description="Detector initializing...")

                        status = calc.update(classified)

                    if draw_hud:
                        img = UIOverlayRenderer.render(img, status, fps=self.fps)

                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_ctx = webrtc_streamer(
                key="visioncalc-webrtc-streamer",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                    ]}
                ),
                video_processor_factory=WebRTCVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if webrtc_ctx.state.playing:
                st.success("🟢 Live camera feed active! Hold gestures steady in front of your camera.")
            else:
                st.info("👆 Click **START** above and allow camera access in your browser to begin.")

        with info_col:
            st.subheader("📋 Calculation History")
            if calc.history:
                df_hist = pd.DataFrame([
                    {"Time": r.timestamp, "Formula": r.expression, "Result": r.result}
                    for r in reversed(calc.history[-8:])
                ])
                st.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                st.caption("No calculations recorded yet.")

            st.subheader("💡 Dynamic Step Guide")
            render_dynamic_guide(calc.state, calc.running_result)

# ------------------------------------------------------------------------------
# MODE 2: BROWSER CAMERA SNAPSHOT (HTML5 for Cloud & Mobile)
# ------------------------------------------------------------------------------
elif mode == "📸 Browser Camera Snapshot (Photo Input)":
    st.subheader("📸 Browser Camera Snapshot")
    st.caption("Snap a photo of your hand gesture directly through your browser. Zero setup required.")

    snap_col, info_col = st.columns([3.5, 1.5])

    with snap_col:
        snap_file = st.camera_input("Take a photo of your hand gesture")
        if snap_file is not None:
            file_bytes = np.asarray(bytearray(snap_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if detector is not None:
                hands = detector.process(frame)
                detector.draw_skeleton(frame, hands)
                classified = GestureEngine.classify(hands)

                # For snapshots, commit immediately with hold_threshold=0
                temp_calc = GestureCalculator(hold_threshold_seconds=0.0)
                temp_calc.state = calc.state
                temp_calc.active_operator = calc.active_operator
                temp_calc.first_operand = calc.first_operand
                temp_calc.running_result = calc.running_result
                status = temp_calc.update(classified)

                calc.state = temp_calc.state
                calc.active_operator = temp_calc.active_operator
                calc.first_operand = temp_calc.first_operand
                calc.running_result = temp_calc.running_result
                if temp_calc.history:
                    calc.history.extend(temp_calc.history)

                if draw_hud:
                    frame = UIOverlayRenderer.render(frame, status, fps=1.0)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Detected: {classified.description}", use_container_width=True)
                st.success(f"Status: {status.get('message', '')}")
            else:
                st.error("Detector is not loaded in this container.")

    with info_col:
        st.subheader("📋 Calculation History")
        if calc.history:
            df_hist = pd.DataFrame([
                {"Time": r.timestamp, "Formula": r.expression, "Result": r.result}
                for r in reversed(calc.history[-8:])
            ])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.caption("No calculations recorded yet.")

        st.subheader("💡 Dynamic Step Guide")
        render_dynamic_guide(calc.state, calc.running_result)

# ------------------------------------------------------------------------------
# MODE 3: AUTO-SIMULATION DEMO
# ------------------------------------------------------------------------------
elif mode == "🤖 Auto-Simulation Demo":
    st.subheader("🤖 Automated Hand Gesture Simulation")
    st.caption("Simulates dynamic math gestures on a synthetic grid feed without requiring any webcam.")

    run_feed = st.toggle("🟢 Run Simulation Feed", value=True)

    feed_col, info_col = st.columns([3.5, 1.5])

    with feed_col:
        video_placeholder = st.empty()
        status_banner = st.empty()

    with info_col:
        st.subheader("📋 Calculation History")
        history_placeholder = st.empty()

        st.subheader("💡 Dynamic Step Guide")
        guide_placeholder = st.empty()

    simulator = SyntheticHandSimulator(1280, 720)

    if run_feed:
        prev_time = time.time()
        fps = 30.0

        while run_feed:
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = (0.9 * fps) + (0.1 * (1.0 / dt))

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

            if draw_hud:
                frame = UIOverlayRenderer.render(frame, calc_status, fps=fps)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)

            step_msg = calc_status.get("message", "")
            status_banner.info(f"**Live Engine Feedback:** {step_msg}")

            if calc.history:
                df_hist = pd.DataFrame([
                    {"Time": r.timestamp, "Formula": r.expression, "Result": r.result}
                    for r in reversed(calc.history[-8:])
                ])
                history_placeholder.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                history_placeholder.caption("No calculations recorded yet.")

            cur_state = calc.state
            if cur_state == CalcState.WAITING_FOR_OPERATOR:
                guide_placeholder.markdown("""
                **👉 STEP 1: Select Operator**  
                Simulating: **Multiply (`*`)**
                """)
            elif cur_state == CalcState.WAITING_FOR_FIRST_OPERAND:
                guide_placeholder.markdown("""
                **👉 STEP 2: Show First Number**  
                Simulating: **Finger counts (1 - 5)**
                """)
            elif cur_state == CalcState.WAITING_FOR_NEXT_OPERAND:
                guide_placeholder.markdown("""
                **👉 STEP 3: Show Next Number**  
                Simulating: **Second operand**
                """)
            elif cur_state == CalcState.RESULT_DISPLAY:
                guide_placeholder.markdown(f"""
                **✅ Result Ready: `{calc.running_result}`**  
                Simulating next action...
                """)

            time.sleep(0.02)
    else:
        st.info("Click the toggle switch above to start the simulation feed.")

# ------------------------------------------------------------------------------
# MODE 4: LOCAL DESKTOP DIRECT STREAM (OpenCV)
# ------------------------------------------------------------------------------
elif mode == "💻 Local Desktop Stream (Physical USB/Webcam)":
    st.subheader("💻 Local Desktop Direct OpenCV Stream")
    st.caption("Direct USB/built-in camera access for running Python locally on your PC.")

    # Probe camera hardware availability
    test_cap = cv2.VideoCapture(0)
    has_cam = test_cap.isOpened()
    if not has_cam:
        test_cap.open(0, cv2.CAP_DSHOW)
        has_cam = test_cap.isOpened()
    test_cap.release()

    if not has_cam:
        st.warning("⚠️ **No physical webcam hardware detected on this server.**")
        st.info(
            "ℹ️ When deployed to **Streamlit Community Cloud** or any cloud server, Python runs inside a headless datacenter container with no physical camera attached.\n\n"
            "👉 **Use one of the browser camera modes below to stream from your computer or phone:**"
        )
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🌐 Switch to Live Browser Camera (WebRTC)", use_container_width=True):
                st.session_state["app_mode"] = "🌐 Live Browser Camera (WebRTC - Cloud & Local)"
                st.rerun()
        with col_btn2:
            if st.button("📸 Switch to Browser Camera Snapshot", use_container_width=True):
                st.session_state["app_mode"] = "📸 Browser Camera Snapshot (Photo Input)"
                st.rerun()
    else:
        run_feed = st.toggle("🟢 Run Local Camera Feed", value=True)

        feed_col, info_col = st.columns([3.5, 1.5])

        with feed_col:
            video_placeholder = st.empty()
            status_banner = st.empty()

        with info_col:
            st.subheader("📋 Calculation History")
            history_placeholder = st.empty()

            st.subheader("💡 Dynamic Step Guide")
            guide_placeholder = st.empty()

        if run_feed:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except Exception:
                pass

            prev_time = time.time()
            fps = 30.0

            try:
                while run_feed:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Could not read frame from local camera.")
                        break

                    curr_time = time.time()
                    dt = curr_time - prev_time
                    prev_time = curr_time
                    if dt > 0:
                        fps = (0.9 * fps) + (0.1 * (1.0 / dt))

                    frame = cv2.flip(frame, 1)
                    if detector is not None:
                        hands = detector.process(frame)
                        detector.draw_skeleton(frame, hands)
                        classified = GestureEngine.classify(hands)
                    else:
                        classified = ClassifiedGesture(GestureType.NONE, description="Detector unavailable")

                    calc_status = calc.update(classified)

                    if draw_hud:
                        frame = UIOverlayRenderer.render(frame, calc_status, fps=fps)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, use_container_width=True)

                    step_msg = calc_status.get("message", "")
                    status_banner.info(f"**Live Engine Feedback:** {step_msg}")

                    if calc.history:
                        df_hist = pd.DataFrame([
                            {"Time": r.timestamp, "Formula": r.expression, "Result": r.result}
                            for r in reversed(calc.history[-8:])
                        ])
                        history_placeholder.dataframe(df_hist, use_container_width=True, hide_index=True)
                    else:
                        history_placeholder.caption("No calculations recorded yet.")

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
                    cap.release()
        else:
            st.info("Click the toggle switch above to start the local camera feed.")
