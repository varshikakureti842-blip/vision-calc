# FinMath: AI Hand Gesture Computer Vision Math Calculator 🧮 🖐️

A real-time, interactive computer vision calculator powered by **OpenCV** and **MediaPipe**. Compute mathematical operations seamlessly in mid-air using hand gestures captured by your webcam with zero physical contact!

---

## 🌟 Gesture Mapping Specification

The project strictly follows these ergonomic, intuitive gesture controls:

| Gesture | Fingers Extended | Mathematical Action | Description |
| :--- | :--- | :---: | :--- |
| **Thumb Only** | 👍 Only Thumb | **`*` (Multiplication)** | Selects multiplication of the first number and successive numbers |
| **Index Only** | ☝️ Only Index | **`+` (Addition)** | Selects addition |
| **Pinky Only** | 🤙 Only Pinky | **`-` (Subtraction)** | Selects subtraction |
| **Index + Middle** | ✌️ Peace / V Sign | **`/` (Division)** | Selects division (includes division-by-zero protection) |
| **Finger Count** | 1, 2, 3, 4, 5 (up to 10) | **Numerical Value** | Sets operand value ($N_1$, $N_2$, $N_3 \dots$) |
| **Closed Fist** | ✊ 0 Fingers | **Hold to Clear / Reset** | Holding for 1.4s resets registers back to zero |

---

## 🚀 How It Works: The 3-Step Flow

```
1. SHOW OPERATOR GESTURE ─────────► 2. SHOW FIRST NUMBER ─────────► 3. SHOW SUCCESSIVE NUMBERS
   (Thumb: *, Index: +,                (Show 1-5 fingers steady        (Show next fingers steady ->
    Pinky: -, Peace: /)                 to lock in first operand)       Instant result + chaining!)
```

### Example: Successive Multiplication
1. Show **Thumb only** for 1 sec ➔ Operator locked to `*`.
2. Show **4 fingers** steady ➔ Operand 1 locked to `4` (Equation: `4 * ?`).
3. Show **3 fingers** steady ➔ Operand 2 locked to `3` (Result: `4 * 3 = 12`).
4. Show **2 fingers** steady ➔ Successive operation executes immediately (Result: `12 * 2 = 24`)!
5. Hold a **Closed Fist** for 1.4s ➔ Resets calculator back to zero!

---

## 🛠 Tech Stack & Architecture

* **Computer Vision:** OpenCV (`opencv-python 5.0+`) for camera streaming, alpha-blended transparent HUD, and frame drawing.
* **Hand Tracking:** Google MediaPipe (`mediapipe 1.1+`) Hand Landmarker for 21 3D joint landmark extraction.
* **Temporal Stability & Debounce:** State machine with temporal hold accumulation (`hold_threshold = 0.85s`) to eliminate jitter and transitional false positives.
* **Acoustic Feedback:** Windows audio feedback (`winsound`) with frequency modulation on confirmation and calculation events.
* **Test Suite:** Automated `unittest` verifying gesture classification, state transitions, and math edge cases.

---

## 💻 Getting Started & Running

### 1. Activate the Virtual Environment
```powershell
cd C:\Users\varsh.VARSHI\gesture_math_calculator
.\.venv\Scripts\Activate.ps1
```

### 2. Run Automated Unit Tests (10/10 Passing)
```powershell
python -m unittest discover -s tests
```

### 3. Launch Real-Time Webcam Calculator
```powershell
python main.py
```

### 4. Run in Simulation / Demo Mode (No Webcam Required)
```powershell
python main.py --demo
```

---

## 🎮 Interactive Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| **`Q`** or **`ESC`** | Terminate and close application |
| **`C`** or **`R`** | Instant clear / reset registers |
| **`S`** | Capture and save timestamped PNG screenshot to `screenshots/` |
| **`H`** | Toggle HUD elements on / off |

---

## 📁 Project Structure

```
gesture_math_calculator/
├── .gitignore
├── requirements.txt
├── README.md
├── main.py                          # Main webcam execution & CLI
├── models/
│   └── hand_landmarker.task        # MediaPipe 21-landmark model weights
├── src/
│   ├── __init__.py
│   ├── hand_detector.py            # Landmark extraction & neon skeleton rendering
│   ├── gesture_engine.py           # Gesture classification engine
│   ├── calculator_logic.py         # State machine, debounce, chaining & math
│   └── ui_overlay.py               # Sci-Fi glassmorphism HUD & progress rings
├── tests/
│   ├── __init__.py
│   └── test_calculator.py          # 10 automated unit tests
└── screenshots/                    # Captured frames and demos
```
