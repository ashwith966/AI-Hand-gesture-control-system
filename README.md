HEAD
# AI-Based Hand Gesture Recognition and Computer Control System

Control your computer with hand gestures using just a webcam. No mouse or keyboard needed.

Built with **Python 3.11, OpenCV, MediaPipe, NumPy, PyAutoGUI and Tkinter**.

## Features

- Real-time hand tracking with 21 landmarks (MediaPipe)
- 14 gestures recognised using landmark geometry
- Gesture smoothing, confidence check and cooldown to avoid accidental actions
- Left and right hand detection
- Modern dark-themed Tkinter dashboard with live webcam, gesture history and settings
- Lock mode, mouse on/off switch, sensitivity sliders
- Save gesture history to CSV

## Gestures

| Gesture | Action |
|---|---|
| Open Palm | Play / Pause media |
| Peace Sign | Take screenshot |
| Thumbs Up / Down | Volume up / down |
| Closed Fist | Pause video |
| Index Finger | Move mouse cursor |
| OK Sign (hold) | Left click |
| Pinch | Volume by finger distance |
| Three Fingers | Scroll (move hand up / down) |
| Four Fingers (thumb tucked) | Swipe for next / previous slide |
| Shaka | Mute / Unmute |
| Rock Sign (hold) | Right click |
| Middle Pinch (hold) | Double click |
| Pinky Up (hold 1 s) | Lock / Unlock all gestures |

## Installation

Python **3.11** is required (MediaPipe 0.10.14 has no build for newer versions).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main_v2.py
```

`main.py` is the basic first version with 8 gestures.

## Safety

PyAutoGUI fail-safe is on. If the cursor goes out of control, move the mouse to a screen corner to stop the program.

## Project structure

```
.
├── main.py             # Version 1 (8 gestures)
├── main_v2.py          # Version 2 (14 gestures + extra controls)
├── requirements.txt
├── .gitignore
└── README.md
```

## Author

<Your Name>, Dept. of CSE (AIML), SCEM Mangalore

# AI-Hand-gesture-control-system
6b7b426c50b4c2b042c4cd6d252373cc612935a9
