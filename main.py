"""
AI-Based Hand Gesture Recognition and Computer Control System  (Version 2)
Python 3.11 | OpenCV | MediaPipe | NumPy | PyAutoGUI | Tkinter

New in v2: scroll, swipe (slides), mute, right click, double click,
control lock, mouse on/off switch, sensitivity sliders, save history to CSV.
"""
import csv
import os
import time
import tkinter as tk
from collections import deque, Counter
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PIL import Image, ImageTk

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

# ----------------------------- Configuration -----------------------------
CAM_W, CAM_H = 640, 480
SMOOTH_WINDOW = 7
SMOOTH_MIN_VOTES = 5
MIN_HAND_CONF = 0.75
FRAME_MARGIN = 100
CLICK_HOLD_SEC = 0.5         # hold time for click / double click / right click
LOCK_HOLD_SEC = 1.0          # hold time to lock / unlock
ONE_SHOT_COOLDOWN = 1.5
REPEAT_COOLDOWN = 0.35
PINCH_MIN, PINCH_MAX = 0.15, 1.1
SWIPE_DIST = 0.22            # fraction of frame width the hand must travel
SWIPE_TIME = 0.5             # ...within this many seconds
SCROLL_DEADZONE = 0.05       # small movement ignored while scrolling
SCREENSHOT_DIR = "screenshots"

GESTURE_INFO = {
    "Open Palm":     "Play / Pause media",
    "Peace Sign":    "Capture screenshot",
    "Thumbs Up":     "Volume up",
    "Thumbs Down":   "Volume down",
    "Closed Fist":   "Pause video",
    "Index Finger":  "Move cursor",
    "OK Sign":       "Left click (hold)",
    "Pinch":         "Volume by distance",
    "Three Fingers": "Scroll (move hand up/down)",
    "Four Fingers":  "Swipe: next / prev slide",
    "Shaka":         "Mute / Unmute",
    "Rock Sign":     "Right click (hold)",
    "Middle Pinch":  "Double click (hold)",
    "Pinky Up":      "Lock / Unlock (hold 1s)",
}

# gestures that need a short hold, then fire once
HOLD_ACTIONS = {
    "OK Sign":      (pyautogui.click, "Mouse clicked"),
    "Middle Pinch": (pyautogui.doubleClick, "Double click"),
    "Rock Sign":    (pyautogui.rightClick, "Right click"),
}
MOUSE_GESTURES = {"Index Finger", "OK Sign", "Middle Pinch", "Rock Sign"}

BG, PANEL, ACCENT = "#0f1117", "#1a1d27", "#00d4aa"
TEXT, MUTED, WARN = "#e8eaf0", "#8b90a0", "#ff5c6c"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


# --------------------------- Gesture recognition ---------------------------
def dist(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def finger_extended(lm, tip, pip):
    return dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]) * 1.05


def thumb_extended(lm):
    return dist(lm[4], lm[17]) > dist(lm[3], lm[17]) * 1.1


def classify(lm):
    """Return (gesture_name, thumb-index pinch ratio)."""
    idx = finger_extended(lm, 8, 6)
    mid = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pnk = finger_extended(lm, 20, 18)
    thumb = thumb_extended(lm)

    palm = max(dist(lm[0], lm[9]), 1e-6)
    r_idx = dist(lm[4], lm[8]) / palm     # thumb <-> index tip
    r_mid = dist(lm[4], lm[12]) / palm    # thumb <-> middle tip

    if r_idx < 0.35 and mid and ring and pnk:
        return "OK Sign", r_idx
    if r_mid < 0.35 and idx and ring and pnk:
        return "Middle Pinch", r_idx
    if idx and mid and ring and pnk:
        return ("Open Palm" if thumb else "Four Fingers"), r_idx
    if idx and mid and ring:
        return "Three Fingers", r_idx
    if idx and mid and not ring and not pnk:
        return "Peace Sign", r_idx
    if idx and pnk and not mid and not ring:
        return "Rock Sign", r_idx
    if thumb and pnk and not (idx or mid or ring):
        return "Shaka", r_idx
    if pnk and not (thumb or idx or mid or ring):
        return "Pinky Up", r_idx
    if idx and thumb and not (mid or ring or pnk):
        return "Pinch", r_idx
    if idx and not (mid or ring or pnk):
        return "Index Finger", r_idx
    if not (idx or mid or ring or pnk):
        if thumb:
            if lm[4].y < lm[2].y - 0.3 * palm:
                return "Thumbs Up", r_idx
            if lm[4].y > lm[2].y + 0.3 * palm:
                return "Thumbs Down", r_idx
        return "Closed Fist", r_idx
    return "None", r_idx


# ------------------------------- Application -------------------------------
class GestureApp:
    def __init__(self, root):
        self.root = root
        root.title("AI-Based Hand Gesture Recognition and Computer Control System")
        root.configure(bg=BG)
        root.geometry("1280x820")
        root.minsize(1150, 740)
        root.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.cap = None
        self.running = False
        self.hands = None
        self.screen_w, self.screen_h = pyautogui.size()

        self.votes = deque(maxlen=SMOOTH_WINDOW)
        self.stable = "None"
        self.stable_since = time.time()
        self.fired_for_stable = False
        self.last_action = {}
        self.cursor = None
        self.media_playing = True
        self.vol_estimate = 50
        self.fps_t = time.time()

        self.locked = False
        self.action_count = 0
        self.anchor_y = None
        self.swipe_hist = deque()
        self.log_entries = []

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self.build_ui()
        self.show_placeholder()

    # ------------------------------- GUI -------------------------------
    def build_ui(self):
        head = tk.Frame(self.root, bg=PANEL, height=70)
        head.pack(fill="x", padx=12, pady=(12, 6))
        head.pack_propagate(False)
        logo = tk.Canvas(head, width=52, height=52, bg=PANEL, highlightthickness=0)
        logo.pack(side="left", padx=12)
        logo.create_oval(4, 4, 48, 48, fill=ACCENT, outline="")
        logo.create_text(26, 26, text="\u270B", font=("Segoe UI Emoji", 20), fill=BG)
        tk.Label(head, text="AI Hand Gesture Control System", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(head, text="OpenCV  |  MediaPipe  |  PyAutoGUI", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="right", padx=16)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=6)

        # ---- left: webcam + buttons
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        # Camera selection panel
        camera_frame = tk.Frame(left, bg=PANEL)
        camera_frame.pack(fill="x", pady=(0, 8))

        tk.Label(
        camera_frame,
        text="Select Camera:",
        bg=PANEL,
        fg=TEXT,
        font=("Segoe UI", 10, "bold")
        ).pack(side="left", padx=10, pady=8)

        self.camera_var = tk.StringVar(value="0")

        self.camera_combo = ttk.Combobox(
        camera_frame,
        textvariable=self.camera_var,
        state="readonly",
        width=10
        )
        self.camera_combo.pack(side="left", padx=5)

        tk.Button(
        camera_frame,
        text="Refresh Cameras",
        command=self.refresh_cameras,
        bg=ACCENT,
        fg=BG,
        bd=0
        ).pack(side="left", padx=10)

        self.refresh_cameras()
        self.video = tk.Label(left, bg="black", width=CAM_W, height=CAM_H)
        self.video.pack(pady=(0, 8))
        btns = tk.Frame(left, bg=BG)
        btns.pack()
        self.make_btn(btns, "Start Camera", self.start_camera, ACCENT, BG)
        self.make_btn(btns, "Stop Camera", self.stop_camera, "#f5a623", BG)
        self.make_btn(btns, "Screenshot", self.take_screenshot, "#4a90e2", "white")
        self.make_btn(btns, "Save History", self.save_history, "#7c5cff", "white")
        self.make_btn(btns, "Exit", self.exit_app, WARN, "white")

        # ---- right: info panels
        right = tk.Frame(body, bg=BG, width=520)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        status = tk.Frame(right, bg=PANEL)
        status.pack(fill="x", pady=(0, 6))
        self.lbl_detected = self.make_row(status, "Detected Gesture", "-")
        self.lbl_name = self.make_row(status, "Recognized Gesture", "-")
        self.lbl_hand = self.make_row(status, "Hand / Confidence", "-")
        self.lbl_action = self.make_row(status, "Computer Action", "Idle")
        self.lbl_mode = self.make_row(status, "Control Mode", "ACTIVE")
        self.lbl_count = self.make_row(status, "Actions Performed", "0")
        self.lbl_cam = self.make_row(status, "Camera / FPS", "Stopped")

        tk.Label(right, text="Gesture Instructions", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        guide = tk.Frame(right, bg=PANEL)
        guide.pack(fill="x", pady=(2, 6))
        items = list(GESTURE_INFO.items())
        half = (len(items) + 1) // 2
        for i, (g, a) in enumerate(items):
            col, row = divmod(i, half)
            cell = tk.Frame(guide, bg=PANEL)
            cell.grid(row=row, column=col, sticky="w", padx=8, pady=1)
            tk.Label(cell, text=g, bg=PANEL, fg=TEXT, anchor="w",
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            tk.Label(cell, text=a, bg=PANEL, fg=MUTED, anchor="w",
                     font=("Segoe UI", 8)).pack(anchor="w")

        tk.Label(right, text="Settings", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        sett = tk.Frame(right, bg=PANEL)
        sett.pack(fill="x", pady=(2, 6))
        self.mouse_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(sett, text="Enable mouse control gestures", variable=self.mouse_enabled,
                       bg=PANEL, fg=TEXT, selectcolor=BG, activebackground=PANEL,
                       activeforeground=TEXT, font=("Segoe UI", 9)).pack(anchor="w", padx=8)
        self.cursor_speed = self.make_slider(sett, "Cursor responsiveness", 0.10, 0.60, 0.25, 0.05)
        self.scroll_speed = self.make_slider(sett, "Scroll speed", 1, 10, 5, 1)

        tk.Label(right, text="Gesture Recognition History", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        hf = tk.Frame(right, bg=PANEL)
        hf.pack(fill="both", expand=True, pady=(2, 0))
        sb = tk.Scrollbar(hf)
        sb.pack(side="right", fill="y")
        self.history = tk.Listbox(hf, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0,
                                  font=("Consolas", 9), yscrollcommand=sb.set,
                                  selectbackground=ACCENT)
        self.history.pack(fill="both", expand=True, padx=6, pady=4)
        sb.config(command=self.history.yview)

    def make_btn(self, parent, text, cmd, bg, fg):
        tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, bd=0, width=12,
                  font=("Segoe UI", 10, "bold"), activebackground=TEXT, cursor="hand2",
                  pady=6).pack(side="left", padx=4)

    def make_row(self, parent, title, value):
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill="x", padx=10, pady=2)
        tk.Label(r, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                 width=19, anchor="w").pack(side="left")
        lbl = tk.Label(r, text=value, bg=PANEL, fg=ACCENT, font=("Segoe UI", 11, "bold"),
                       anchor="w")
        lbl.pack(side="left")
        return lbl

    def make_slider(self, parent, text, lo, hi, default, step):
        var = tk.DoubleVar(value=default)
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill="x", padx=8)
        tk.Label(r, text=text, bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
                 width=21, anchor="w").pack(side="left")
        tk.Scale(r, from_=lo, to=hi, resolution=step, orient="horizontal", variable=var,
                 bg=PANEL, fg=TEXT, troughcolor=BG, highlightthickness=0, length=220,
                 font=("Segoe UI", 8)).pack(side="left")
        return var

    def show_placeholder(self):
        img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        cv2.putText(img, "Camera is off - press Start Camera", (95, CAM_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 144, 160), 2)
        self.set_video(img)

    def set_video(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=tkimg)
        self.video.image = tkimg

    # ---------------------------- Camera control ----------------------------
    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open the webcam.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.hands = mp_hands.Hands(max_num_hands=1, model_complexity=1,
                                    min_detection_confidence=0.7,
                                    min_tracking_confidence=0.6)
        self.running = True
        self.votes.clear()
        self.lbl_cam.config(text="Running")
        self.log("Camera started")
        self.loop()

    def stop_camera(self):
        if not self.running:
            return
        self.running = False
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        self.cap, self.hands, self.cursor = None, None, None
        for lbl, txt in ((self.lbl_cam, "Stopped"), (self.lbl_detected, "-"),
                         (self.lbl_name, "-"), (self.lbl_hand, "-"),
                         (self.lbl_action, "Idle")):
            lbl.config(text=txt)
        self.show_placeholder()
        self.log("Camera stopped")

    def exit_app(self):
        self.stop_camera()
        self.root.destroy()

    # ------------------------------ Main loop ------------------------------
    def loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            self.process(frame)
            self.set_video(frame)
            now = time.time()
            fps = 1.0 / max(now - self.fps_t, 1e-6)
            self.fps_t = now
            self.lbl_cam.config(text=f"Running  {fps:4.1f} FPS")
        self.root.after(10, self.loop)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        gesture, ratio, label, conf = "None", 0.0, "-", 0.0
        lm = None

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            cls = res.multi_handedness[0].classification[0]
            label, conf = cls.label, cls.score
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                   mp_style.get_default_hand_landmarks_style(),
                                   mp_style.get_default_hand_connections_style())
            if conf >= MIN_HAND_CONF:
                lm = hand.landmark
                gesture, ratio = classify(lm)

        self.votes.append(gesture)
        top, count = Counter(self.votes).most_common(1)[0]
        new_stable = top if count >= SMOOTH_MIN_VOTES else self.stable
        if new_stable != self.stable:
            self.stable, self.stable_since = new_stable, time.time()
            self.fired_for_stable = False
            self.anchor_y = None
            self.swipe_hist.clear()

        shown = self.stable if self.stable != "None" else "-"
        self.lbl_detected.config(text=gesture if gesture != "None" else "-")
        self.lbl_name.config(text=shown)
        self.lbl_hand.config(text=f"{label} hand ({conf:.0%})" if label != "-" else "-")

        cv2.putText(frame, f"{shown}  [{label}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 212, 170), 2)
        if self.locked:
            cv2.putText(frame, "LOCKED", (CAM_W - 150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (92, 92, 255), 2)

        if lm is not None and self.stable != "None":
            self.act(self.stable, lm, ratio, frame)

    # ------------------------------- Actions -------------------------------
    def ready(self, key, cooldown):
        now = time.time()
        if now - self.last_action.get(key, 0) >= cooldown:
            self.last_action[key] = now
            return True
        return False

    def act(self, g, lm, ratio, frame):
        held = time.time() - self.stable_since

        # ---- lock / unlock (always available) ----
        if g == "Pinky Up":
            if held >= LOCK_HOLD_SEC and not self.fired_for_stable:
                if self.ready("lock", 1.5):
                    self.fired_for_stable = True
                    self.locked = not self.locked
                    self.lbl_mode.config(text="LOCKED" if self.locked else "ACTIVE",
                                         fg=WARN if self.locked else ACCENT)
                    self.report(g, "Controls LOCKED" if self.locked else "Controls UNLOCKED")
            elif not self.fired_for_stable:
                self.lbl_action.config(text=f"Hold to toggle lock... {held / LOCK_HOLD_SEC:.0%}")
            return

        if self.locked:
            self.lbl_action.config(text="Locked (hold Pinky Up to unlock)")
            return

        if g in MOUSE_GESTURES and not self.mouse_enabled.get():
            self.lbl_action.config(text="Mouse control is disabled")
            return

        # ---- one-shot toggles ----
        if g in ("Open Palm", "Peace Sign", "Closed Fist", "Shaka"):
            if not self.fired_for_stable and self.ready("oneshot", ONE_SHOT_COOLDOWN):
                self.fired_for_stable = True
                if g == "Open Palm":
                    pyautogui.press("playpause")
                    self.media_playing = not self.media_playing
                    self.report(g, "Play/Pause toggled")
                elif g == "Peace Sign":
                    path = self.take_screenshot(silent=True)
                    self.report(g, f"Screenshot saved ({os.path.basename(path)})")
                elif g == "Shaka":
                    pyautogui.press("volumemute")
                    self.report(g, "Mute toggled")
                else:
                    if self.media_playing:
                        pyautogui.press("playpause")
                        self.media_playing = False
                        self.report(g, "Video paused")
                    else:
                        self.report(g, "Already paused")

        # ---- hold-to-click gestures (click, double click, right click) ----
        elif g in HOLD_ACTIONS:
            func, msg = HOLD_ACTIONS[g]
            if held >= CLICK_HOLD_SEC and not self.fired_for_stable:
                if self.ready("click", 1.0):
                    self.fired_for_stable = True
                    func()
                    self.report(g, msg)
            elif not self.fired_for_stable:
                self.lbl_action.config(text=f"Hold {g}... {held / CLICK_HOLD_SEC:.0%}")

        # ---- continuous gestures ----
        elif g == "Thumbs Up":
            if self.ready("vol", REPEAT_COOLDOWN):
                pyautogui.press("volumeup", presses=2)
                self.vol_estimate = min(100, self.vol_estimate + 4)
                self.report(g, "Volume increased")
        elif g == "Thumbs Down":
            if self.ready("vol", REPEAT_COOLDOWN):
                pyautogui.press("volumedown", presses=2)
                self.vol_estimate = max(0, self.vol_estimate - 4)
                self.report(g, "Volume decreased")
        elif g == "Pinch":
            self.pinch_volume(ratio, frame)
        elif g == "Index Finger":
            self.move_cursor(lm, frame)
        elif g == "Three Fingers":
            self.scroll(lm, frame)
        elif g == "Four Fingers":
            self.swipe(lm)

    def move_cursor(self, lm, frame):
        h, w = frame.shape[:2]
        x, y = lm[8].x * w, lm[8].y * h
        cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN),
                      (w - FRAME_MARGIN, h - FRAME_MARGIN), (0, 212, 170), 1)
        sx = np.interp(x, (FRAME_MARGIN, w - FRAME_MARGIN), (0, self.screen_w - 1))
        sy = np.interp(y, (FRAME_MARGIN, h - FRAME_MARGIN), (0, self.screen_h - 1))
        if self.cursor is None:
            self.cursor = (sx, sy)
        k = self.cursor_speed.get()
        cx = self.cursor[0] + (sx - self.cursor[0]) * k
        cy = self.cursor[1] + (sy - self.cursor[1]) * k
        self.cursor = (cx, cy)
        pyautogui.moveTo(int(cx), int(cy))
        self.lbl_action.config(text="Moving cursor")
        if self.ready("log_move", 3.0):
            self.log("Index Finger -> Moving cursor")

    def scroll(self, lm, frame):
        """Joystick-style scroll: move hand up/down from where the gesture started."""
        y = lm[9].y
        if self.anchor_y is None:
            self.anchor_y = y
        dy = y - self.anchor_y
        ay = int(self.anchor_y * frame.shape[0])
        cv2.line(frame, (0, ay), (CAM_W, ay), (245, 166, 35), 1)
        if abs(dy) > SCROLL_DEADZONE and self.ready("scroll", 0.06):
            amount = int(np.clip(-dy * self.scroll_speed.get() * 400, -400, 400))
            pyautogui.scroll(amount)
            self.lbl_action.config(text="Scrolling " + ("up" if amount > 0 else "down"))
            if self.ready("log_scroll", 2.0):
                self.log("Three Fingers -> Scrolling " + ("up" if amount > 0 else "down"))
        elif abs(dy) <= SCROLL_DEADZONE:
            self.lbl_action.config(text="Scroll ready (move hand up/down)")

    def swipe(self, lm):
        """Fast sideways hand movement -> Right / Left arrow key."""
        now, x = time.time(), lm[9].x
        self.swipe_hist.append((now, x))
        while self.swipe_hist and now - self.swipe_hist[0][0] > SWIPE_TIME:
            self.swipe_hist.popleft()
        self.lbl_action.config(text="Swipe ready (move hand sideways)")
        if len(self.swipe_hist) >= 3:
            dx = x - self.swipe_hist[0][1]
            if abs(dx) > SWIPE_DIST and self.ready("swipe", 1.0):
                pyautogui.press("right" if dx > 0 else "left")
                self.swipe_hist.clear()
                self.report("Four Fingers", "Next slide" if dx > 0 else "Previous slide")

    def pinch_volume(self, ratio, frame):
        target = int(np.interp(ratio, (PINCH_MIN, PINCH_MAX), (0, 100)))
        cv2.putText(frame, f"Pinch volume: {target}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 166, 35), 2)
        diff = target - self.vol_estimate
        if abs(diff) >= 4 and self.ready("pinch", 0.15):
            steps = min(5, abs(diff) // 2)
            pyautogui.press("volumeup" if diff > 0 else "volumedown", presses=steps)
            self.vol_estimate += steps * 2 * (1 if diff > 0 else -1)
            self.lbl_action.config(text=f"Pinch volume -> {target}%")
            if self.ready("log_pinch", 2.0):
                self.log(f"Pinch -> Volume target {target}%")

    # ------------------------------- Helpers -------------------------------
    def report(self, gesture, action):
        self.action_count += 1
        self.lbl_count.config(text=str(self.action_count))
        self.lbl_action.config(text=action)
        self.log(f"{gesture} -> {action}")

    def log(self, msg):
        stamp = datetime.now()
        self.log_entries.append((stamp.strftime("%Y-%m-%d %H:%M:%S"), msg))
        self.history.insert(0, f"[{stamp:%H:%M:%S}] {msg}")
        if self.history.size() > 200:
            self.history.delete(200, "end")

    def take_screenshot(self, silent=False):
        name = f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
        path = os.path.join(SCREENSHOT_DIR, name)
        pyautogui.screenshot().save(path)
        if not silent:
            self.report("Button", f"Screenshot saved ({name})")
        return path

    def save_history(self):
        if not self.log_entries:
            messagebox.showinfo("Save History", "No history to save yet.")
            return
        path = f"gesture_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Time", "Event"])
            w.writerows(self.log_entries)
        messagebox.showinfo("Save History", f"History saved to {path}")


if __name__ == "__main__":
    root = tk.Tk()
    GestureApp(root)
    root.mainloop()