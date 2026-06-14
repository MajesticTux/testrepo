#!/usr/bin/env python3
"""
Drone Detector & PTZ Controller — Modern UI
─────────────────────────────────────────────
• Inference resolution slider  (30 → 100 %)
• Runtime confidence slider    (10 → 95 %)
• Screenshot / video recording
• Live stats bar, uptime, peak-detection tracking
• Always-on-top pin
• Colour-coded bounding boxes with label backgrounds
• Full dark theme
"""

import cv2
import time
import sys
import numpy as np
import torch
import requests
import threading
import json
import os
from datetime import datetime
from requests.auth import HTTPDigestAuth
from threading import Thread, Lock
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QMessageBox, QGroupBox, QSizePolicy,
    QComboBox, QFileDialog, QSlider, QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════
#  User Configuration
# ═══════════════════════════════════════════════════════════════════
USER_MODEL_PATH  = "./models/optimizations/final_opt3/model.pt"
USER_VIDEO_SOURCE = ""
CONFIDENCE_THRESHOLD = 0.50
USE_GPU     = True
CONFIG_FILE = "camera_config.json"

# ═══════════════════════════════════════════════════════════════════
#  Dark Theme
# ═══════════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f0f1a;
    color: #e0e0f0;
    font-family: 'Segoe UI', 'Ubuntu', sans-serif;
    font-size: 12px;
}
QGroupBox {
    background-color: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    color: #9090cc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;  padding: 0 6px;
    color: #a0a0dd;
}
QPushButton {
    background-color: #252545;
    color: #d0d0f0;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #35355a; border-color: #5a5a8a; }
QPushButton:pressed { background-color: #1a1a35; }
QPushButton:disabled { background-color: #1a1a2a; color: #555577; border-color: #2a2a3a; }
QLineEdit, QComboBox {
    background-color: #1a1a30;
    color: #e0e0f0;
    border: 1px solid #3a3a5a;
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: #6366f1;
}
QLineEdit:focus { border-color: #6366f1; }
QComboBox::drop-down { border: none; padding-right: 8px; }
QComboBox QAbstractItemView {
    background-color: #1a1a30; color: #e0e0f0;
    border: 1px solid #3a3a5a; selection-background-color: #6366f1;
}
QSlider::groove:horizontal {
    background: #1a1a30; height: 6px; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6366f1; width: 18px; height: 18px;
    margin: -6px 0; border-radius: 9px; border: 2px solid #818cf8;
}
QSlider::handle:horizontal:hover { background: #818cf8; border-color: #a5b4fc; }
QSlider::sub-page:horizontal { background: #4f46e5; border-radius: 3px; }
QTextEdit {
    background-color: #0a0a18; color: #40e090;
    border: 1px solid #2a2a4a; border-radius: 5px;
    font-family: 'Consolas', 'Courier New', monospace; font-size: 11px;
    padding: 4px;
}
QCheckBox { color: #c0c0e0; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #3a3a5a; background-color: #1a1a30;
}
QCheckBox::indicator:checked { background-color: #6366f1; border-color: #818cf8; }
QLabel { color: #c0c0e0; background: transparent; }
QToolTip {
    background-color: #252545; color: #e0e0f0;
    border: 1px solid #4a4a6a; padding: 5px; border-radius: 4px;
}
"""


# ═══════════════════════════════════════════════════════════════════
#  Threaded Camera Reader
# ═══════════════════════════════════════════════════════════════════
class CameraStream:
    def __init__(self, src):
        self.src = int(src) if str(src).isdigit() else src
        if sys.platform == "win32" and isinstance(self.src, int):
            self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            self.stream.set(cv2.CAP_PROP_FOURCC,
                            cv2.VideoWriter_fourcc(*"MJPG"))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.stream.set(cv2.CAP_PROP_FPS, 60)
        else:
            self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()
        Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy()
                              if self.frame is not None else None)

    def stop(self):
        self.stopped = True
        self.stream.release()


# ═══════════════════════════════════════════════════════════════════
#  GPU Inference Worker
# ═══════════════════════════════════════════════════════════════════
class GPUWorker(QThread):
    frame_processed = pyqtSignal(QImage, float, int)
    error_signal    = pyqtSignal(str)
    info_signal     = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, source, model_path, conf,
                 is_stream=True, resolution_scale=1.0):
        super().__init__()
        self.source           = source
        self.model_path       = model_path
        self.conf             = conf          # read each frame
        self.is_stream        = is_stream
        self.resolution_scale = resolution_scale  # read each frame
        self.is_recording     = False
        self.record_path      = ""
        self._running         = True
        self._video_writer    = None
        self.detect_drone     = True          # Target toggle
        self.detect_bird      = True          # Target toggle

    # ── main loop ─────────────────────────────────────────────────
    def run(self):                                       # noqa: C901
        device = 0 if USE_GPU and torch.cuda.is_available() else "cpu"
        if device == 0:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            self.info_signal.emit(
                f"🟢 GPU: {gpu_name}  "
                f"({vram / 1024**3:.1f} GB)")
        else:
            self.info_signal.emit("🟡 Running on CPU")

        try:
            model = YOLO(self.model_path)

            if self.is_stream:
                cap = CameraStream(self.source)
                if not cap.ret:
                    self.error_signal.emit(
                        f"Failed to open source: {self.source}")
                    return
            else:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    self.error_signal.emit(
                        f"Failed to open video: {self.source}")
                    return

            self.info_signal.emit("🟢 Inference running")
            t_prev = time.perf_counter()

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                h_orig, w_orig = frame.shape[:2]
                scale = self.resolution_scale

                # ── resize for inference ──
                if scale < 0.99:
                    w_inf = max(32, int(w_orig * scale))
                    h_inf = max(32, int(h_orig * scale))
                    inf_frame = cv2.resize(frame, (w_inf, h_inf),
                                           interpolation=cv2.INTER_LINEAR)
                else:
                    inf_frame = frame
                    w_inf, h_inf = w_orig, h_orig

                results = model.predict(
                    inf_frame, conf=self.conf, device=device,
                    half=(device == 0), verbose=False, max_det=100)

                # ── draw boxes on *original* frame ──
                det_count = 0
                if results[0].boxes is not None and len(results[0].boxes):
                    sx = w_orig / w_inf
                    sy = h_orig / h_inf
                    boxes   = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    confs   = results[0].boxes.conf.cpu().numpy()

                    for box, cls, c in zip(boxes, classes, confs):
                        label_name = results[0].names[cls]
                        label_lower = label_name.lower()
                        
                        # Apply dynamic class filtering
                        if "drone" in label_lower and not self.detect_drone:
                            continue
                        if "bird" in label_lower and not self.detect_bird:
                            continue
                            
                        det_count += 1
                        
                        x1 = int(box[0] * sx)
                        y1 = int(box[1] * sy)
                        x2 = int(box[2] * sx)
                        y2 = int(box[3] * sy)

                        color = ((0, 220, 0) if c >= 0.75
                                 else (0, 200, 255) if c >= 0.50
                                 else (0, 80, 255))
                        thick = 3 if c >= 0.70 else 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      color, thick)

                        label = f"{label_name} {c:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame,
                                      (x1, y1 - th - 8),
                                      (x1 + tw + 4, y1),
                                      color, -1)
                        cv2.putText(frame, label,
                                    (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 255, 255), 2)

                # ── recording ──
                if self.is_recording:
                    if self._video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self._video_writer = cv2.VideoWriter(
                            self.record_path, fourcc, 25.0, (w, h))
                    self._video_writer.write(frame)
                elif self._video_writer is not None:
                    self._video_writer.release()
                    self._video_writer = None

                # ── fps & emit ──
                t_now = time.perf_counter()
                fps = (1.0 / (t_now - t_prev)
                       if (t_now - t_prev) > 0 else 0)
                t_prev = t_now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_img = QImage(rgb.data, w, h, ch * w,
                                QImage.Format.Format_RGB888)
                self.frame_processed.emit(qt_img.copy(), fps, det_count)

            # ── cleanup ──
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None
            cap.stop() if self.is_stream else cap.release()
            self.finished_signal.emit()

        except Exception as exc:
            self.error_signal.emit(str(exc))

    def stop(self):
        self._running = False
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self.wait()


# ═══════════════════════════════════════════════════════════════════
#  PTZ / Camera Control Panel
# ═══════════════════════════════════════════════════════════════════
class CameraControlWidget(QWidget):
    log_signal          = pyqtSignal(str)
    login_status_signal = pyqtSignal(bool, int)

    def __init__(self):
        super().__init__()
        self.session_id   = 0
        self.http_session = requests.Session()
        self._req_id      = 0
        self.logged_in    = False
        self._stop_event  = threading.Event()
        self._setup_ui()
        self.load_config()
        self.log_signal.connect(self.append_log)
        self.login_status_signal.connect(self._update_ui_login)

    # ── UI ────────────────────────────────────────────────────────
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.setFixedWidth(360)

        # ▸ connection
        conn = QGroupBox("🔗 Connection")
        cl = QVBoxLayout(conn)

        self.ip_input   = QLineEdit("142.168.100.23")
        self.user_input = QLineEdit("admin")
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.save_pass_check = QCheckBox("Remember password")

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("IP:"));  r1.addWidget(self.ip_input)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("User:")); r2.addWidget(self.user_input)
        r2.addWidget(QLabel("Pass:")); r2.addWidget(self.pass_input)
        r3 = QHBoxLayout()
        r3.addWidget(self.save_pass_check)
        self.login_btn = QPushButton("🔑 Login")
        self.login_btn.clicked.connect(self.login)
        r3.addWidget(self.login_btn)
        r4 = QHBoxLayout()
        self.session_input = QLineEdit()
        r4.addWidget(QLabel("Session:")); r4.addWidget(self.session_input)
        btn_set = QPushButton("Set")
        btn_set.clicked.connect(self.set_session_manual)
        r4.addWidget(btn_set)

        self.status_label = QLabel("● Not connected")
        self.status_label.setStyleSheet(
            "color: #ef4444; font-weight: bold;")

        for r in (r1, r2, r3, r4):
            cl.addLayout(r)
        cl.addWidget(self.status_label)
        layout.addWidget(conn)

        # ▸ speed
        spd = QGroupBox("⚡ Speed")
        sl = QVBoxLayout(spd)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 64); self.speed_slider.setValue(8)
        self.speed_lbl = QLabel("Move: 8")
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_lbl.setText(f"Move: {v}"))
        self.zoom_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_speed_slider.setRange(1, 100)
        self.zoom_speed_slider.setValue(30)
        self.zoom_lbl = QLabel("Zoom: 30")
        self.zoom_speed_slider.valueChanged.connect(
            lambda v: self.zoom_lbl.setText(f"Zoom: {v}"))
        sl.addWidget(self.speed_lbl);      sl.addWidget(self.speed_slider)
        sl.addWidget(self.zoom_lbl);       sl.addWidget(self.zoom_speed_slider)
        layout.addWidget(spd)

        # ▸ d-pad + zoom
        ptz = QGroupBox("🎮 PTZ Control")
        pl = QVBoxLayout(ptz)

        BTN = 60, 40
        self.btn_up    = QPushButton("▲");  self.btn_up.setFixedSize(*BTN)
        self.btn_down  = QPushButton("▼");  self.btn_down.setFixedSize(*BTN)
        self.btn_left  = QPushButton("◄");  self.btn_left.setFixedSize(*BTN)
        self.btn_right = QPushButton("►");  self.btn_right.setFixedSize(*BTN)
        self.btn_stop  = QPushButton("STOP")
        self.btn_stop.setFixedSize(*BTN)
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#dc2626;color:white;font-weight:bold;"
            "border-radius:6px}"
            "QPushButton:hover{background:#ef4444}"
            "QPushButton:pressed{background:#b91c1c}")
        self.btn_stop.clicked.connect(self.emergency_stop)

        for row_w, widgets in (
            ("up",   [self.btn_up]),
            ("mid",  [self.btn_left, self.btn_stop, self.btn_right]),
            ("down", [self.btn_down]),
        ):
            h = QHBoxLayout()
            h.addStretch()
            for w in widgets:
                h.addWidget(w)
            h.addStretch()
            pl.addLayout(h)

        zr = QHBoxLayout()
        self.btn_zi = QPushButton("🔍+ Zoom In")
        self.btn_zo = QPushButton("🔍− Zoom Out")
        zr.addWidget(self.btn_zi); zr.addWidget(self.btn_zo)
        pl.addLayout(zr)
        layout.addWidget(ptz)

        # bindings  (Up/Down inverted per camera spec)
        self._bind_ptz(self.btn_up,    "Down")
        self._bind_ptz(self.btn_down,  "Up")
        self._bind_ptz(self.btn_left,  "Left")
        self._bind_ptz(self.btn_right, "Right")
        self.btn_zi.pressed.connect(lambda: self.zoom_start(1))
        self.btn_zi.released.connect(self.zoom_stop)
        self.btn_zo.pressed.connect(lambda: self.zoom_start(-1))
        self.btn_zo.released.connect(self.zoom_stop)

        # ▸ log
        lg = QGroupBox("📋 Log")
        ll = QVBoxLayout(lg)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        ll.addWidget(self.log_output)
        layout.addWidget(lg)

        layout.addStretch()

    # ── helpers ───────────────────────────────────────────────────
    def _bind_ptz(self, btn, code):
        btn.pressed.connect(lambda c=code:  self.start_move(c))
        btn.released.connect(lambda c=code: self.stop_move(c))

    def append_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{ts}] {msg}")

    # ── config persistence ────────────────────────────────────────
    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            self.ip_input.setText(cfg.get("ip", ""))
            self.user_input.setText(cfg.get("user", "admin"))
            self.save_pass_check.setChecked(cfg.get("save_pass", False))
            if cfg.get("save_pass"):
                self.pass_input.setText(cfg.get("password", ""))
        except Exception:
            pass

    def save_config(self):
        cfg = {
            "ip":        self.ip_input.text(),
            "user":      self.user_input.text(),
            "save_pass": self.save_pass_check.isChecked(),
            "password":  (self.pass_input.text()
                          if self.save_pass_check.isChecked() else ""),
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    # ── login / session ───────────────────────────────────────────
    def login(self):
        self.login_btn.setEnabled(False)
        threading.Thread(target=self._do_login, daemon=True).start()

    def _do_login(self):
        ip     = self.ip_input.text()
        user   = self.user_input.text()
        passwd = self.pass_input.text()
        url    = f"http://{ip}/SDK/UNIV_API"
        self._stop_event.set()
        self.http_session = requests.Session()
        try:
            payload = {
                "id": 1,
                "call": {"service": "session", "method": "login"},
                "params": {"userName": user, "password": ""},
            }
            resp = self.http_session.post(
                url, json=payload,
                auth=HTTPDigestAuth(user, passwd), timeout=5)
            data = resp.json()
            sid = self._extract_session(data)
            if sid is None and resp.cookies.get("WebSessionID"):
                sid = int(resp.cookies["WebSessionID"])
            if sid:
                self.session_id = sid
                self.http_session.cookies.set("WebSessionID", str(sid))
                self._stop_event.clear()
                threading.Thread(target=self._heartbeat,
                                 daemon=True).start()
                self.login_status_signal.emit(True, sid)
                self.save_config()
                self.log_signal.emit(f"Login OK  session={sid}")
            else:
                self.login_status_signal.emit(False, 0)
                self.log_signal.emit("Login failed: no session ID")
        except Exception as exc:
            self.login_status_signal.emit(False, 0)
            self.log_signal.emit(f"Login error: {exc}")

    def _extract_session(self, data):
        if not isinstance(data, dict):
            return None
        for k in ("session", "sessionId", "SessionID", "WebSessionID"):
            v = data.get(k)
            if v and v != 0:
                return int(v)
        for k in ("result", "data", "params"):
            if isinstance(data.get(k), dict):
                found = self._extract_session(data[k])
                if found:
                    return found
        return None

    def _update_ui_login(self, ok, sid):
        self.login_btn.setEnabled(True)
        if ok:
            self.status_label.setText(f"● Connected ({sid})")
            self.status_label.setStyleSheet(
                "color:#22c55e;font-weight:bold;")
            self.session_input.setText(str(sid))
            self.logged_in = True
        else:
            self.status_label.setText("● Login Failed")
            self.status_label.setStyleSheet(
                "color:#ef4444;font-weight:bold;")

    def _heartbeat(self):
        url = f"http://{self.ip_input.text()}/SDK/UNIV_API"
        while not self._stop_event.is_set():
            try:
                self.http_session.post(url, json={
                    "call": {"service": "session",
                             "method": "heartbeat"},
                    "params": {"clientTag": "GUI"},
                    "session": self.session_id, "id": 999,
                }, timeout=2)
            except Exception:
                pass
            time.sleep(3)

    def set_session_manual(self):
        try:
            self.session_id = int(self.session_input.text())
            self.http_session.cookies.set(
                "WebSessionID", str(self.session_id))
            self.login_status_signal.emit(True, self.session_id)
            self.log_signal.emit(
                f"Manual session: {self.session_id}")
        except Exception:
            pass

    # ── movement ──────────────────────────────────────────────────
    def start_move(self, code):
        url = (f"http://{self.ip_input.text()}/cgi-bin/ptz.cgi?"
               f"action=PTZControl&channel=0&code={code}"
               f"&arg1=1&arg2={self.speed_slider.value()}"
               f"&arg3=0&arg4=0")
        threading.Thread(
            target=lambda: self._get(url), daemon=True).start()

    def stop_move(self, code):
        url = (f"http://{self.ip_input.text()}/cgi-bin/ptz.cgi?"
               f"action=PTZControl&channel=0&code={code}"
               f"&arg1=0&arg2=0&arg3=0&arg4=0")
        threading.Thread(
            target=lambda: self._get(url), daemon=True).start()

    def zoom_start(self, direction):
        self._send_zoom(self.zoom_speed_slider.value() * direction)

    def zoom_stop(self):
        self._send_zoom(0)

    def _send_zoom(self, z):
        self._req_id += 1
        url = f"http://{self.ip_input.text()}/SDK/UNIV_API"
        payload = {
            "session": self.session_id, "id": self._req_id,
            "call": {"service": "ptz", "method": "setPTZCmd"},
            "params": {"channel": 0,
                       "continuousZoomSpace": {"z": z}},
        }
        threading.Thread(
            target=lambda: self._post(url, payload),
            daemon=True).start()

    def emergency_stop(self):
        self.stop_move("Up")
        self.zoom_stop()
        self.log_signal.emit("⛔ Emergency stop")

    # ── safe HTTP ─────────────────────────────────────────────────
    def _get(self, url):
        try:
            self.http_session.get(url, timeout=1)
        except Exception:
            pass

    def _post(self, url, payload):
        try:
            self.http_session.post(url, json=payload, timeout=1)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
#  Main Application Window
# ═══════════════════════════════════════════════════════════════════
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛸  Drone Detector  &  PTZ Controller")
        self.resize(1600, 900)
        self.worker          = None
        self._last_pixmap    = None
        self._peak_det       = 0
        self._total_frames   = 0
        self._uptime_start   = None
        self._setup_ui()
        self._uptime_timer = QTimer()
        self._uptime_timer.timeout.connect(self._tick_uptime)

    # ── build UI ──────────────────────────────────────────────────
    def _setup_ui(self):                                 # noqa: C901
        cw = QWidget();  self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setSpacing(10); root.setContentsMargins(10, 10, 10, 10)

        # ── left pane ──
        left = QWidget();  lv = QVBoxLayout(left)
        lv.setSpacing(6);  lv.setContentsMargins(0, 0, 0, 0)

        # status
        self.lbl_status = QLabel("⏸  Ready")
        self.lbl_status.setStyleSheet(
            "background:#1a1a2e;color:#22c55e;padding:8px 12px;"
            "font-weight:bold;font-size:13px;border-radius:6px;"
            "border:1px solid #2a2a4a;")
        lv.addWidget(self.lbl_status)

        # stats bar
        stats = QWidget()
        sh = QHBoxLayout(stats)
        sh.setContentsMargins(0, 0, 0, 0); sh.setSpacing(5)
        self.st_fps   = self._pill("FPS: --",      "#6366f1",
                                   "Frames per second")
        self.st_det   = self._pill("Det: 0",       "#f59e0b",
                                   "Detections in current frame")
        self.st_peak  = self._pill("Peak: 0",      "#ef4444",
                                   "Max detections seen in one frame")
        self.st_up    = self._pill("⏱ 00:00:00",   "#22c55e",
                                   "Session uptime")
        self.st_res   = self._pill("Res: 100 %",   "#8b5cf6",
                                   "Inference resolution scale")
        self.st_rec   = self._pill("⏹ REC",        "#555577",
                                   "Recording indicator")
        for w in (self.st_fps, self.st_det, self.st_peak,
                  self.st_up,  self.st_res,  self.st_rec):
            sh.addWidget(w)
        sh.addStretch()
        lv.addWidget(stats)

        # video
        self.video_label = QLabel("No Video Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#0a0a14;border:2px solid #2a2a4a;"
            "border-radius:8px;color:#4a4a6a;"
            "font-size:20px;font-weight:bold;")
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        lv.addWidget(self.video_label, 1)

        # ── settings group ──
        sg = QGroupBox("⚙  Configuration")
        sv = QVBoxLayout(sg)

        # row 1 — mode / source
        r1 = QHBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["📡  Live Stream", "📁  Video File"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode)
        r1.addWidget(QLabel("Mode:")); r1.addWidget(self.combo_mode)
        self.input_src = QLineEdit(USER_VIDEO_SOURCE)
        self.input_src.setPlaceholderText("0  or  rtsp://...")
        r1.addWidget(QLabel("Source:")); r1.addWidget(self.input_src, 1)
        self.btn_browse = QPushButton("📂 Browse")
        self.btn_browse.setEnabled(False)
        self.btn_browse.clicked.connect(self._browse)
        r1.addWidget(self.btn_browse)
        sv.addLayout(r1)

        # row 2 — model
        r2 = QHBoxLayout()
        self.input_model = QLineEdit(USER_MODEL_PATH)
        r2.addWidget(QLabel("Model:")); r2.addWidget(self.input_model, 1)
        sv.addLayout(r2)

        # row 3 — sliders
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Inference Res:"))
        self.sld_res = QSlider(Qt.Orientation.Horizontal)
        self.sld_res.setRange(30, 100); self.sld_res.setValue(100)
        self.sld_res.setTickPosition(
            QSlider.TickPosition.TicksBelow)
        self.sld_res.setTickInterval(10)
        self.sld_res.setToolTip(
            "Lower = faster inference, less accuracy")
        self.lbl_res_v = QLabel("100 %")
        self.lbl_res_v.setFixedWidth(42)
        self.lbl_res_v.setStyleSheet(
            "color:#8b5cf6;font-weight:bold;")
        self.sld_res.valueChanged.connect(self._on_res)
        r3.addWidget(self.sld_res, 1); r3.addWidget(self.lbl_res_v)

        r3.addSpacing(20)

        r3.addWidget(QLabel("Confidence:"))
        self.sld_conf = QSlider(Qt.Orientation.Horizontal)
        self.sld_conf.setRange(10, 95)
        self.sld_conf.setValue(int(CONFIDENCE_THRESHOLD * 100))
        self.sld_conf.setTickPosition(
            QSlider.TickPosition.TicksBelow)
        self.sld_conf.setTickInterval(10)
        self.sld_conf.setToolTip(
            "Higher = fewer but more confident detections")
        self.lbl_conf_v = QLabel(f"{CONFIDENCE_THRESHOLD:.2f}")
        self.lbl_conf_v.setFixedWidth(35)
        self.lbl_conf_v.setStyleSheet(
            "color:#f59e0b;font-weight:bold;")
        self.sld_conf.valueChanged.connect(self._on_conf)
        r3.addWidget(self.sld_conf, 1); r3.addWidget(self.lbl_conf_v)
        sv.addLayout(r3)

        # row 4 — action buttons
        r4 = QHBoxLayout()
        self.btn_start = QPushButton("▶  START")
        self.btn_start.setStyleSheet(
            "QPushButton{background:#4f46e5;color:white;"
            "font-weight:bold;padding:8px 20px;border-radius:6px;"
            "font-size:13px}"
            "QPushButton:hover{background:#6366f1}")
        self.btn_start.clicked.connect(self.toggle_inference)
        r4.addWidget(self.btn_start)

        self.btn_snap = QPushButton("📸  Screenshot")
        self.btn_snap.setToolTip("Save current frame as PNG")
        self.btn_snap.clicked.connect(self._screenshot)
        r4.addWidget(self.btn_snap)

        self.btn_rec = QPushButton("⏺  Record")
        self.btn_rec.setToolTip("Toggle video recording")
        self.btn_rec.clicked.connect(self._toggle_rec)
        r4.addWidget(self.btn_rec)

        self.chk_top = QCheckBox("📌 On Top")
        self.chk_top.setToolTip("Keep window above others")
        self.chk_top.toggled.connect(self._on_top)
        r4.addWidget(self.chk_top)

        r4.addStretch()
        self.lbl_fps_big = QLabel("FPS: --")
        self.lbl_fps_big.setStyleSheet(
            "font-size:16px;font-weight:bold;color:#6366f1;")
        r4.addWidget(self.lbl_fps_big)
        sv.addLayout(r4)

        lv.addWidget(sg)
        root.addWidget(left, 1)

        # ── right pane ──
        right_pane = QWidget()
        rv = QVBoxLayout(right_pane)
        rv.setSpacing(10)
        rv.setContentsMargins(0, 0, 0, 0)

        # Target classes
        cg = QGroupBox("🎯 Detection Classes")
        cl = QVBoxLayout(cg)
        self.chk_drone = QCheckBox("🚁 Detect Drones")
        self.chk_drone.setChecked(True)
        self.chk_drone.toggled.connect(self._on_classes_changed)
        self.chk_bird = QCheckBox("🐦 Detect Birds")
        self.chk_bird.setChecked(True)
        self.chk_bird.toggled.connect(self._on_classes_changed)
        cl.addWidget(self.chk_drone)
        cl.addWidget(self.chk_bird)
        rv.addWidget(cg)

        # PTZ Panel
        self.ptz_ctrl = CameraControlWidget()
        rv.addWidget(self.ptz_ctrl)

        root.addWidget(right_pane)

    # ── tiny helper ───────────────────────────────────────────────
    @staticmethod
    def _pill(text, colour, tip=""):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setToolTip(tip)
        lbl.setStyleSheet(
            f"background-color:#1a1a2e;color:{colour};"
            f"padding:4px 10px;border-radius:4px;font-weight:bold;"
            f"font-size:11px;border:1px solid #2a2a4a;")
        return lbl

    # ── slot: class filters ───────────────────────────────────────
    def _on_classes_changed(self):
        if self.worker:
            self.worker.detect_drone = self.chk_drone.isChecked()
            self.worker.detect_bird = self.chk_bird.isChecked()

    # ── slot: mode ────────────────────────────────────────────────
    def _on_mode(self, idx):
        self.btn_browse.setEnabled(idx == 1)
        self.input_src.setPlaceholderText(
            "0  or  rtsp://..." if idx == 0
            else "Select video file...")

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if p:
            self.input_src.setText(p)

    # ── slot: sliders ─────────────────────────────────────────────
    def _on_res(self, v):
        self.lbl_res_v.setText(f"{v} %")
        self.st_res.setText(f"Res: {v} %")
        if self.worker:
            self.worker.resolution_scale = v / 100.0

    def _on_conf(self, v):
        c = v / 100.0
        self.lbl_conf_v.setText(f"{c:.2f}")
        if self.worker:
            self.worker.conf = c

    # ── start / stop inference ────────────────────────────────────
    def toggle_inference(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            self.btn_start.setText("▶  START")
            self.btn_start.setStyleSheet(
                "QPushButton{background:#4f46e5;color:white;"
                "font-weight:bold;padding:8px 20px;"
                "border-radius:6px;font-size:13px}"
                "QPushButton:hover{background:#6366f1}")
            self.lbl_status.setText("⏸  Stopped")
            self._uptime_timer.stop()
            self._peak_det = 0
            self._total_frames = 0
        else:
            self.worker = GPUWorker(
                source           = self.input_src.text(),
                model_path       = self.input_model.text(),
                conf             = self.sld_conf.value() / 100.0,
                is_stream        = self.combo_mode.currentIndex() == 0,
                resolution_scale = self.sld_res.value() / 100.0,
            )
            self.worker.detect_drone = self.chk_drone.isChecked()
            self.worker.detect_bird = self.chk_bird.isChecked()
            self.worker.frame_processed.connect(self.update_frame)
            self.worker.info_signal.connect(self.lbl_status.setText)
            self.worker.error_signal.connect(
                lambda e: QMessageBox.critical(self, "Error", e))
            self.worker.finished_signal.connect(self.toggle_inference)
            self.worker.start()
            self.btn_start.setText("⏹  STOP")
            self.btn_start.setStyleSheet(
                "QPushButton{background:#dc2626;color:white;"
                "font-weight:bold;padding:8px 20px;"
                "border-radius:6px;font-size:13px}"
                "QPushButton:hover{background:#ef4444}")
            self._uptime_start = time.time()
            self._uptime_timer.start(1000)
            self._peak_det = 0
            self._total_frames = 0

    # ── frame display ─────────────────────────────────────────────
    _PILL_NORMAL = ("background-color:#1a1a2e;color:{c};"
                    "padding:4px 10px;border-radius:4px;"
                    "font-weight:bold;font-size:11px;"
                    "border:1px solid #2a2a4a;")
    _PILL_ALERT  = ("background-color:#451a03;color:#f59e0b;"
                    "padding:4px 10px;border-radius:4px;"
                    "font-weight:bold;font-size:11px;"
                    "border:1px solid #f59e0b;")

    @pyqtSlot(QImage, float, int)
    def update_frame(self, img, fps, count):
        self._total_frames += 1
        self._peak_det = max(self._peak_det, count)

        self.st_fps.setText(f"FPS: {fps:.0f}")
        self.lbl_fps_big.setText(f"FPS: {fps:.0f}")
        self.st_det.setText(f"Det: {count}")
        self.st_peak.setText(f"Peak: {self._peak_det}")

        self.st_det.setStyleSheet(
            self._PILL_ALERT if count
            else self._PILL_NORMAL.format(c="#f59e0b"))

        if self.video_label.width() > 0:
            pix = QPixmap.fromImage(img).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation)
            self.video_label.setPixmap(pix)
            self._last_pixmap = pix

    # ── uptime ────────────────────────────────────────────────────
    def _tick_uptime(self):
        if self._uptime_start:
            e = time.time() - self._uptime_start
            self.st_up.setText(
                f"⏱ {int(e//3600):02d}:{int(e%3600//60):02d}"
                f":{int(e%60):02d}")

    # ── screenshot ────────────────────────────────────────────────
    def _screenshot(self):
        if self._last_pixmap:
            os.makedirs("screenshots", exist_ok=True)
            fn = (f"screenshots/capture_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            self._last_pixmap.save(fn)
            self.lbl_status.setText(f"📸  Saved: {fn}")
            self.ptz_ctrl.log_signal.emit(f"Screenshot → {fn}")
        else:
            self.lbl_status.setText("⚠  No frame to capture")

    # ── recording ─────────────────────────────────────────────────
    def _toggle_rec(self):
        if not self.worker:
            self.lbl_status.setText("⚠  Start inference first")
            return
        if self.worker.is_recording:
            self.worker.is_recording = False
            self.btn_rec.setText("⏺  Record")
            self.btn_rec.setStyleSheet("")
            self.st_rec.setText("⏹ REC")
            self.st_rec.setStyleSheet(
                self._PILL_NORMAL.format(c="#555577"))
            self.lbl_status.setText("⏹  Recording stopped")
            self.ptz_ctrl.log_signal.emit("Recording stopped")
        else:
            os.makedirs("recordings", exist_ok=True)
            fn = (f"recordings/rec_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            self.worker.record_path  = fn
            self.worker.is_recording = True
            self.btn_rec.setText("⏹  Stop Rec")
            self.btn_rec.setStyleSheet(
                "QPushButton{background:#dc2626;color:white;"
                "font-weight:bold;border-radius:6px}"
                "QPushButton:hover{background:#ef4444}")
            self.st_rec.setText("🔴 REC")
            self.st_rec.setStyleSheet(
                "background-color:#450a0a;color:#ef4444;"
                "padding:4px 10px;border-radius:4px;"
                "font-weight:bold;font-size:11px;"
                "border:1px solid #ef4444;")
            self.lbl_status.setText(f"🔴  Recording → {fn}")
            self.ptz_ctrl.log_signal.emit(f"Recording → {fn}")

    # ── always on top ─────────────────────────────────────────────
    def _on_top(self, on):
        flags = self.windowFlags()
        if on:
            self.setWindowFlags(
                flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(
                flags & ~Qt.WindowType.WindowStaysOnTopHint)
        self.show()

    # ── shutdown ──────────────────────────────────────────────────
    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        self.ptz_ctrl._stop_event.set()
        event.accept()


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = App()
    window.show()
    sys.exit(app.exec())
