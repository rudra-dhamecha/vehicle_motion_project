"""
Vehicle Motion Estimation — GUI (TTKBootstrap).
Entry point: launch_gui()
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, TclError

# Optional: drag-and-drop from file manager
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

from optical_flow import OpticalFlowProcessor


def _normalize_drop_path(path):
    """Convert DnD path (e.g. file:///path) to a normal filesystem path."""
    path = (path or "").strip()
    if path.startswith("file://"):
        path = path[7:]
    if path.startswith("//"):
        path = path[2:]
    return path


def _choose_video_file(parent):
    """Open a file picker for video files. Returns path or None."""
    path = filedialog.askopenfilename(
        parent=parent,
        title="Select a video file",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("MP4", "*.mp4"),
            ("All files", "*.*"),
        ],
    )
    return path if path and path.strip() else None


def launch_gui():
    if _HAS_DND:
        root = TkinterDnD.Tk()
        ttk.Style(theme="darkly")
        root.title("Vehicle Motion Estimation")
        root.geometry("1280x760")
        root.minsize(900, 600)
        root.configure(bg="#222222")
    else:
        root = ttk.Window(
            title="Vehicle Motion Estimation",
            themename="darkly",
            size=(1280, 760),
            minsize=(900, 600),
        )
    app = MotionApp(root)
    if _HAS_DND:
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", app._on_drop)
    root.mainloop()


class MotionApp:
    def __init__(self, root):
        self.root = root
        self.processor = OpticalFlowProcessor()
        self.cap = None
        self.playing = False
        self._current_path = None
        self._photo_refs = []

        # Outer container (website-style padding)
        self.container = ttk.Frame(root, padding=24)
        self.container.pack(fill=BOTH, expand=True)

        # ----- Header -----
        header = ttk.Frame(self.container)
        header.pack(fill=X, pady=(0, 8))

        ttk.Label(
            header,
            text="Vehicle Motion Estimation",
            font=("Segoe UI", 26, "bold"),
            bootstyle="light",
        ).pack(anchor=W)

        ttk.Label(
            header,
            text="Optical flow visualization · Drop a video or click to browse",
            font=("Segoe UI", 11),
            bootstyle="light",
        ).pack(anchor=W)

        # ----- Drop zone + Browse (shown when no video loaded) -----
        self.drop_frame = ttk.Frame(self.container)
        self.drop_frame.pack(fill=BOTH, expand=True, pady=16)

        self.drop_zone = ttk.Frame(self.drop_frame)
        self.drop_zone.pack(fill=BOTH, expand=True)
        self.drop_zone.configure(bootstyle="dark")

        # Visual drop area (clickable)
        self.drop_inner = ttk.Frame(self.drop_zone, padding=48)
        self.drop_inner.pack(expand=True, fill=BOTH, padx=32, pady=32)
        self.drop_inner.bind("<Button-1>", lambda e: self._open_and_load())
        self.drop_inner.bind("<Enter>", self._drop_enter)
        self.drop_inner.bind("<Leave>", self._drop_leave)

        ttk.Label(
            self.drop_inner,
            text="Drop video here or click to browse",
            font=("Segoe UI", 16),
            bootstyle="light",
        ).pack(pady=(0, 8))
        ttk.Label(
            self.drop_inner,
            text="MP4, AVI, MOV, MKV, WebM",
            font=("Segoe UI", 11),
            bootstyle="light",
        ).pack()

        ttk.Separator(self.drop_frame, bootstyle="light").pack(fill=X, pady=16)

        browse_btn = ttk.Button(
            self.drop_frame,
            text="Browse for file…",
            bootstyle="light-outline",
            command=self._open_and_load,
        )
        browse_btn.pack(pady=8)

        # ----- Video area (hidden until file loaded) -----
        self.video_frame = ttk.Frame(self.container)
        # not packed initially

        # Top bar: change video + play/pause
        bar = ttk.Frame(self.video_frame)
        bar.pack(fill=X, pady=(0, 12))

        ttk.Button(
            bar,
            text="Change video",
            bootstyle="light-link",
            command=self._change_video,
        ).pack(side=LEFT, padx=(0, 16))

        self.play_btn = ttk.Button(
            bar,
            text="Play",
            bootstyle="success",
            command=self.toggle_play,
        )
        self.play_btn.pack(side=LEFT)

        self.motion_label = ttk.Label(
            bar,
            text="Motion: —",
            font=("Segoe UI", 11),
            bootstyle="light",
        )
        self.motion_label.pack(side=LEFT, padx=(24, 0))

        # Two panels: input | optical flow
        panels = ttk.Frame(self.video_frame)
        panels.pack(fill=BOTH, expand=True, pady=8)

        left_card = ttk.Labelframe(panels, text="Input", bootstyle="primary", padding=8)
        left_card.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))
        self.input_label = ttk.Label(left_card)
        self.input_label.pack(fill=BOTH, expand=True)

        right_card = ttk.Labelframe(
            panels, text="Optical flow", bootstyle="info", padding=8
        )
        right_card.pack(side=LEFT, fill=BOTH, expand=True, padx=(8, 0))
        self.output_label = ttk.Label(right_card)
        self.output_label.pack(fill=BOTH, expand=True)

    def _drop_enter(self, event):
        try:
            self.drop_inner.configure(bootstyle="primary")
        except TclError:
            pass

    def _drop_leave(self, event):
        try:
            self.drop_inner.configure(bootstyle="dark")
        except TclError:
            pass

    def _on_drop(self, event):
        """Handle files dropped from file manager. Uses tk.splitlist for correct parsing."""
        if not _HAS_DND or not event.data:
            return
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        for raw in paths:
            path = _normalize_drop_path(raw)
            if path and os.path.isfile(path):
                self._load_video(path)
                return

    def _open_and_load(self):
        path = _choose_video_file(self.root)
        if path:
            self._load_video(path)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        if self.cap:
            self.cap.release()
        self.cap = cap
        self._current_path = path
        self.processor.prev_gray = None
        self.playing = False
        self.play_btn.config(text="Play")

        # Show first frame
        ret, frame = self.cap.read()
        if ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            processed = self.processor.process_frame(frame)
            self._show_frames(frame, processed)
            self.motion_label.config(text="Motion: —")

        # Switch to video view
        self.drop_frame.pack_forget()
        self.video_frame.pack(fill=BOTH, expand=True)

    def _change_video(self):
        path = _choose_video_file(self.root)
        if path:
            self._load_video(path)

    def _show_frames(self, frame_bgr, processed_bgr, max_display_size=(640, 360)):
        """Scale frames to fit display and show in both panels."""
        h, w = frame_bgr.shape[:2]
        mw, mh = max_display_size
        scale = min(mw / w, mh / h, 1.0)
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            processed_bgr = cv2.resize(
                processed_bgr, (nw, nh), interpolation=cv2.INTER_AREA
            )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        self._photo_refs[:] = []
        img1 = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        img2 = ImageTk.PhotoImage(Image.fromarray(processed_rgb))
        self._photo_refs.extend([img1, img2])
        self.input_label.configure(image=img1)
        self.output_label.configure(image=img2)

    def toggle_play(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="Pause")
            self.update_frame()
        else:
            self.play_btn.config(text="Play")

    def update_frame(self):
        if not self.playing or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            self.play_btn.config(text="Play")
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        processed = self.processor.process_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.processor.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.processor.prev_gray,
                gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg = np.mean(magnitude)
            self.motion_label.config(text=f"Motion: {avg:.4f}")

        self._show_frames(frame, processed)
        self.root.after(30, self.update_frame)
