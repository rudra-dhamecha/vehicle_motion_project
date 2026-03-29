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

from optical_flow import (
    OpticalFlowProcessor,
    CarDetector,
    draw_detections,
)


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
        root.configure(bg="#10131a")
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
        self._detector = None  # Lazy-loaded on first video
        self.cap = None
        self.playing = False
        self._current_path = None
        self._photo_refs = []
        self._max_preview_size = (540, 360)
        self._current_frame_idx = 0
        self._frame_delay_ms = 50

        self.container = ttk.Frame(root, padding=14)
        self.container.pack(fill=BOTH, expand=True)

        hero = ttk.Frame(self.container, bootstyle="secondary")
        hero.pack(fill=X, pady=(0, 12))

        ttk.Label(
            hero,
            text="Vehicle Motion Estimation",
            font=("Segoe UI", 24, "bold"),
            bootstyle="light",
        ).pack(anchor=W, padx=14, pady=(10, 2))

        ttk.Label(
            hero,
            text="Professional dark dashboard · Optical flow and detections in real-time",
            font=("Segoe UI", 10),
            bootstyle="secondary",
        ).pack(anchor=W, padx=14, pady=(0, 10))

        shell = ttk.Frame(self.container)
        shell.pack(fill=BOTH, expand=True)

        # Left control rail
        rail = ttk.Labelframe(shell, text="Control Panel", bootstyle="secondary", padding=12)
        rail.pack(side=LEFT, fill=Y, padx=(0, 10))
        rail.configure(width=280)
        rail.pack_propagate(False)

        self.open_btn = ttk.Button(
            rail,
            text="Open Video",
            bootstyle="primary",
            command=self._open_and_load,
            width=24,
        )
        self.open_btn.pack(fill=X, pady=(0, 8))

        self.change_btn = ttk.Button(
            rail,
            text="Change Video",
            bootstyle="secondary-outline",
            command=self._change_video,
            state=DISABLED,
            width=24,
        )
        self.change_btn.pack(fill=X, pady=(0, 8))

        self.play_btn = ttk.Button(
            rail,
            text="Play",
            bootstyle="success",
            command=self.toggle_play,
            state=DISABLED,
            width=24,
        )
        self.play_btn.pack(fill=X, pady=(0, 14))

        ttk.Separator(rail, bootstyle="secondary").pack(fill=X, pady=(0, 12))

        ttk.Label(
            rail,
            text="CURRENT VIDEO",
            font=("Segoe UI", 8, "bold"),
            bootstyle="secondary",
        ).pack(anchor=W)
        self.video_name_label = ttk.Label(
            rail,
            text="No file loaded",
            font=("Segoe UI", 10),
            bootstyle="light",
            wraplength=240,
        )
        self.video_name_label.pack(anchor=W, fill=X, pady=(3, 10))

        stat_card = ttk.Frame(rail, bootstyle="dark")
        stat_card.pack(fill=X, pady=(0, 10))
        self.motion_label = ttk.Label(
            stat_card,
            text="Motion: —",
            font=("Segoe UI Semibold", 10),
            bootstyle="info",
        )
        self.motion_label.pack(anchor=W, padx=10, pady=(8, 2))
        self.detection_count_label = ttk.Label(
            stat_card,
            text="Detections: —",
            font=("Segoe UI Semibold", 10),
            bootstyle="success",
        )
        self.detection_count_label.pack(anchor=W, padx=10, pady=2)
        self.frame_label = ttk.Label(
            stat_card,
            text="Frame: —",
            font=("Segoe UI", 10),
            bootstyle="secondary",
        )
        self.frame_label.pack(anchor=W, padx=10, pady=(2, 8))

        ttk.Label(
            rail,
            text="Drag & drop a file anywhere in this window.",
            font=("Segoe UI", 9),
            bootstyle="secondary",
            wraplength=240,
        ).pack(anchor=W, fill=X, pady=(2, 10))

        self.status_label = ttk.Label(
            rail,
            text="Ready. Load a video to begin analysis.",
            bootstyle="secondary",
            anchor=W,
            wraplength=240,
        )
        self.status_label.pack(fill=X)

        # Right workspace
        workspace = ttk.Labelframe(
            shell, text="Visualization Workspace", bootstyle="primary", padding=10
        )
        workspace.pack(side=LEFT, fill=BOTH, expand=True)

        # Drop zone + Browse (shown when no video loaded)
        self.drop_frame = ttk.Frame(workspace)
        self.drop_frame.pack(fill=BOTH, expand=True)

        self.drop_zone = ttk.Labelframe(self.drop_frame, text="Drop Zone", bootstyle="info", padding=14)
        self.drop_zone.pack(fill=BOTH, expand=True)

        self.drop_inner = ttk.Frame(self.drop_zone, padding=56, bootstyle="dark")
        self.drop_inner.pack(expand=True, fill=BOTH, padx=30, pady=30)
        self.drop_inner.bind("<Button-1>", lambda e: self._open_and_load())
        self.drop_inner.bind("<Enter>", self._drop_enter)
        self.drop_inner.bind("<Leave>", self._drop_leave)

        ttk.Label(
            self.drop_inner,
            text="DROP YOUR VIDEO HERE",
            font=("Segoe UI", 18, "bold"),
            bootstyle="light",
        ).pack(pady=(0, 8))
        ttk.Label(
            self.drop_inner,
            text="or click Open Video from the control panel",
            font=("Segoe UI", 11),
            bootstyle="secondary",
        ).pack()
        ttk.Label(
            self.drop_inner,
            text="MP4 · AVI · MOV · MKV · WebM",
            font=("Segoe UI", 9, "bold"),
            bootstyle="info",
        ).pack(pady=(14, 2))
        ttk.Label(
            self.drop_inner,
            text="Shortcut: Space toggles Play/Pause",
            font=("Segoe UI", 9),
            bootstyle="secondary",
        ).pack()

        # Video area (hidden until file loaded)
        self.video_frame = ttk.Frame(workspace)

        # Three panels: input | optical flow | object detection
        panels = ttk.Frame(self.video_frame)
        panels.pack(fill=BOTH, expand=True, pady=2)
        panels.columnconfigure(0, weight=1, uniform="video")
        panels.columnconfigure(1, weight=1, uniform="video")
        panels.columnconfigure(2, weight=1, uniform="video")
        panels.rowconfigure(0, weight=1)

        left_card = ttk.Labelframe(
            panels, text="Input Video", bootstyle="primary", padding=8
        )
        left_card.grid(row=0, column=0, sticky=NSEW, padx=(0, 6))
        left_card.rowconfigure(0, weight=1)
        left_card.columnconfigure(0, weight=1)
        self.input_label = ttk.Label(left_card, anchor=CENTER, bootstyle="dark")
        self.input_label.pack(fill=BOTH, expand=True)

        mid_card = ttk.Labelframe(
            panels, text="Optical Flow", bootstyle="info", padding=8
        )
        mid_card.grid(row=0, column=1, sticky=NSEW, padx=6)
        mid_card.rowconfigure(0, weight=1)
        mid_card.columnconfigure(0, weight=1)
        self.output_label = ttk.Label(mid_card, anchor=CENTER, bootstyle="dark")
        self.output_label.pack(fill=BOTH, expand=True)

        right_card = ttk.Labelframe(
            panels, text="Object Detection", bootstyle="success", padding=8
        )
        right_card.grid(row=0, column=2, sticky=NSEW, padx=(6, 0))
        right_card.rowconfigure(0, weight=1)
        right_card.columnconfigure(0, weight=1)
        self.detection_label = ttk.Label(right_card, anchor=CENTER, bootstyle="dark")
        self.detection_label.pack(fill=BOTH, expand=True)

        self.input_label.configure(text="Input preview")
        self.output_label.configure(text="Optical flow preview")
        self.detection_label.configure(text="Detection preview")

        self.root.bind("<space>", self._on_space_toggle)

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
            self.status_label.config(text="Could not open selected file.")
            return
        if self.cap:
            self.cap.release()
        self.cap = cap
        self._current_path = path
        self._current_frame_idx = 0
        self.processor.prev_gray = None
        self.playing = False
        if self._detector is None:
            self._detector = CarDetector(conf_threshold=0.4)
        self.play_btn.config(text="Play")
        self.play_btn.config(state=NORMAL)
        self.change_btn.config(state=NORMAL)
        self.video_name_label.config(text=os.path.basename(path))
        self.status_label.config(text="Video loaded. Press Play to start.")

        # Show first frame
        ret, frame = self.cap.read()
        if ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            processed, mag, ang = self.processor.process_frame(frame)
            detections = self._detector.detect(frame)
            detection_frame = draw_detections(frame, detections, mag, ang)
            self._show_frames(frame, processed, detection_frame, len(detections))
            self.motion_label.config(text="Motion: —")
            self.detection_count_label.config(text=f"Detections: {len(detections)}")
            self.frame_label.config(text="Frame: 0")

        # Switch to video view
        self.drop_frame.pack_forget()
        self.video_frame.pack(fill=BOTH, expand=True)

    def _change_video(self):
        path = _choose_video_file(self.root)
        if path:
            self._load_video(path)

    def _show_frames(
        self, frame_bgr, processed_bgr, detection_bgr, detection_count, max_display_size=None
    ):
        """Scale each frame to the current panel size and show all three views."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        detection_rgb = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2RGB)
        self._photo_refs[:] = []
        img1 = self._create_panel_image(frame_rgb, self.input_label, max_display_size)
        img2 = self._create_panel_image(processed_rgb, self.output_label, max_display_size)
        img3 = self._create_panel_image(detection_rgb, self.detection_label, max_display_size)
        self._photo_refs.extend([img1, img2, img3])
        self.input_label.configure(image=img1, text="")
        self.output_label.configure(image=img2, text="")
        self.detection_label.configure(image=img3, text="")
        self.detection_count_label.config(text=f"Detections: {detection_count}")

    def _create_panel_image(self, rgb_frame, widget, max_display_size):
        if max_display_size is None:
            max_display_size = self._max_preview_size
        max_w, max_h = max_display_size


        target_w = widget.winfo_width()
        target_h = widget.winfo_height()
        if target_w < 20 or target_h < 20:
            target_w, target_h = max_w, max_h

        frame_h, frame_w = rgb_frame.shape[:2]
        scale = min(target_w / frame_w, target_h / frame_h, 1.0)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))

        if new_w != frame_w or new_h != frame_h:
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return ImageTk.PhotoImage(Image.fromarray(rgb_frame))

    def toggle_play(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_btn.config(text="Pause")
            self.status_label.config(text="Processing video...")
            self.update_frame()
        else:
            self.play_btn.config(text="Play")
            self.status_label.config(text="Paused.")

    def update_frame(self):
        if not self.playing or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            self.play_btn.config(text="Play")
            self.status_label.config(text="Reached end of video. Rewound to frame 0.")
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame_idx = 0
            self.frame_label.config(text="Frame: 0")
            return

        processed, mag, ang = self.processor.process_frame(frame)

        if mag is not None:
            avg = np.mean(mag)
            self.motion_label.config(text=f"Motion: {avg:.4f}")

        detections = self._detector.detect(frame)
        detection_frame = draw_detections(frame, detections, mag, ang)
        self._current_frame_idx += 1
        self.frame_label.config(text=f"Frame: {self._current_frame_idx}")

        self._show_frames(frame, processed, detection_frame, len(detections))
        self.root.after(self._frame_delay_ms, self.update_frame)

    def _on_space_toggle(self, _event):
        if self.cap is not None:
            self.toggle_play()
