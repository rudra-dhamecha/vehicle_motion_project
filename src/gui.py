import cv2
import numpy as np
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from optical_flow import OpticalFlowProcessor


# class MotionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Vehicle Motion Estimation - Phase 2")
#         self.root.geometry("1200x600")

#         self.processor = OpticalFlowProcessor()
#         self.cap = None

#         # Top Frame (Controls)
#         control_frame = ttk.Frame(root)
#         control_frame.pack(pady=10)

#         self.load_btn = ttk.Button(
#             control_frame,
#             text="Import Traffic Video",
#             bootstyle="success",
#             command=self.load_video
#         )
#         self.load_btn.pack()

#         # Video Display Frame
#         display_frame = ttk.Frame(root)
#         display_frame.pack(expand=True, fill=BOTH)

#         self.input_label = ttk.Label(display_frame)
#         self.input_label.pack(side=LEFT, expand=True, padx=10)

#         self.output_label = ttk.Label(display_frame)
#         self.output_label.pack(side=RIGHT, expand=True, padx=10)

#     def load_video(self):
#         file_path = filedialog.askopenfilename(
#             filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
#         )

#         if file_path:
#             self.cap = cv2.VideoCapture(file_path)
#             self.update_frame()

#     def update_frame(self):
#         if self.cap is None:
#             return

#         ret, frame = self.cap.read()
#         if not ret:
#             self.cap.release()
#             return

#         processed = self.processor.process_frame(frame)

#         # Convert BGR to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

#         # Convert to ImageTk
#         img1 = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
#         img2 = ImageTk.PhotoImage(Image.fromarray(processed_rgb))

#         self.input_label.imgtk = img1
#         self.input_label.configure(image=img1)

#         self.output_label.imgtk = img2
#         self.output_label.configure(image=img2)

#         self.root.after(30, self.update_frame)


def launch_gui():
    root = ttk.Window(themename="darkly")  # Professional dark theme
    app = MotionApp(root)
    root.mainloop() 


# v3
class MotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Motion Estimation - Phase 2")
        self.root.geometry("1300x700")

        self.processor = OpticalFlowProcessor()
        self.cap = None
        self.playing = False

        # =========================
        # HEADER
        # =========================
        header = ttk.Label(
            root,
            text="Vehicle Motion Estimation using Optical Flow",
            font=("Helvetica", 22, "bold"),
            bootstyle="info"
        )
        header.pack(pady=20)

        # =========================
        # CONTROL FRAME
        # =========================
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=10)

        self.load_btn = ttk.Button(
            control_frame,
            text="Import Video",
            bootstyle="success",
            command=self.load_video
        )
        self.load_btn.grid(row=0, column=0, padx=10)

        self.play_btn = ttk.Button(
            control_frame,
            text="Play",
            bootstyle="primary",
            command=self.toggle_play
        )
        self.play_btn.grid(row=0, column=1, padx=10)

        self.stop_btn = ttk.Button(
            control_frame,
            text="Stop",
            bootstyle="danger",
            command=self.stop_video
        )
        self.stop_btn.grid(row=0, column=2, padx=10)

        # =========================
        # MOTION INFO LABEL
        # =========================
        self.motion_label = ttk.Label(
            root,
            text="Average Motion Intensity: 0.00",
            font=("Helvetica", 14),
            bootstyle="warning"
        )
        self.motion_label.pack(pady=10)

        # =========================
        # VIDEO DISPLAY FRAME
        # =========================
        display_frame = ttk.Frame(root)
        display_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)

        self.input_label = ttk.Label(display_frame)
        self.input_label.pack(side=LEFT, expand=True, padx=15)

        self.output_label = ttk.Label(display_frame)
        self.output_label.pack(side=RIGHT, expand=True, padx=15)

    # =========================
    # VIDEO FUNCTIONS
    # =========================

    def load_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )

        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.processor.prev_gray = None
            self.playing = False
            self.play_btn.config(text="Play")

    def toggle_play(self):
        if self.cap is None:
            return

        self.playing = not self.playing

        if self.playing:
            self.play_btn.config(text="Pause")
            self.update_frame()
        else:
            self.play_btn.config(text="Play")

    def stop_video(self):
        self.playing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.play_btn.config(text="Play")

    def update_frame(self):
        if not self.playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        processed = self.processor.process_frame(frame)

        # Calculate motion intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.processor.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.processor.prev_gray,
                gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_motion = np.mean(magnitude)
            self.motion_label.config(
                text=f"Average Motion Intensity: {avg_motion:.4f}"
            )

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        img1 = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        img2 = ImageTk.PhotoImage(Image.fromarray(processed_rgb))

        self.input_label.imgtk = img1
        self.input_label.configure(image=img1)

        self.output_label.imgtk = img2
        self.output_label.configure(image=img2)

        self.root.after(30, self.update_frame)