import cv2
import numpy as np


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    print("Starting Optical Flow Analysis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farneback Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        avg_motion = np.mean(magnitude)
        print(f"Average Motion Intensity: {avg_motion:.4f}")

        # Visualization
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow("Vehicle Motion - Optical Flow", rgb)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()
    print("Analysis Finished.")


# v2: frame by frame
class OpticalFlowProcessor:
    def __init__(self):
        self.prev_gray = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame  # First frame, no motion yet

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.prev_gray = gray
        return output
