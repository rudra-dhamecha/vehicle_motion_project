import cv2
import numpy as np

# COCO vehicle class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


class CarDetector:
    """Detects vehicles (cars, trucks, buses, motorcycles) using YOLOv8."""

    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.4):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """Returns list of (bbox_xyxy, confidence, class_id, class_name)."""
        results = self.model(frame, verbose=False)[0]
        detections = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = results.names.get(cls_id, "vehicle")
            detections.append(((x1, y1, x2, y2), conf, cls_id, name))
        return detections


def draw_detections(frame, detections, flow_magnitude=None, flow_angle=None):
    """
    Draw bounding boxes and labels on frame.
    If flow_magnitude/flow_angle are provided (same shape as frame), compute
    per-box average flow magnitude as a speed proxy.
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2), conf, cls_id, name in detections:
        # Clamp to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # Per-box speed proxy from optical flow
        speed_str = ""
        if flow_magnitude is not None and flow_angle is not None:
            roi_mag = flow_magnitude[y1:y2, x1:x2]
            if roi_mag.size > 0:
                avg_mag = np.mean(roi_mag)
                speed_str = f" {avg_mag:.2f}"

        label = f"{name} {conf:.1%}{speed_str}"
        color = (0, 200, 100)  # BGR green
        thickness = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    return out


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
        self._last_magnitude = None
        self._last_angle = None

    def process_frame(self, frame):
        """Returns (optical_flow_vis_bgr, magnitude, angle). magnitude/angle are None on first frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self._last_magnitude = None
            self._last_angle = None
            return frame, None, None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self._last_magnitude = magnitude
        self._last_angle = angle

        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.prev_gray = gray
        return output, magnitude, angle
