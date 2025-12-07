"""
YOLO + MediaPipe FaceMesh detector.
For each detected person (YOLO bbox), runs FaceMesh on the cropped region
and returns:
 - bboxes: List[(x, y, w, h)]
 - face_contours: List[List[(x, y)]] per person (absolute frame coords)
"""

from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from .yolo_person_detector import YoloPersonDetector
from .config import Config
import mediapipe as mp


class YoloHolisticDetector:
    def __init__(self, yolo_model: str = "yolov8n.pt", conf: float = 0.35, iou: float = 0.45):
        self.config = Config()
        self.yolo = YoloPersonDetector(model_name=yolo_model, conf=conf, iou=iou)
        # Holistic for pose + hands + face in one pass
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.config.MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    def detect(self, frame) -> Dict[str, Any]:
        if frame is None:
            return {"bboxes": [], "face_contours": []}

        # Debug: log YOLO detection
        import time
        yolo_start = time.time()
        bboxes = self.yolo.detect(frame)
        yolo_time = time.time() - yolo_start
        
        if yolo_time > 0.05:  # More than 50ms
            print(f"[Debug] Slow YOLO detection: {yolo_time:.3f}s, found {len(bboxes)} people")
        
        face_contours: List[List[Tuple[int, int]]] = []
        pose_lines: List[List[Tuple[int, int, int, int]]] = []
        hand_lines: List[List[Tuple[int, int, int, int]]] = []

        if not bboxes:
            return {"bboxes": [], "face_contours": [], "pose_lines": [], "hand_lines": []}

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x, y, bw, bh) in bboxes:
            # Expand bbox to better include shoulders/hands
            margin = 0.2
            exp_w = int(bw * (1.0 + margin))
            exp_h = int(bh * (1.0 + margin))
            cx = x + bw // 2
            cy = y + bh // 2
            x0 = max(0, cx - exp_w // 2)
            y0 = max(0, cy - exp_h // 2)
            x1 = min(w, cx + exp_w // 2)
            y1 = min(h, cy + exp_h // 2)
            if x1 - x0 < 10 or y1 - y0 < 10:
                face_contours.append([])
                continue

            crop_rgb = rgb_frame[y0:y1, x0:x1]
            # Run Holistic on the crop
            try:
                res = self.holistic.process(crop_rgb)
            except Exception:
                face_contours.append([])
                pose_lines.append([])
                hand_lines.append([])
                continue

            # Convert normalized to pixel coords in original frame
            crop_h, crop_w = crop_rgb.shape[:2]

            # Face contour from holistic face landmarks (if present)
            if res.face_landmarks and res.face_landmarks.landmark:
                pts = []
                for lm in res.face_landmarks.landmark:
                    px = int(lm.x * crop_w) + x0
                    py = int(lm.y * crop_h) + y0
                    px = max(0, min(px, w - 1))
                    py = max(0, min(py, h - 1))
                    pts.append([px, py])
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    hull = cv2.convexHull(pts_np)
                    contour = hull.reshape(-1, 2).tolist()
                    face_contours.append(contour)
                else:
                    face_contours.append([])
            else:
                face_contours.append([])

            # Pose lines from POSE_CONNECTIONS
            lines_pose: List[Tuple[int, int, int, int]] = []
            if res.pose_landmarks and res.pose_landmarks.landmark:
                # Map visible points
                pts_pose = {}
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    # Use visibility if provided, otherwise draw anyway
                    px = int(lm.x * crop_w) + x0
                    py = int(lm.y * crop_h) + y0
                    if 0 <= px < w and 0 <= py < h:
                        pts_pose[i] = (px, py)
                for (i1, i2) in self.mp_pose.POSE_CONNECTIONS:
                    a = int(i1.value) if hasattr(i1, 'value') else int(i1)
                    b = int(i2.value) if hasattr(i2, 'value') else int(i2)
                    if a in pts_pose and b in pts_pose:
                        x1, y1 = pts_pose[a]
                        x2, y2 = pts_pose[b]
                        lines_pose.append((x1, y1, x2, y2))
                # Debug key joints
                try:
                    LS = int(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                    RS = int(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                    LW = int(self.mp_pose.PoseLandmark.LEFT_WRIST.value)
                    RW = int(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
                    debug_pts = {
                        'L_SHOULDER': pts_pose.get(LS),
                        'R_SHOULDER': pts_pose.get(RS),
                        'L_WRIST': pts_pose.get(LW),
                        'R_WRIST': pts_pose.get(RW),
                    }
                    print(f"[Holistic Debug] Pose keypoints: {debug_pts}")
                except Exception:
                    pass
            pose_lines.append(lines_pose)

            # Hand lines from HAND_CONNECTIONS for both hands
            lines_hands: List[Tuple[int, int, int, int]] = []
            for hand_landmarks in [res.left_hand_landmarks, res.right_hand_landmarks]:
                if hand_landmarks and hand_landmarks.landmark:
                    pts_hand = {}
                    for i, lm in enumerate(hand_landmarks.landmark):
                        px = int(lm.x * crop_w) + x0
                        py = int(lm.y * crop_h) + y0
                        if 0 <= px < w and 0 <= py < h:
                            pts_hand[i] = (px, py)
                    for (i1, i2) in self.mp_hands.HAND_CONNECTIONS:
                        a = int(i1.value) if hasattr(i1, 'value') else int(i1)
                        b = int(i2.value) if hasattr(i2, 'value') else int(i2)
                        if a in pts_hand and b in pts_hand:
                            x1, y1 = pts_hand[a]
                            x2, y2 = pts_hand[b]
                            lines_hands.append((x1, y1, x2, y2))
            print(f"[Holistic Debug] pose_lines={len(lines_pose)} hand_lines={len(lines_hands)}")
            hand_lines.append(lines_hands)

        return {
            "bboxes": bboxes,
            "face_contours": face_contours,
            "pose_lines": pose_lines,
            "hand_lines": hand_lines,
        }


