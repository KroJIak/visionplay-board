"""
YOLO-based person detector using ultralytics.
Returns a list of bounding boxes (x, y, w, h) for detected persons (class id 0).
"""

from typing import List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # Will raise at runtime if used without install


class YoloPersonDetector:
    """Ultralytics YOLO person detector (CPU by default)."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35, iou: float = 0.45):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Please install with: pip install ultralytics")
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        if frame is None:
            return []

        # YOLO expects BGR or RGB; we pass BGR ndarray
        # Use smaller image size for better performance
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=640,  # Fixed size for better performance
            verbose=False,
            device="cpu",
        )

        bboxes: List[Tuple[int, int, int, int]] = []
        max_people = 2  # Limit to maximum 2 people for performance
        
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                if len(bboxes) >= max_people:  # Stop after 2 people
                    break
                    
                cls_id = int(b.cls.item()) if hasattr(b.cls, 'item') else int(b.cls)
                # COCO person class id is 0
                if cls_id != 0:
                    continue
                xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy, 'cpu') else np.array(b.xyxy[0])
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                if w > 0 and h > 0:
                    bboxes.append((x1, y1, w, h))
            
            if len(bboxes) >= max_people:  # Stop processing if we have 2 people
                break

        return bboxes


# Backward-compatible alias
PersonDetector = YoloPersonDetector


