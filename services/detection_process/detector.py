from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

class YoloDetector:
    """
    YOLO detector wrapper.

    1. load model
    2. run inference
    3. convert YOLO output into simple boxes
    """

    def __init__(
        self,
        model_path: str,
        image_size: int,
        confidence: float,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = model_path
        self.image_size = image_size
        self.confidence = confidence

        print(f"[detector] loading model: {model_path}")

        self.model = YOLO(model_path)

        print("[detector] model loaded")
        print(f"[detector] image_size           = {image_size}")
        print(f"[detector] confidence threshold = {confidence}")

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run YOLO inference"""

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )

        if len(results) == 0:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        names: dict[int, str] = result.names
        boxes: list[dict] = []

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            x1, y1, x2, y2 = xyxy

            boxes.append(
                {
                    "label": names.get(cls_id, str(cls_id)),
                    "confidence": float(conf),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            )

        return boxes