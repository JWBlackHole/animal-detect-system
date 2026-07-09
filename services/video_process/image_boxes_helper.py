import cv2
import numpy as np
from typing import Optional

def get_boxes(frame_id: int, labels: dict[int,dict], detect_every_n_frame: int) -> Optional[dict]:
    label_frame_id = frame_id - (frame_id % detect_every_n_frame)

    labels_to_be_popped: list[int] = []
    for key in labels:
        if key < label_frame_id:
            labels_to_be_popped.append(key)

    for key in labels_to_be_popped:
       labels.pop(key, None) 

    if label_frame_id in labels:
        return labels[label_frame_id]
    else:
        return None

def draw_boxes(image: np.ndarray, boxes: dict) -> None:
    if not boxes:
        return
    
    height, width = image.shape[:2]

    for box in boxes:
        label = str(box["label"])
        confidence = float(box.get("confidence", 0.0))

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return

        text = f"{label} {confidence:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            text,
            (x1, max(y1 - 8, 0) if y1 > 8 else y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )