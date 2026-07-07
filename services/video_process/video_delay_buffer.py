from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PendingFrame:
    frame_id: int
    timestamp: int
    image: np.ndarray


class VideoDelayBuffer:
    """Video delay buffer"""

    def __init__(self, buffer_size: int) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")

        self.buffer_size = buffer_size
        self._frames: list[Optional[PendingFrame]] = [None] * buffer_size

        self._start = 0
        self._end = 0
        self._count = 0

        self._frame_id_to_slot: dict[int, int] = {}

        self.dropped_frames = 0

    @property
    def used_n_slot(self) -> int:
        return self._count

    @property
    def left_n_slot(self) -> int:
        return self.buffer_size - self._count

    @property
    def full(self) -> bool:
        return self._count == self.buffer_size

    @property
    def empty(self) -> bool:
        return self._count == 0

    def append_frame(
        self,
        frame_id: int,
        timestamp: int,
        image: np.ndarray,
    ) -> Optional[PendingFrame]:
        """Append one frame """

        dropped = None

        if self.full:
            dropped = self.pop_frame()
            self.dropped_frames += 1

        slot = self._end

        frame = PendingFrame(
            frame_id=int(frame_id),
            timestamp=int(timestamp),
            image=image,
        )

        self._frames[slot] = frame
        self._frame_id_to_slot[frame.frame_id] = slot

        self._end = self._next_slot(self._end)
        self._count += 1

        return dropped

    def pop_frame(self) -> Optional[PendingFrame]:
        """Pop the oldest frame"""

        if self.empty:
            return None

        frame = self._frames[self._start]

        if frame is None:
            raise RuntimeError(f"Buffer corrupted: slot {self._start} is empty.")

        self._frames[self._start] = None
        self._frame_id_to_slot.pop(frame.frame_id, None)

        self._start = self._next_slot(self._start)
        self._count -= 1

        return frame

    def draw_boxes_on_frame(
        self,
        frame_id: int,
        boxes: list[dict],
    ) -> bool:
        """Draw boxes directly onto the buffered frame image."""

        slot = self._frame_id_to_slot.get(int(frame_id))

        if slot is None:
            return False

        frame = self._frames[slot]

        if frame is None:
            self._frame_id_to_slot.pop(int(frame_id), None)
            return False

        if frame.frame_id != int(frame_id):
            self._frame_id_to_slot.pop(int(frame_id), None)
            return False

        for box in boxes:
            self._draw_box(frame.image, box)

        return True

    def _next_slot(self, slot: int) -> int:
        return (slot + 1) % self.buffer_size

    @staticmethod
    def _draw_box(image: np.ndarray, box: dict) -> None:
        height, width = image.shape[:2]

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