import copy
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

class MonitorStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {
            "camera": {},
            "detection": {},
            "video": {},
        }

    def update(self, message: dict[str, Any]) -> None:
        source = message.get("source")

        if source not in self._data:
            return

        data = {
            **message.get("stats", {}),
            "interval_s"   : message["interval_s"],
            "cpu_percent"  : message["cpu_percent"],
            "updated_at_ts": datetime.now().timestamp(),
        }

        with self._lock:
            self._data[source] = data

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data)

    def response(self) -> dict[str, Any]:
        raw = self.snapshot()

        return raw

        camera = raw["camera"]
        detection = raw["detection"]
        video = raw["video"]

        detection_count = detection.get("detection_count", 0)
        detection_output = detection.get("output", 0)
        detection_interval_s = detection.get("interval_s", 0.0)

        video_interval_s = video.get("interval_s", 0.0)

        return {
            "summary": {
                "camera_fps": camera.get("camera_fps", 0.0),
                "camera_target_fps": camera.get(
                    "camera_target_fps",
                    0.0,
                ),
                "video_fps": (
                    video.get("output", 0) / video_interval_s
                    if video_interval_s > 0
                    else 0.0
                ),
                "avg_inference_ms": (
                    detection.get("total_inference_ms", 0.0)
                    / detection_count
                    if detection_count > 0
                    else 0.0
                ),
                "avg_detection_interval_s": (
                    detection_interval_s / detection_output
                    if detection_output > 0
                    else 0.0
                ),
                "buffer_pending": video.get("pending", 0),
            },
            "camera": camera,
            "detection": detection,
            "video": video,
        }