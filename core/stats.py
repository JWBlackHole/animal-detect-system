from dataclasses import dataclass, asdict
from typing import Optional
import psutil

def _build_cpu_meter():
    try:
        import psutil
    except ImportError:
        return None

    process = psutil.Process()
    process.cpu_percent(interval=None)
    return process

@dataclass
class CameraStats:
    """for camera statistic"""
    capture_count: int = 0

    last_frame_id       : int = -1
    max_frame_interval_ns: int = 0.0

    def clear(self):
        self.capture_count = 0

        self.last_frame_id = -1
        self.max_frame_interval_ns = 0.0

    def to_stats(self, interval_s: float, cpu_percent: float):
        data = {
            "source"  : "camera",
            "stats"    : asdict(self),
            "interval_s": interval_s,
            "cpu_percent": cpu_percent
        }

        return data

@dataclass
class DetectionStats:
    """for statistic log"""
    # for frame socket receiving meta
    meta: int = 0

    # detection results
    detection_count   : int = 0
    total_inference_ms: float = 0.0
    max_inference_ms  : float = 0.0

    # for result output
    output   : int = 0

    def clear(self):
        self.meta = 0

        self.detection_count = 0
        self.total_inference_ms = 0.0
        self.max_inference_ms = 0.0

        self.output = 0

    def summarize(self, interval_s: float):
        print(
            f"---------------------\n"
            f"[detection]\n"
            f"avg. output interval: {0.0 if abs(self.output) < 0.001 else interval_s / self.output:.2f} seconds\n"
            f"avg. inference time : {0.0 if self.detection_count == 0 else self.total_inference_ms / self.detection_count:.3f}\n"
            f"max. inference time : {self.max_inference_ms:.3f}\n"
            f"meta: {self.meta}\n"
            f"results: {self.detection_count}\n"
            f"output : {self.output}\n"
            "---------------------\n"
        )

    def to_stats(self, interval_s: float, cpu_percent: float):
        data = {
            "source"  : "detection",
            "stats"    : asdict(self),
            "interval_s": interval_s,
            "cpu_percent": cpu_percent
        }

        return data

@dataclass
class VideoStats:
    """for statistic log"""
    last_frame_id        : int = -1
    max_frame_interval_ns: Optional[int] = None

    # for frame socket receiving meta
    meta      : int = 0
    fetched   : int = 0
    fetch_miss: int = 0

    # for result socket recieing detection result
    results: int = 0
    matched: int = 0
    late   : int = 0

    # for video output
    output   : int = 0
    pending  : int = 0
    dropped  : int = 0
    underflow: int = 0

    def clear(self):
        self.last_frame_id = -1
        self.max_frame_interval_ns = None

        self.meta = 0
        self.fetched = 0
        self.fetch_miss = 0

        self.results = 0
        self.matched = 0
        self.late = 0

        self.output = 0
        self.pending = 0
        self.dropped = 0
        self.underflow = 0

    def summarize(self, interval_s: float):
        print(
            f"---------------------\n"
            f"[video]\n"
            f"avg. fps: {0.0 if abs(interval_s) < 0.001 else self.output / interval_s:.2f}\n"
            f"meta: {self.meta}, fetched: {self.fetched}, fetch_miss: {self.fetch_miss}\n"
            f"results: {self.results}, matched: {self.matched}, late: {self.late}\n"
            f"output : {self.output}, dropped: {self.dropped}, underflow: {self.underflow}\n"
            f"buffer pending: {self.pending}\n"
            "---------------------\n"
        )

    def to_stats(self, interval_s: float, cpu_percent: float):
        data = {
            "source"  : "video",
            "stats"    : asdict(self),
            "interval_s": interval_s,
            "cpu_percent": cpu_percent
        }

        return data