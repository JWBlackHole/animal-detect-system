
import json
import socket
from pathlib import Path
from typing import Any, Mapping, Optional


DetectionResult = dict[str, Any]
DetectionBox = dict[str, Any]


REQUIRED_DETECTION_RESULT_KEYS = (
    "frame_id",
    "timestamp",
    "boxes",
    "inference_ms",
)

REQUIRED_BOX_KEYS = (
    "label",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


def normalize_detection_box(box: Mapping[str, Any]) -> DetectionBox:
    """
    Validate and normalize one detection box.

    Required keys:
        label: str
        confidence: float
        x1, y1, x2, y2: int
    """

    for key in REQUIRED_BOX_KEYS:
        if key not in box:
            raise ValueError(f"Missing detection box key: {key}")

    normalized: DetectionBox = {
        "label": str(box["label"]),
        "confidence": float(box["confidence"]),
        "x1": int(box["x1"]),
        "y1": int(box["y1"]),
        "x2": int(box["x2"]),
        "y2": int(box["y2"]),
    }

    if not 0.0 <= normalized["confidence"] <= 1.0:
        raise ValueError(
            f"box.confidence must be between 0.0 and 1.0, "
            f"got {normalized['confidence']}"
        )

    return normalized


def normalize_detection_result(result: Mapping[str, Any]) -> DetectionResult:
    """
    Validate and normalize DetectionResult.
    """

    for key in REQUIRED_DETECTION_RESULT_KEYS:
        if key not in result:
            raise ValueError(f"Missing detection result key: {key}")

    boxes_raw = result["boxes"]

    if not isinstance(boxes_raw, list):
        raise ValueError("detection result 'boxes' must be a list.")

    normalized = dict(result)

    normalized["frame_id"] = int(result["frame_id"])
    normalized["timestamp"] = int(result["timestamp"])
    normalized["inference_ms"] = float(result["inference_ms"])
    normalized["boxes"] = [
        normalize_detection_box(box)
        for box in boxes_raw
    ]

    return normalized


def encode_detection_result(result: Mapping[str, Any]) -> bytes:
    normalized = normalize_detection_result(result)

    return json.dumps(
        normalized,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_detection_result(payload: bytes) -> DetectionResult:
    raw = json.loads(payload.decode("utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Detection result payload must decode to a dict.")

    return normalize_detection_result(raw)


class DetectionResultSender:
    """
    Unix domain datagram sender for detection results.
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        strict: bool = False,
    ) -> None:
        self.socket_path = str(socket_path)
        self.strict = strict
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def send(self, result: Mapping[str, Any]) -> bool:
        payload = encode_detection_result(result)

        try:
            self.sock.sendto(payload, self.socket_path)
            return True

        except (FileNotFoundError, ConnectionRefusedError, OSError):
            if self.strict:
                raise

            return False

    def close(self) -> None:
        self.sock.close()


class DetectionResultReceiver:
    """
    Unix domain datagram receiver for detection results.
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout: Optional[float] = 0.1,
        max_packet_size: int = 65536,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self.max_packet_size = max_packet_size

        if self.socket_path.exists():
            self.socket_path.unlink()

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(str(self.socket_path))

        if timeout is not None:
            self.sock.settimeout(timeout)

    def recv(self) -> Optional[DetectionResult]:
        try:
            payload = self.sock.recv(self.max_packet_size)
        except socket.timeout:
            return None

        return decode_detection_result(payload)

    def close(self) -> None:
        self.sock.close()

        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass