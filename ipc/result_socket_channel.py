
import json
import socket
from pathlib import Path
from typing import Any, Optional

DetectionResult = dict[str, Any]
DetectionBox = dict[str, Any]

def encode_detection_result(result: dict[str, Any]) -> bytes:

    return json.dumps(
        result,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_detection_result(payload: bytes) -> DetectionResult:
    raw = json.loads(payload.decode("utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Detection result payload must decode to a dict.")
    return raw


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

    def send(self, result: dict[str, Any]) -> bool:
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