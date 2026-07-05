
import json
import socket
from pathlib import Path
from typing import Any, Mapping, Optional

FrameMetadata = dict[str, Any]

REQUIRED_FRAME_METADATA_KEYS = (
    "frame_id",
    "timestamp",
    "slot",
)

def normalize_frame_metadata(metadata: Mapping[str, Any]) -> FrameMetadata:
    """
    Validate and normalize frame metadata.

    Required keys:
        frame_id: int
        timestamp: int
        slot: int

    Extra keys are allowed and preserved.
    """

    for key in REQUIRED_FRAME_METADATA_KEYS:
        if key not in metadata:
            raise ValueError(f"Missing frame metadata key: {key}")

    normalized = dict(metadata)

    normalized["frame_id"] = int(metadata["frame_id"])
    normalized["timestamp"] = int(metadata["timestamp"])
    normalized["slot"] = int(metadata["slot"])

    return normalized


def encode_frame_metadata(metadata: Mapping[str, Any]) -> bytes:
    normalized = normalize_frame_metadata(metadata)

    return json.dumps(
        normalized,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_frame_metadata(payload: bytes) -> FrameMetadata:
    raw = json.loads(payload.decode("utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Frame metadata payload must decode to a dict.")

    return normalize_frame_metadata(raw)


class FrameMetadataSender:
    """
    Unix domain datagram sender for frame metadata.

    This sender does not require a persistent connection.

    If strict=False:
        send() returns False when receiver socket does not exist.

    If strict=True:
        send() raises the socket error.
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

    def send(self, metadata: Mapping[str, Any]) -> bool:
        payload = encode_frame_metadata(metadata)

        try:
            self.sock.sendto(payload, self.socket_path)
            return True

        except (FileNotFoundError, ConnectionRefusedError, OSError):
            if self.strict:
                raise

            return False

    def close(self) -> None:
        self.sock.close()


class FrameMetadataReceiver:
    """
    Unix domain datagram receiver for frame metadata.

    Example:
        receiver = FrameMetadataReceiver("/tmp/animal_detection_frame_meta.sock")

        while True:
            metadata = receiver.recv()
            if metadata is None:
                continue

            frame_id = metadata["frame_id"]
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout: Optional[float] = 0.5,
        max_packet_size: int = 8192,
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

    def recv(self) -> Optional[FrameMetadata]:
        try:
            payload = self.sock.recv(self.max_packet_size)
        except socket.timeout:
            return None

        return decode_frame_metadata(payload)

    def close(self) -> None:
        self.sock.close()

        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass