from __future__ import annotations

import os
import subprocess
from typing import Optional

import numpy as np


class FFmpegRTSPPublisher:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        pixel_format: str,
        rtsp_url: str,
        codec: str = "h264_v4l2m2m",
        bitrate: str = "1000k",
        gop_seconds: int = 1,
        stop_timeout: float = 8.0,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.rtsp_url = rtsp_url
        self.codec = codec
        self.bitrate = bitrate
        self.gop_seconds = gop_seconds
        self.stop_timeout = stop_timeout

        self.process: Optional[subprocess.Popen] = None
        self._video_write_fd: Optional[int] = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        read_fd, write_fd = os.pipe()
        self._video_write_fd = write_fd

        gop = max(1, self.fps * self.gop_seconds)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",

            "-f", "rawvideo",
            "-pix_fmt", self._ffmpeg_input_pix_fmt(),
            "-s", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", f"pipe:{read_fd}",

            "-an",
            "-vf", "format=yuv420p",

            *self._codec_args(gop),

            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]

        print("[ffmpeg]", " ".join(cmd))

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=None,
            pass_fds=(read_fd,),
            bufsize=0,
        )

        os.close(read_fd)

    def write(self, image: np.ndarray) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False

        if self._video_write_fd is None:
            return False

        if image.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected frame shape {(self.height, self.width, 3)}, got {image.shape}"
            )

        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {image.dtype}")

        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        try:
            data = memoryview(image).cast("B")
            total = 0

            while total < len(data):
                written = os.write(self._video_write_fd, data[total:])
                if written == 0:
                    return False
                total += written

            return True

        except OSError as exc:
            print(f"[ffmpeg] write failed: {exc}")
            return False

    def stop(self) -> bool:
        if self.process is None:
            self._close_video_pipe()
            return True

        proc = self.process
        self.process = None

        try:
            if proc.stdin is not None:
                proc.stdin.write(b"q\n")
                proc.stdin.flush()
                proc.stdin.close()
        except OSError:
            pass

        self._close_video_pipe()

        try:
            proc.wait(timeout=self.stop_timeout)
            print(f"[ffmpeg] stopped, code={proc.returncode}")
            return True

        except subprocess.TimeoutExpired:
            print(f"[ffmpeg] still alive, pid={proc.pid}; not killing it")
            return False

    def close(self) -> bool:
        return self.stop()

    def _close_video_pipe(self) -> None:
        if self._video_write_fd is not None:
            try:
                os.close(self._video_write_fd)
            except OSError:
                pass
            self._video_write_fd = None

    def _codec_args(self, gop: int) -> list[str]:
        if self.codec == "h264_v4l2m2m":
            return [
                "-c:v", "h264_v4l2m2m",
                "-b:v", self.bitrate,
                "-g", str(gop),
            ]

        if self.codec == "libx264":
            return [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-b:v", self.bitrate,
                "-maxrate", self.bitrate,
                "-bufsize", self.bitrate,
                "-g", str(gop),
                "-keyint_min", str(gop),
                "-sc_threshold", "0",
            ]

        raise ValueError(f"Unsupported codec: {self.codec}")

    def _ffmpeg_input_pix_fmt(self) -> str:
        if self.pixel_format == "RGB888":
            return "bgr24"
            return "rgb24"

        if self.pixel_format == "BGR888":
            return "bgr24"

        raise ValueError(f"Unsupported pixel_format: {self.pixel_format}")