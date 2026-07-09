# services/camera_process/pi_camera.py

from typing import Optional

from picamera2 import Picamera2

class PiCamera:
    def __init__(self, width: int, height: int, fps: int, pixel_format: str = "RGB888") -> None:
        # hardware info
        self.width : int = width
        self.height: int = height
        self.fps: int = fps
        self.pixel_format: str = pixel_format

        self.picam: Optional[Picamera2] = None

        # frame information
        self._frame_id: int = 0
        self.last_ts: Optional[int] = None

        # camera States
        self.started: bool = False

    def start(self) -> None:

        self.picam = Picamera2()

        frame_duration_us = int(1_000_000 / self.fps)

        camera_config = self.picam.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": self.pixel_format,
            },
            controls={
                "FrameDurationLimits": (
                    frame_duration_us, # min duration
                    frame_duration_us, # max duration
                )
            },
        )

        self.picam.configure(camera_config)
        self.picam.start()

        self.started = True

        print(
            "---------------------------\n"
            "[PiCamera] started\n"
            # f"{self.width}x{self.height} \n"
            # f"{self.fps}fps\n"
            # f"{self.pixel_format}\n"
            "---------------------------\n"
        )
    
    def stop(self) -> None:
        """stop the pi camera"""

        if not self.started:
            return

        if self.picam is not None:
            try:
                self.picam.stop()
            finally:
                self.picam.close()

        self.picam = None
        self.started = False

    def capture_once(self) -> dict:
        if not self.started or self.picam is None:
            raise RuntimeError("PiCamera must be started before capture_once().")

        with self.picam.captured_request() as request:
            image    = request.make_array("main")
            metadata = request.get_metadata()

        ts = metadata["SensorTimestamp"]
        frame: dict = {
            "frame_id"      : self._frame_id,
            "timestamp"     : ts,
            "width"         : self.width,
            "height"        : self.height,
            "pixel_format"  : self.pixel_format,
            "image"         : image
        }

        if(self.last_ts is not None):
            frame_gap = (ts - self.last_ts) / 1_000_000
            # print(frame_gap)
            # if(frame_gap > 50.0):
            #     print(f"[camera] picam frame gap > 50: {frame_gap}")

        self._frame_id += 1
        self.last_ts = ts

        return frame