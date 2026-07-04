@dataclass(forzen=True)
class PiCameraConfig:
    """configurations for pi camera"""
    def __init__(self, width: int, height: int, pixel_format: str = "RGB888"):
        self.width : int = width
        self.height: int = height
        self.pixel_format: str = pixel_format

class PiCamera:
    def __init__(self, width: int, height: int, fps: int, pixel_format: str = "RGB888"):
        self.width : int = width
        self.height: int = height
        self.fps: int = fps
        self.pixel_format: str = pixel_format

        self.started = False
        self.running = False

        self.start()

    def start(self):
        from picamera2 import Picamera2

        self.picam = Picamera2()
        
        frame_duration_us = int(1_000_000 / self.config.fps)

        camera_config = self.picam2.create_video_configuration(
            main={
                "size": (self.config.width, self.config.height),
                "format": self.config.pixel_format,
            },
            controls={
                "FrameDurationLimits": (
                    frame_duration_us,
                    frame_duration_us,
                )
            },
        )

        self.picam2.configure(camera_config)
        self.picam2.start()

        self.started = True
        self.running = False

        print(
            "---------------------------\n"
            "[PiCamera] started\n"
            f"{self.config.width}x{self.config.height} \n"
            f"{self.config.fps}fps\n"
            f"{self.config.pixel_format}\n"
            "---------------------------\n"
        )