from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    pixel_format: str
    dtype: np.dtype

    log_every_n_seconds: int

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (
            self.height,
            self.width,
            self.channel,
        )
    
    @property
    def channel(self) -> int:
        if self.pixel_format in ("RGB888", "BGR888"):
            return 3

        if self.pixel_format in ("RGBA8888", "BGRA8888", "XRGB8888", "XBGR8888"):
            return 4

        raise ValueError(f"Unsupported pixel_format: {self.pixel_format}")
    
    @property
    def frame_bytes(self) -> int:
        return int(np.prod(self.frame_shape) * self.dtype.itemsize)


@dataclass(frozen=True)
class SharedMemoryConfig:
    name: str
    buffer_size: int

@dataclass(frozen=True)
class DetectionConfig:
    enabled: bool
    detect_every_n_frames: int
    model_path: str
    image_size: int
    confidence: float

    cpu_affinity: list[int] | None
    torch_num_threads: int
    torch_num_interop_threads: int
    process_nice: int

    log_every_n_seconds: int

@dataclass(frozen=True)
class VideoConfig:
    buffer_size: int
    startup_delay_ms: int
    log_every_n_seconds: int

@dataclass(frozen=True)
class IPCConfig:
    video_frame_meta_socket: str
    detection_frame_meta_socket: str
    detection_result_socket: str

@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    shared_memory: SharedMemoryConfig
    detection: DetectionConfig
    video: VideoConfig
    ipc: IPCConfig
    
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    camera_raw = raw["camera"]
    shm_raw = raw["shared_memory"]
    detection_raw = raw["detection"]
    video_raw = raw["video"]
    ipc_raw = raw["ipc"]
    runtime_raw = raw.get("runtime", {})

    # load camera config
    camera = CameraConfig(
        width=int(camera_raw["width"]),
        height=int(camera_raw["height"]),
        fps=int(camera_raw["fps"]),
        pixel_format=str(camera_raw["pixel_format"]),
        dtype=np.dtype(camera_raw["dtype"]),
        log_every_n_seconds=int(camera_raw["log_every_n_seconds"]),
    )

    # load shared memory config
    shared_memory = SharedMemoryConfig(
        name=str(shm_raw["name"]),
        buffer_size=int(shm_raw["buffer_size"]),
    )
    if shared_memory.buffer_size <= 0:
        raise ValueError("shared_memory.buffer_size must be positive.")
    
    # load detection config
    detection = DetectionConfig(
        enabled               = bool(detection_raw["enabled"]),
        detect_every_n_frames = int(detection_raw["detect_every_n_frames"]),
        model_path            = str(detection_raw["model_path"]),
        image_size            = int(detection_raw["image_size"]),
        confidence            = float(detection_raw["confidence"]),

        cpu_affinity          = list(detection_raw["cpu_affinity"]),
        torch_num_threads     = int(detection_raw["torch_num_threads"]),
        torch_num_interop_threads = int(detection_raw["torch_num_interop_threads"]),
        process_nice              = int(detection_raw["process_nice"]),

        log_every_n_seconds=int(detection_raw["log_every_n_seconds"])
    )
    if detection.detect_every_n_frames <= 0:
        raise ValueError("detection.detect_every_n_frames must be positive.")
    if detection.image_size <= 0:
        raise ValueError("detection.image_size must be positive.")
    if not 0.0 <= detection.confidence <= 1.0:
        raise ValueError("detection.confidence must be between 0.0 and 1.0.")
    
    # load video config
    video = VideoConfig(
        buffer_size=int(video_raw["buffer_size"]),
        startup_delay_ms=int(video_raw["startup_delay_ms"]),
        log_every_n_seconds=int(video_raw["log_every_n_seconds"]),
    )
        
    # load IPC config
    ipc = IPCConfig(
        video_frame_meta_socket=str(ipc_raw["video_frame_meta_socket"]),
        detection_frame_meta_socket=str(ipc_raw["detection_frame_meta_socket"]),
        detection_result_socket=str(ipc_raw["detection_result_socket"])
    )

    if camera.dtype != np.dtype(np.uint8):
        raise ValueError("Currently only uint8 frame dtype is supported.")

    print("---------------------------")
    print("[config] loaded")
    print(f"camera      : {camera.width}x{camera.height} {camera.fps}fps")
    print(f"pixel format: {camera.pixel_format}")
    print(f"frame shape : {camera.frame_shape}")
    print(f"dtype       : {camera.dtype}")
    print(f"buffer name : {shared_memory.name}")
    print(f"buffer size : {shared_memory.buffer_size}")
    print(f"detection enable: {detection_raw["enabled"]}")
    print(f"video meta socket    : {ipc.video_frame_meta_socket}")
    print(f"detection meta socket: {ipc.detection_frame_meta_socket}")
    print("---------------------------")

    return AppConfig(
        camera=camera,
        shared_memory=shared_memory,
        detection=detection,
        video=video,
        ipc=ipc,
    )