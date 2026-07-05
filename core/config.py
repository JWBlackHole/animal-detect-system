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
    dtype: str


@dataclass(frozen=True)
class SharedMemoryConfig:
    name: str
    buffer_size: int


@dataclass(frozen=True)
class RuntimeConfig:
    log_every_n_frames: int = 30


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    shared_memory: SharedMemoryConfig
    runtime: RuntimeConfig


def get_channels(pixel_format: str) -> int:
    if pixel_format in ("RGB888", "BGR888"):
        return 3

    if pixel_format in ("RGBA8888", "BGRA8888", "XRGB8888", "XBGR8888"):
        return 4

    raise ValueError(f"Unsupported pixel_format: {pixel_format}")


def get_frame_shape(camera_config: CameraConfig) -> tuple[int, int, int]:
    channels = get_channels(camera_config.pixel_format)
    return (
        camera_config.height,
        camera_config.width,
        channels,
    )

def get_frame_dtype(camera_config: CameraConfig) -> np.dtype:
    return np.dtype(camera_config.dtype)

def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    camera_raw = raw["camera"]
    shm_raw = raw["shared_memory"]
    runtime_raw = raw.get("runtime", {})

    camera = CameraConfig(
        width=int(camera_raw["width"]),
        height=int(camera_raw["height"]),
        fps=int(camera_raw["fps"]),
        pixel_format=str(camera_raw["pixel_format"]),
        dtype=str(camera_raw.get("dtype", "uint8")),
    )

    shared_memory = SharedMemoryConfig(
        name=str(shm_raw["name"]),
        buffer_size=int(shm_raw["buffer_size"]),
    )

    runtime = RuntimeConfig(
        log_every_n_frames=int(runtime_raw.get("log_every_n_frames", 30)),
    )

    if shared_memory.buffer_size <= 0:
        raise ValueError("shared_memory.buffer_size must be positive.")

    frame_shape = get_frame_shape(camera)
    frame_dtype = get_frame_dtype(camera)

    if frame_dtype != np.dtype(np.uint8):
        raise ValueError("Currently only uint8 frame dtype is supported.")

    print("---------------------------")
    print("[config] loaded")
    print(f"camera      : {camera.width}x{camera.height} {camera.fps}fps")
    print(f"pixel format: {camera.pixel_format}")
    print(f"frame shape : {frame_shape}")
    print(f"dtype       : {frame_dtype}")
    print(f"buffer name : {shared_memory.name}")
    print(f"buffer size : {shared_memory.buffer_size}")
    print("---------------------------")

    return AppConfig(
        camera=camera,
        shared_memory=shared_memory,
        runtime=runtime,
    )