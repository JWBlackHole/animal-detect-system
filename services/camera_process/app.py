from __future__ import annotations

from multiprocessing import Event
from multiprocessing.synchronize import Event as SyncEvent
from typing import Any

from core.config import (
    get_frame_dtype,
    get_frame_shape,
    load_config,
)
from ipc.shared_frame_buffer import SharedFrameBuffer
from services.camera_process.pi_camera import PiCamera


def camera_main(
    lock: Any,
    stop_event: SyncEvent,
) -> None:

    config = load_config()

    camera_config = config.camera
    shm_config = config.shared_memory
    runtime_config = config.runtime

    frame_shape = get_frame_shape(camera_config)
    frame_dtype = get_frame_dtype(camera_config)

    camera = PiCamera(
        width=camera_config.width,
        height=camera_config.height,
        fps=camera_config.fps,
        pixel_format=camera_config.pixel_format,
    )
    camera.start()

    frame_buffer = SharedFrameBuffer(
        name=shm_config.name,
        frame_shape=frame_shape,
        dtype=frame_dtype,
        buffer_size=shm_config.buffer_size,
        lock=lock,
        create=False,
    )

    # =========================
    # Socket placeholders
    # =========================
    #
    # import json
    # import socket
    #
    # streaming_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    # detection_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #
    # streaming_sock.connect("/tmp/animal_streaming.sock")
    # detection_sock.connect("/tmp/animal_detection.sock")
    #
    # def send_metadata(metadata: dict) -> None:
    #     data = json.dumps(metadata).encode("utf-8")
    #     streaming_sock.send(data)
    #     detection_sock.send(data)

    try:
        print("[camera] started")

        while not stop_event.is_set():
            frame = camera.capture_once()

            frame_id = int(frame["frame_id"])
            timestamp = int(frame["timestamp"])
            image = frame["image"]

            slot = frame_buffer.write_frame(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
            )

            metadata = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "slot": slot,
                "width": camera_config.width,
                "height": camera_config.height,
                "pixel_format": camera_config.pixel_format,
            }

            # TODO:
            # send metadata to streaming_process
            # send metadata to detection_process

            # logging info
            # if frame_id % runtime_config.log_every_n_frames == 0:
            #     print(
            #         f"[camera] frame_id={metadata['frame_id']}, "
            #         f"slot={metadata['slot']}, "
            #         f"timestamp={metadata['timestamp']}"
            #     )

    finally:
        print("[camera] stopping...")

        frame_buffer.close()
        camera.stop()

        # 如果之後有 socket，要在這裡 close。
        #
        # streaming_sock.close()
        # detection_sock.close()

        print("[camera] stopped")