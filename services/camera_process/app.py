# services/camera_process/app.py

import queue

from multiprocessing.synchronize import Event as SyncEvent
from multiprocessing import Queue
from typing import Any, Optional
import time

from core.config import load_config
from core.stats import CameraStats, _build_cpu_meter
from ipc.frame_socket_channel import FrameMetadataSender
from ipc.shared_frame_buffer import SharedFrameBuffer
from services.camera_process.pi_camera import PiCamera

def camera_main(
    lock: Any,
    stop_event: SyncEvent,
    stats_queue: Any
) -> None:
    """
    Camera process
    1. Start camera
    2. Write frames into shared memory
    3. Send metadata to video_process and detection_process
    """

    # load config
    config = load_config()

    camera_config    = config.camera
    shm_config       = config.shared_memory
    detection_config = config.detection
    ipc_config       = config.ipc

    frame_shape = camera_config.frame_shape
    frame_dtype = camera_config.dtype

    if detection_config is not None:
        detect_every_n_frames = detection_config.detect_every_n_frames
    else:
        detect_every_n_frames = None

    # initialize camera
    camera = PiCamera(
        width=camera_config.width,
        height=camera_config.height,
        fps=camera_config.fps,
        pixel_format=camera_config.pixel_format,
    )

    # initialize buffer
    frame_buffer = SharedFrameBuffer(
        name=shm_config.name,
        frame_shape=frame_shape,
        dtype=frame_dtype,
        buffer_size=shm_config.buffer_size,
        lock=lock,
        create=False,
    )

    # initialized socket sender
    video_meta_sender = FrameMetadataSender(
        ipc_config.video_frame_meta_socket,
        strict=False,
    )

    # initialized socket receiver
    detection_meta_sender = FrameMetadataSender(
        ipc_config.detection_frame_meta_socket,
        strict=False,
    )
    cpu_meter = _build_cpu_meter()

    stats = CameraStats()

    try:
        camera.start()
        print("[camera] started")
        print(f"[camera] frame_shape={frame_shape}")
        print(f"[camera] dtype={frame_dtype}")
        print(f"[camera] shared_memory={shm_config.name}")
        print(f"[camera] buffer_size={shm_config.buffer_size}")
        print(f"[camera] video_meta_socket={ipc_config.video_frame_meta_socket}")
        print(f"[camera] detection_meta_socket={ipc_config.detection_frame_meta_socket}")

        if detect_every_n_frames is not None:
            print(f"[camera] detection enabled, every {detect_every_n_frames} frames")
        else:
            print("[camera] detection disabled")
        
        session_start_time_ns = None
        last_capture_time_ns = None
        while not stop_event.is_set():
            now_ns = time.monotonic_ns()

            # ------ logs stats ------ #
            if session_start_time_ns == None:
                session_start_time_ns = now_ns
            elif (now_ns - session_start_time_ns) >= camera_config.log_every_n_seconds * 1_000_000_000:
                # TO-DO: print logs and send to monitor process
                try:
                    stats_queue.put_nowait(stats.to_stats((now_ns - session_start_time_ns) / 1_000_000_000, cpu_meter.cpu_percent(interval=None)))
                except queue.Full:
                    pass
                stats.clear()
                session_start_time_ns += camera_config.log_every_n_seconds * 1_000_000_000
            
            # ------ camera capture ------ #
            frame = camera.capture_once()
            

            frame_id = int(frame["frame_id"])
            timestamp = int(frame["timestamp"])
            image = frame["image"]

            # update stats
            now_ns = time.monotonic_ns()
            if(last_capture_time_ns):
                stats.max_frame_interval_ns = max(stats.max_frame_interval_ns, now_ns - last_capture_time_ns)
            stats.capture_count += 1
            stats.last_frame_id = frame_id
            last_capture_time_ns = now_ns


            # print(f"frame_id: {frame_id}, get frame")

            if image.shape != frame_shape:
                raise RuntimeError(
                    f"Unexpected camera image shape: {image.shape}, "
                    f"expected: {frame_shape}"
                )

            if image.dtype != frame_dtype:
                raise RuntimeError(
                    f"Unexpected camera image dtype: {image.dtype}, "
                    f"expected: {frame_dtype}"
                )


            # ====== Write to Shared Memory ====== #
            slot = frame_buffer.write_frame(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
            )

            # ====== Socket ====== #
            metadata = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "slot": slot,
            }
            
            # ------- Send to Detection Process ------- #
            # print(frame_id % detect_every_n_frames)
            if detect_every_n_frames and frame_id % detect_every_n_frames == 0:
                # print(f"frame_id: {frame_id}, sent to detection process")
                detection_meta_sender.send(metadata)
                

            # ------- Send to Video Process ------- #
            # print(f"frame_id: {frame_id}, before video send", flush=True)
            video_meta_sender.send(metadata)
            # print(f"frame_id: {frame_id}, after video send", flush=True)

    finally:
        print("[camera] stopping...")

        video_meta_sender.close()
        detection_meta_sender.close()

        frame_buffer.close()
        camera.stop()

        print("[camera] stopped")