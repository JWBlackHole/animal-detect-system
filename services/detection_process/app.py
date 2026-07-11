from __future__ import annotations

import time
import queue

from multiprocessing.synchronize import Event as SyncEvent
from multiprocessing import Queue
from typing import Any

from core.config import load_config
from core.stats import DetectionStats, _build_cpu_meter
from ipc.frame_socket_channel import FrameMetadataReceiver
from ipc.result_socket_channel import DetectionResultSender
from ipc.shared_frame_buffer import SharedFrameBuffer
from services.detection_process.detector import YoloDetector

from services.detection_process.resource_control import configure_detection_process

def detection_main(
    lock: Any,
    stop_event: SyncEvent,
    stats_queue: Any
) -> None:
    """
    Detection process.
    - Receive frame metadata from camera_process
    - Fetch frame from SharedFrameBuffer by frame_id
    - Run YOLO inference
    - Build detection result
    """

    config = load_config()

    camera_config = config.camera
    shm_config = config.shared_memory
    detection_config = config.detection
    ipc_config = config.ipc

    if detection_config.enabled == False:
        print("[detection] detection is disabled in config.yaml")
        return
    
    configure_detection_process(
        cpu_affinity=detection_config.cpu_affinity,
        torch_num_threads=detection_config.torch_num_threads,
        torch_num_interop_threads=detection_config.torch_num_interop_threads,
        process_nice=detection_config.process_nice,
    )

    frame_shape = camera_config.frame_shape
    frame_dtype = camera_config.dtype

    frame_buffer = SharedFrameBuffer(
        name=shm_config.name,
        frame_shape=frame_shape,
        dtype=frame_dtype,
        buffer_size=shm_config.buffer_size,
        lock=lock,
        create=False,
    )

    # socket setting
    metadata_receiver = FrameMetadataReceiver(
        ipc_config.detection_frame_meta_socket,
        timeout=0.1,
    )

    result_sender = DetectionResultSender(
        ipc_config.detection_result_socket,
        strict=False,
    )

    detector = YoloDetector(
        model_path=detection_config.model_path,
        image_size=detection_config.image_size,
        confidence=detection_config.confidence,
    )

    cpu_meter = _build_cpu_meter()

    stats = DetectionStats()

    try:
        print("[detection] started")
        print(f"[detection] metadata_socket={ipc_config.detection_frame_meta_socket}")
        print(f"[detection] shared_memory={shm_config.name}")
        print(f"[detection] frame_shape={frame_shape}")
        print(f"[detection] dtype={frame_dtype}")

        session_start_time_ns = None
        while not stop_event.is_set():
            now_ns = time.monotonic_ns()

            # ------ logs stats ------ #
            if session_start_time_ns == None:
                session_start_time_ns = now_ns
            elif (now_ns - session_start_time_ns) >= detection_config.log_every_n_seconds * 1_000_000_000:
                # TO-DO: print logs and send to monitor process
                try:
                    stats_queue.put_nowait(stats.to_stats((now_ns - session_start_time_ns) / 1_000_000_000, cpu_meter.cpu_percent(interval=None)))
                except queue.Full:
                    pass
                stats.summarize((now_ns - session_start_time_ns) / 1_000_000_000)
                stats.clear()
                session_start_time_ns += detection_config.log_every_n_seconds * 1_000_000_000

                
            # ------ receving metadata ------ #
            metadata = metadata_receiver.recv()
            if metadata is None:
                continue
            stats.meta += 1

            frame_id = int(metadata["frame_id"])
            metadata_timestamp = int(metadata["timestamp"])

            # print(f"[detection] frame_id={frame_id} meta received from socket sender ")
            frame = frame_buffer.read_frame(frame_id)

            if frame is None:
                print(
                    f"[detection] frame_id={frame_id} unavailable "
                    f"(probably overwritten)"
                )
                continue
            # print(f"[detection] frame_id={frame_id} image loaded successfully")

            image = frame["image"]

            if image.shape != frame_shape:
                raise RuntimeError(
                    f"Unexpected frame shape: {image.shape}, expected: {frame_shape}"
                )

            if image.dtype != frame_dtype:
                raise RuntimeError(
                    f"Unexpected frame dtype: {image.dtype}, expected: {frame_dtype}"
                )

            # detection by yolo
            start = time.perf_counter()
            boxes = detector.detect(image)
            inference_ms = (time.perf_counter() - start) * 1000.0
            # update stats
            stats.detection_count += 1
            stats.total_inference_ms += inference_ms
            stats.max_inference_ms = max(stats.max_inference_ms, inference_ms)

            result = {
                "frame_id": frame_id,
                "timestamp": metadata_timestamp,
                "boxes": boxes,
                "inference_ms": inference_ms,
            }

            # print(
            #     f"[detection] frame_id={frame_id}, "
            #     f"boxes={len(boxes)}, "
            #     f"inference={inference_ms:.1f}ms, "
            #     f"avg={avg_inference_ms:.1f}ms, "
            #     f"max={max_inference_ms:.1f}ms"
            #     f"{cpu_text}"
            # )

            # send result to video process
            result_sender.send(result)
            stats.output += 1

    finally:
        print("[detection] stopping...")

        metadata_receiver.close()
        result_sender.close()
        frame_buffer.close()

        print("[detection] stopped")