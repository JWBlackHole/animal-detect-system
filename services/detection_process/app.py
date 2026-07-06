from __future__ import annotations

import time
from multiprocessing import Event
from multiprocessing.synchronize import Event as SyncEvent
from typing import Any

from core.config import load_config
from ipc.frame_socket_channel import FrameMetadataReceiver
from ipc.result_socket_channel import DetectionResultSender
from ipc.shared_frame_buffer import SharedFrameBuffer
from services.detection_process.detector import YoloDetector

from services.detection_process.resource_control import configure_detection_process

def _build_cpu_meter():
    try:
        import psutil
    except ImportError:
        return None

    process = psutil.Process()
    process.cpu_percent(interval=None)

    return process

def detection_main(
    lock: Any,
    stop_event: SyncEvent,
) -> None:
    """
    Detection process.

    Responsibilities:
    - Load config.yaml
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

    detection_count = 0
    total_inference_ms = 0.0
    max_inference_ms = 0.0

    try:
        print("[detection] started")
        print(f"[detection] metadata_socket={ipc_config.detection_frame_meta_socket}")
        print(f"[detection] shared_memory={shm_config.name}")
        print(f"[detection] frame_shape={frame_shape}")
        print(f"[detection] dtype={frame_dtype}")

        while not stop_event.is_set():
            metadata = metadata_receiver.recv()

            if metadata is None:
                continue
            

            frame_id = int(metadata["frame_id"])
            metadata_timestamp = int(metadata["timestamp"])

            print(f"[detection] frame_id={frame_id} meta received from socket sender ")

            frame = frame_buffer.read_frame(frame_id)

            if frame is None:
                print(
                    f"[detection] frame_id={frame_id} unavailable "
                    f"(probably overwritten)"
                )
                continue
            print(f"[detection] frame_id={frame_id} image loaded successfully")

            image = frame["image"]

            if image.shape != frame_shape:
                raise RuntimeError(
                    f"Unexpected frame shape: {image.shape}, expected: {frame_shape}"
                )

            if image.dtype != frame_dtype:
                raise RuntimeError(
                    f"Unexpected frame dtype: {image.dtype}, expected: {frame_dtype}"
                )

            start = time.perf_counter()
            boxes = detector.detect(image)
            inference_ms = (time.perf_counter() - start) * 1000.0

            detection_count += 1
            total_inference_ms += inference_ms
            max_inference_ms = max(max_inference_ms, inference_ms)

            avg_inference_ms = total_inference_ms / detection_count

            result = {
                "frame_id": frame_id,
                "timestamp": metadata_timestamp,
                "boxes": boxes,
                "inference_ms": inference_ms,
            }

            if cpu_meter is not None:
                cpu_percent = cpu_meter.cpu_percent(interval=None)
                cpu_text = f", cpu={cpu_percent:.1f}%"
            else:
                cpu_text = ""

            print(
                f"[detection] frame_id={frame_id}, "
                f"boxes={len(boxes)}, "
                f"inference={inference_ms:.1f}ms, "
                f"avg={avg_inference_ms:.1f}ms, "
                f"max={max_inference_ms:.1f}ms"
                f"{cpu_text}"
            )

            # send result to video process
            result_sender.send(result)

    finally:
        print("[detection] stopping...")

        metadata_receiver.close()
        result_sender.close()
        frame_buffer.close()

        print("[detection] stopped")