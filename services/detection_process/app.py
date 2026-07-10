from __future__ import annotations

import time
from multiprocessing.synchronize import Event as SyncEvent
from typing import Any
from dataclasses import dataclass

from core.config import load_config
from ipc.frame_socket_channel import FrameMetadataReceiver
from ipc.result_socket_channel import DetectionResultSender
from ipc.shared_frame_buffer import SharedFrameBuffer
from services.detection_process.detector import YoloDetector

from services.detection_process.resource_control import configure_detection_process

@dataclass
class DetectionStats:
    """for statistic log"""
    # for frame socket receiving meta
    meta      : int = 0
    fetched   : int = 0
    fetch_miss: int = 0

    # detection results
    detection_count   : int = 0
    total_inference_ms: float = 0.0
    max_inference_ms  : float = 0.0

    # for result output
    output   : int = 0

    def clear(self):
        self.meta = 0
        self.fetched = 0
        self.fetch_miss = 0

        self.detection_count = 0
        self.total_inference_ms = 0.0
        self.max_inference_ms = 0.0

        self.output = 0

    def summarize(self, interval_s: int):
        print(
            f"---------------------\n"
            f"[detection]\n"
            f"avg. detection interval: {0.0 if self.output == 0 else interval_s / self.output:.2f} seconds\n"
            f"avg. inference time    : {0.0 if self.detection_count == 0 else self.total_inference_ms / self.detection_count:.3f}"
            f"max. inference time    : {self.max_inference_ms:.3f}"
            f"meta: {self.meta}, fetched : {self.fetched}, fetch_miss: {self.fetch_miss}\n"
            f"results: {self.detection_count}\n"
            f"output : {self.output}\n"
            "---------------------\n"
        )

def detection_main(
    lock: Any,
    stop_event: SyncEvent,
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

    stats = DetectionStats()

    try:
        print("[detection] started")
        print(f"[detection] metadata_socket={ipc_config.detection_frame_meta_socket}")
        print(f"[detection] shared_memory={shm_config.name}")
        print(f"[detection] frame_shape={frame_shape}")
        print(f"[detection] dtype={frame_dtype}")

        last_log_time_ns = None
        while not stop_event.is_set():
            # logs
            now_ns = time.monotonic_ns()
            if not last_log_time_ns or (now_ns - last_log_time_ns) / 1_000_000_000 >= detection_config.log_every_n_seconds:
                if last_log_time_ns == None:
                    last_log_time_ns = now_ns
                stats.summarize((now_ns - last_log_time_ns) / 1_000_000_000)
                stats.clear()
                last_log_time_ns = now_ns
            
            metadata = metadata_receiver.recv()
            stats.meta += 1
            if metadata is None:
                stats.fetch_miss += 1
                continue
            stats.fetched += 1

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

    finally:
        print("[detection] stopping...")

        metadata_receiver.close()
        result_sender.close()
        frame_buffer.close()

        print("[detection] stopped")