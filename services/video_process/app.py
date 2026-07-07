import cv2

import select
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event as SyncEvent
from typing import Any, Optional

from core.config import load_config
from ipc.frame_socket_channel import FrameMetadataReceiver
from ipc.result_socket_channel import DetectionResultReceiver
from ipc.shared_frame_buffer import SharedFrameBuffer
from services.video_process.video_delay_buffer import VideoDelayBuffer

POLL_TIMEOUT_S = 0.010

RESULT_WORK_SLACK_NS = 1_000_000  # 1ms
META_WORK_SLACK_NS = 3_000_000    # 3ms, read_frame copy is heavier

SUMMARY_EVERY_N_OUTPUTS = 30

@dataclass
class VideoStats:
    target_fps: Optional[int] = None

    # for frame socket receiving meta
    meta      : int = 0
    fetched   : int = 0
    fetch_miss: int = 0

    # for result socket recieing detection result
    results: int = 0
    matched: int = 0
    late   : int = 0

    # for video output
    output   : int = 0
    dropped  : int = 0
    underflow: int = 0

    first_frame_ns: Optional[int] = None

    last_output_ns: Optional[int] = None

    # debug
    test_start = False


def video_test_main(lock: Any, stop_event: SyncEvent) -> None:
    config = load_config()

    camera_config = config.camera
    shm_config = config.shared_memory
    ipc_config = config.ipc
    video_config = config.video

    frame_buffer = SharedFrameBuffer(
        name=shm_config.name,
        frame_shape=camera_config.frame_shape,
        dtype=camera_config.dtype,
        buffer_size=shm_config.buffer_size,
        lock=lock,
        create=False,
    )

    meta_receiver = FrameMetadataReceiver(
        ipc_config.video_frame_meta_socket,
        timeout=None,
    )

    result_receiver = DetectionResultReceiver(
        ipc_config.detection_result_socket,
        timeout=None,
    )

    video_buffer = VideoDelayBuffer(buffer_size=video_config.buffer_size)
    stats = VideoStats()

    sockets = [meta_receiver.sock, result_receiver.sock]

    output_interval_ns = int(1_000_000_000 / video_config.output_fps)
    startup_delay_ns = int(video_config.startup_delay_ms * 1_000_000)

    output_started: bool = False
    next_output_ns: Optional[int] = None

    stats.target_fps = video_config.output_fps

    try:
        print("---------------------------")
        print("[video-test] videos process started")
        print("---------------------------")

        while not stop_event.is_set():
            now_ns = time.monotonic_ns()

            # compute first output time
            if next_output_ns == None and stats.first_frame_ns is not None:
                next_output_ns = stats.first_frame_ns + startup_delay_ns
            # output started
            if output_started == False and next_output_ns and now_ns >= next_output_ns:
                output_started = True
                print("[video-test] frame output started")

            # output frame
            if output_started and now_ns >= next_output_ns:
                _output_one_frame(video_buffer=video_buffer, stats=stats)
                next_output_ns += output_interval_ns
                continue
            
            # wait for socket
            now_ns = time.monotonic_ns() # output frame may cost some time
            readable, _, _ = select.select(
                sockets,
                [],
                [],
                min(max(0, next_output_ns - now_ns) / 1_000_000_000, POLL_TIMEOUT_S) if next_output_ns else POLL_TIMEOUT_S,
            )

            # go back to output frame if exceed deadline
            now_ns = time.monotonic_ns()
            if output_started and now_ns >= next_output_ns:
                continue
            
            # receive result socket and add label
            if (
                result_receiver.sock in readable
                and _has_time_before_output(
                    next_output_ns=next_output_ns,
                    slack_ns=RESULT_WORK_SLACK_NS,
                )
            ):
                _receive_one_result(
                    result_receiver=result_receiver,
                    video_buffer=video_buffer,
                    stats=stats,
                )

            # receive frame meta and read shared memory
            if (
                meta_receiver.sock in readable
                and _has_time_before_output(
                    next_output_ns=next_output_ns,
                    slack_ns=META_WORK_SLACK_NS,
                )
            ):
                _receive_one_frame(
                    meta_receiver=meta_receiver,
                    frame_buffer=frame_buffer,
                    video_buffer=video_buffer,
                    stats=stats,
                )

    finally:
        print("[video-test] stopping...")

        meta_receiver.close()
        result_receiver.close()
        frame_buffer.close()

        print("[video-test] stopped")

def _output_one_frame(
    *,
    video_buffer: VideoDelayBuffer,
    stats: VideoStats,
) -> None:
    frame = video_buffer.pop_frame()

    if frame is None:
        stats.underflow += 1
        return

    stats.output += 1

    output_image = frame.image
    
    if(stats.test_start == False and frame.frame_id == 600):
        stats.test_start = True
        cv2.imwrite("output.jpg", output_image)
    
    
    now_ns = time.monotonic_ns()
    # if stats.last_output_ns:
    #     interval_ns = now_ns - stats.last_output_ns
    #     if(abs(interval_ns - 1_000_000_000 / stats.target_fps) > 2_000_000):
    #         print(f"[video-test] frame_id={frame.frame_id}, unusul output interval(ms): {interval_ns / 1_000_000}")
    stats.last_output_ns = now_ns

    # log
    # print(
    #     f"[video-test] output frame_id={frame.frame_id}, "
    #     f"pending={video_buffer.used_n_slot}"
    # )
    # if(stats.last_output_ns):
    #     print(f"[video-test] output interval(ms): {interval_ns / 1_000_000}")

def _has_time_before_output(
    *,
    next_output_ns: Optional[int],
    slack_ns: int,
) -> bool:
    if not next_output_ns:
        return True
    return time.monotonic_ns() + slack_ns < next_output_ns

def _receive_one_frame(
    *,
    meta_receiver: FrameMetadataReceiver,
    frame_buffer: SharedFrameBuffer,
    video_buffer: VideoDelayBuffer,
    stats: VideoStats,
) -> None:
    metadata = meta_receiver.recv()

    if metadata is None:
        return

    stats.meta += 1

    frame_id = int(metadata["frame_id"])
    frame = frame_buffer.read_frame(frame_id)

    if frame is None:
        stats.fetch_miss += 1
        return

    stats.fetched += 1

    if stats.first_frame_ns is None:
        stats.first_frame_ns = time.monotonic_ns()

    dropped = video_buffer.append_frame(
        frame_id=int(frame["frame_id"]),
        timestamp=int(frame["timestamp"]),
        image=frame["image"],
    )

    if dropped is not None:
        stats.dropped += 1

def _receive_one_result(
    *,
    result_receiver: DetectionResultReceiver,
    video_buffer: VideoDelayBuffer,
    stats: VideoStats,
) -> None:
    result = result_receiver.recv()

    if result is None:
        return

    stats.results += 1

    frame_id = int(result["frame_id"])

    matched = video_buffer.draw_boxes_on_frame(
        frame_id=frame_id,
        boxes=result["boxes"],
    )

    if matched:
        stats.matched += 1
    else:
        stats.late += 1

    print(
        f"[video-test] result frame_id={frame_id}, "
        f"boxes={len(result['boxes'])}, "
        f"matched={matched}, "
        f"pending={video_buffer.used_n_slot}"
    )