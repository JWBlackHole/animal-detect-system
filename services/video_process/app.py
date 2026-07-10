import numpy as np

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
from services.video_process.ffmpeg_publisher import FFmpegRTSPPublisher
from services.video_process.image_boxes_helper import draw_boxes, get_boxes

POLL_TIMEOUT_S = 0.010

RESULT_WORK_SLACK_NS = 1_000_000  # 1ms
META_WORK_SLACK_NS = 3_000_000    # 3ms, read_frame copy is heavier

@dataclass
class VideoStats:
    """for statistic log"""
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

    def clear(self):
        self.meta = 0
        self.fetched = 0
        self.fetch_miss = 0

        self.results = 0
        self.matched = 0
        self.late = 0

        self.output = 0
        self.dropped = 0
        self.underflow = 0

    def summarize(self, pending: int, interval_s: int):
        print(
            f"---------------------\n"
            f"[video]\n"
            f"avg. fps: {0.0 if interval_s == 0.0 else self.output / interval_s:.2f}\n"
            f"meta: {self.meta}, fetched : {self.fetched}, fetch_miss: {self.fetch_miss}\n"
            f"results: {self.results}, matched: {self.matched}, late: {self.late}\n"
            f"output : {self.output}, dropped   : {self.dropped}, underflow : {self.underflow}\n"
            f"buffer pending: {pending}\n"
            "---------------------\n"
        )


def video_test_main(lock: Any, stop_event: SyncEvent) -> None:
    config = load_config()

    camera_config = config.camera
    shm_config = config.shared_memory
    detection_config = config.detection
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

    publisher = FFmpegRTSPPublisher(
        width=camera_config.width,
        height=camera_config.height,
        fps=camera_config.fps,
        pixel_format=camera_config.pixel_format,
        rtsp_url="rtsp://127.0.0.1:8554/live",
        codec="libx264",
        bitrate="1200k",
        gop_seconds=1,
    )
    publisher.start()

    sockets = [meta_receiver.sock, result_receiver.sock]

    first_frame_ns    : Optional[int] = None
    startup_delay_ns  : int = int(video_config.startup_delay_ms * 1_000_000)
    output_interval_ns: int = int(1_000_000_000 / camera_config.fps)

    labels: dict[int,dict] = {}

    output_started: bool = False
    next_output_ns: Optional[int] = None

    last_log_time_ns = None
    try:
        print("---------------------------")
        print("[video] videos process started")
        print("---------------------------")

        while not stop_event.is_set():
            now_ns = time.monotonic_ns()

            # compute first output time
            if next_output_ns == None and first_frame_ns is not None:
                next_output_ns = first_frame_ns + startup_delay_ns
            # output started
            if output_started == False and next_output_ns and now_ns >= next_output_ns:
                output_started = True
                print(f"[video] frame output started, summarize every {video_config.log_every_n_seconds} seconds")

            # after output started, it logs and outputs frames
            if(output_started):
                # logs
                if not last_log_time_ns or (now_ns - last_log_time_ns) / 1_000_000_000 >= video_config.log_every_n_seconds:
                    if last_log_time_ns == None:
                        last_log_time_ns = first_frame_ns
                    stats.summarize(video_buffer.used_n_slot, (now_ns - last_log_time_ns) / 1_000_000_000)
                    stats.clear()
                    last_log_time_ns = now_ns

                # draw boxes and output 1 frame
                if now_ns >= next_output_ns:
                    # get frame
                    frame = video_buffer.pop_frame()
                    if frame is None:
                        stats.underflow += 1
                        next_output_ns += output_interval_ns
                        continue
                    # get corresponded boxes
                    boxes = get_boxes(frame.frame_id, labels, detection_config.detect_every_n_frames)
                    # draw boexes on image
                    draw_boxes(frame.image, boxes)

                    # output one frame
                    _output_one_frame(publisher, frame.image, stats)

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
                    labels=labels,
                    stats=stats
                )

            # receive frame meta and read shared memory
            if (
                meta_receiver.sock in readable
                and _has_time_before_output(
                    next_output_ns=next_output_ns,
                    slack_ns=META_WORK_SLACK_NS,
                )
            ):
                ok = _receive_one_frame(
                    meta_receiver=meta_receiver,
                    frame_buffer=frame_buffer,
                    video_buffer=video_buffer,
                    stats=stats,
                )
                if ok and first_frame_ns == None:
                    first_frame_ns = time.monotonic_ns()
    finally:
        print("[video] stopping...")

        meta_receiver.close()
        result_receiver.close()
        frame_buffer.close()
        publisher.stop()

        print("[video] stopped")

def _has_time_before_output(
    *,
    next_output_ns: Optional[int],
    slack_ns: int,
) -> bool:
    if not next_output_ns:
        return True
    return time.monotonic_ns() + slack_ns < next_output_ns

def _output_one_frame(publisher: FFmpegRTSPPublisher, image: np.ndarray, stats: VideoStats):
    ok = publisher.write(image)
    stats.output += 1
    if not ok:
        print("[video] failed to publish frame")

def _receive_one_frame(
    *,
    meta_receiver: FrameMetadataReceiver,
    frame_buffer: SharedFrameBuffer,
    video_buffer: VideoDelayBuffer,
    stats: VideoStats,
) -> bool:
    metadata = meta_receiver.recv()

    if metadata is None:
        return

    stats.meta += 1

    frame_id = int(metadata["frame_id"])
    frame = frame_buffer.read_frame(frame_id)

    if frame is None:
        stats.fetch_miss += 1
        return False

    stats.fetched += 1

    dropped = video_buffer.append_frame(
        frame_id=int(frame["frame_id"]),
        timestamp=int(frame["timestamp"]),
        image=frame["image"],
    )

    if dropped is not None:
        stats.dropped += 1
    
    return True

def _receive_one_result(
    *,
    result_receiver: DetectionResultReceiver,
    video_buffer: VideoDelayBuffer,
    labels: dict[int,dict],
    stats: VideoStats
) -> bool:
    result = result_receiver.recv()

    if result is None:
        return False

    frame_id = int(result["frame_id"])

    stats.results += 1
    labels[frame_id] = result["boxes"]

    matched = video_buffer.in_buffer(frame_id)

    if matched:
        stats.matched += 1
    else:
        stats.late += 1

    return True