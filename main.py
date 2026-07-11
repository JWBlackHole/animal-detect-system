import time
from pathlib import Path
from multiprocessing import Lock, Event, Process, Queue

from core.config import load_config
from core.process_logging import run_with_log_file

from ipc.shared_frame_buffer import SharedFrameBuffer
from services.camera_process.app import camera_main
from services.video_process.app import video_main
from services.detection_process.app import detection_main
from services.monitor_process.app import monitor_main

CONFIG_PATH = Path("config.yaml")

def main() -> None:
    # load config
    
    config = load_config(config_path = CONFIG_PATH)

    frame_shape = config.camera.frame_shape
    frame_dtype = config.camera.dtype

    buffer_name = config.shared_memory.name
    buffer_size = config.shared_memory.buffer_size

    # async setting
    lock = Lock()
    stop_event = Event()
    stats_queue = Queue()

    # creates shared buffer
    frame_buffer = SharedFrameBuffer(
        name=buffer_name,
        frame_shape=frame_shape,
        dtype=frame_dtype,
        buffer_size=buffer_size,
        lock=lock,
        create=True,
    )
    # detached from parent process
    frame_buffer.close()

    print("---------------------------")
    print("[main] SharedFrameBuffer created")
    print("---------------------------")

    # camera_process = Process(
    #     name="camera_process",
    #     target=camera_main,
    #     args=(lock, stop_event, stats_queue),
    # )
    # detection_process = Process(
    #     name="detection_process",
    #     target=detection_main,
    #     args=(lock, stop_event, stats_queue),
    # )
    # video_process = Process(
    #     name="video_process",
    #     target=video_test_main,
    #     args=(lock, stop_event, stats_queue),
    # )

    # with logging
    monitor_process = Process(
        name="monitor_process",
        target=run_with_log_file,
        kwargs={
            "process_name": "monitor_process",
            "target": monitor_main,
            "args": (stats_queue, stop_event),
        },
    )
    camera_process = Process(
        name="camera_process",
        target=run_with_log_file,
        kwargs={
            "process_name": "camera_process",
            "target": camera_main,
            "args": (lock, stop_event, stats_queue),
        }
    )
    detection_process = Process(
        name="detection_process",
        target=run_with_log_file,
        kwargs={
            "process_name": "detection_process",
            "target": detection_main,
            "args": (lock, stop_event, stats_queue),
        },
    )

    video_process = Process(
        name="video_process",
        target=run_with_log_file,
        kwargs={
            "process_name": "video_process",
            "target": video_main,
            "args": (lock, stop_event, stats_queue),
        },
    )

    try:
        monitor_process.start()
        print("[main] monitor_process started")

        detection_process.start()
        print("[main] detection_process started")

        time.sleep(10) # waiting for detection model being loaded

        camera_process.start()
        print("[main] camera_process started")

        video_process.start()
        print("[main] video_process started")

        while camera_process.is_alive():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt received")
        stop_event.set()

    finally:
        print("[main] shutting down...")

        stop_event.set()

        if camera_process.is_alive():
            camera_process.join(timeout=3)

        if camera_process.is_alive():
            print("[main] camera_process did not stop gracefully, terminating...")
            camera_process.terminate()
            camera_process.join(timeout=3)

        cleanup_buffer = SharedFrameBuffer(
            name=buffer_name,
            frame_shape=frame_shape,
            dtype=frame_dtype,
            buffer_size=buffer_size,
            lock=lock,
            create=False,
        )

        cleanup_buffer.close()
        cleanup_buffer.unlink()

        print("[main] SharedFrameBuffer unlinked")
        print("[main] stopped")


if __name__ == "__main__":
    main()