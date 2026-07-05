from core.config import load_config
from ipc.frame_socket_channel import FrameMetadataReceiver


def main() -> None:
    config = load_config()

    receiver = FrameMetadataReceiver(
        config.ipc.detection_frame_meta_socket,
        timeout=1.0,
    )
    # receiver = FrameMetadataReceiver(
    #     config.ipc.video_frame_meta_socket,
    #     timeout=1.0,
    # )

    print("[test] detection frame metadata receiver started")

    try:
        while True:
            metadata = receiver.recv()

            if metadata is None:
                print("[test] waiting...")
                continue

            print("[test] received:", metadata)

    finally:
        receiver.close()
        print("[test] stopped")


if __name__ == "__main__":
    main()