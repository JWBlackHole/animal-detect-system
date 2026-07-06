from core.config import load_config
from ipc.result_socket_channel import DetectionResultReceiver


def main() -> None:
    config = load_config()

    receiver = DetectionResultReceiver(
        config.ipc.detection_result_socket,
        timeout=1.0,
    )

    print("[test] detection result receiver started")
    print(f"[test] socket={config.ipc.detection_result_socket}")

    try:
        while True:
            result = receiver.recv()

            if result is None:
                print("[test] waiting...")
                continue

            print(
                "[test] received "
                f"frame_id={result['frame_id']}, "
                f"boxes={len(result['boxes'])}, "
                f"inference_ms={result['inference_ms']:.1f}"
            )

            for box in result["boxes"]:
                print(
                    "    "
                    f"{box['label']} "
                    f"{box['confidence']:.2f} "
                    f"({box['x1']}, {box['y1']}) "
                    f"({box['x2']}, {box['y2']})"
                )

    finally:
        receiver.close()
        print("[test] stopped")


if __name__ == "__main__":
    main()