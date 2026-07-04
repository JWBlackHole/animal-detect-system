from PiCamera import PiCamera
import cv2

if __name__ == "__main__":
    picam = PiCamera(width = 1280, height = 720, fps = 30)
    picam.start()

    while(True):
        frame: dict = picam.capture_once()
        print(frame["timestamp"])

        # cv2.imwrite("output.jpg", frame["image"])

        input()

        # SEND TO STREMAING

        # SEND TO DETECTION WHEN FRAME_ID % 20 == 0