# ipc/shared_frame_buffer.py

from multiprocessing.synchronize import Lock as LockBase
from multiprocessing.shared_memory import SharedMemory

from typing import Any, Optional

import numpy as np


class SharedFrameBuffer:
    """Shared buffer to read/write frames, implementation of inter-process communication"""

    HEADER_DTYPE = np.dtype([
        ("valid", "u1"),
        ("frame_id", "i8"),
        ("timestamp", "i8"),
    ])

    def __init__(
        self,
        name: str,
        frame_shape: tuple[int, int, int],
        dtype: np.dtype,
        buffer_size: int,
        lock: LockBase,
        create: bool = False,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        
        # name
        self.name        : str = name
        self.header_name : str = f"{name}_header"

        # frame info
        self.frame_shape: tuple[int, int, int] = frame_shape

        # lock to access shared memory
        self.lock: LockBase = lock

        # sahred memory info
        self.create     : bool = create
        self.dtype      : np.dtype = np.dtype(dtype)
        self.buffer_size: int = buffer_size

        # create shared memory
        data_nbytes = int(np.prod(frame_shape) * self.dtype.itemsize) * buffer_size
        self.data_shm: SharedMemory = SharedMemory(
            name=self.name,
            create=create,
            size=data_nbytes if create else 0,
        )
        header_nbytes = self.HEADER_DTYPE.itemsize * buffer_size
        self.header_shm: SharedMemory = SharedMemory(
            name=self.header_name,
            create=create,
            size=header_nbytes if create else 0,
        )

        # interpret shared memory into ndarray
        self.frames: np.ndarray = np.ndarray(
            shape=(buffer_size, *frame_shape),
            dtype=self.dtype,
            buffer=self.data_shm.buf,
        )

        self.headers: np.ndarray = np.ndarray(
            shape=(buffer_size,),
            dtype=self.HEADER_DTYPE,
            buffer=self.header_shm.buf,
        )

        # initialize buffer
        if create:
            self.frames[:] = 0
            self.headers["valid"] = 0
            self.headers["frame_id"] = -1
            self.headers["timestamp"] = 0

    def _slot_index(self, frame_id: int) -> int:
        """get slot index by modding"""
        return frame_id % self.buffer_size

    def write_frame(
        self,
        frame_id: int,
        timestamp: int,
        image: np.ndarray,
    ) -> int:
        frame_id = int(frame_id)
        timestamp = int(timestamp)

        if image.shape != self.frame_shape:
            raise ValueError(
                f"Invalid image shape: {image.shape}, expected: {self.frame_shape}"
            )

        if image.dtype != self.dtype:
            raise ValueError(
                f"Invalid image dtype: {image.dtype}, expected: {self.dtype}"
            )

        slot = self._slot_index(frame_id)

        with self.lock:
            self.headers[slot]["valid"] = 0

            self.frames[slot] = image

            self.headers[slot]["timestamp"] = timestamp
            self.headers[slot]["frame_id"] = frame_id
            self.headers[slot]["valid"] = 1

        return slot

    def read_frame(self, frame_id: int) -> Optional[dict]:
        frame_id = int(frame_id)
        slot = self._slot_index(frame_id)

        with self.lock:
            if int(self.headers[slot]["valid"]) != 1:
                return None

            stored_frame_id = int(self.headers[slot]["frame_id"])

            if stored_frame_id != frame_id:
                return None

            timestamp = int(self.headers[slot]["timestamp"])

            image = self.frames[slot].copy()

        return {
            "frame_id" : frame_id,
            "timestamp": timestamp,
            "image"    : image
        }

    def close(self) -> None:
        self.data_shm.close()
        self.header_shm.close()

    def unlink(self) -> None:
        try:
            self.data_shm.unlink()
        except FileNotFoundError:
            pass

        try:
            self.header_shm.unlink()
        except FileNotFoundError:
            pass