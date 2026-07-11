from typing import Any

import queue
import threading
from http.server import ThreadingHTTPServer
from multiprocessing.synchronize import Event as SyncEvent

from services.monitor_process.monitor_stats import MonitorStats
from services.monitor_process.status_handler import StatusHandler

def _consume_stats(
    *,
    stats_queue: Any,
    stats: MonitorStats,
    stop_event: SyncEvent,
) -> None:
    while not stop_event.is_set():
        try:
            message = stats_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if isinstance(message, dict):
            stats.update(message)


def monitor_main(
    stats_queue: Any,
    stop_event: SyncEvent,
) -> None:
    stats = MonitorStats()

    consumer_thread = threading.Thread(
        target=_consume_stats,
        kwargs={
            "stats_queue": stats_queue,
            "stats": stats,
            "stop_event": stop_event,
        },
        daemon=True,
        name="monitor-stats-consumer",
    )
    consumer_thread.start()

    StatusHandler.stats = stats

    server = ThreadingHTTPServer(
        ("0.0.0.0", 8000),
        StatusHandler,
    )
    server.timeout = 0.5

    print("---------------------------")
    print("[monitor] monitor process started")
    print("[monitor] API: http://0.0.0.0:8000/api/status")
    print("---------------------------")

    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        print("[monitor] stopping...")
        server.server_close()
        print("[monitor] stopped")