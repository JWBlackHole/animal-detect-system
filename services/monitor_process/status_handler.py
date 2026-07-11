import json
from http.server import BaseHTTPRequestHandler
from services.monitor_process.monitor_stats import MonitorStats

class StatusHandler(BaseHTTPRequestHandler):
    stats: MonitorStats | None = None

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path != "/api/status":
            self.send_error(404)
            return

        if self.stats is None:
            self.send_error(500)
            return

        payload = json.dumps(
            self.stats.response(),
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(payload)),
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.end_headers()
        self.wfile.write(payload)

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        return