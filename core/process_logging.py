import sys
import traceback
from pathlib import Path
from typing import Callable, Any

def run_with_log_file(
    *,
    process_name: str,
    target: Callable[..., None],
    args: tuple[Any, ...],
    log_dir: str | Path = "logs",
) -> None:
    """
    Redirect stdout/stderr of current process to logs/{process_name}.log,
    then run target(*args).
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{process_name}.log"

    with log_path.open("w", buffering=1, encoding="utf-8", errors="replace") as f:
        sys.stdout = f
        sys.stderr = f

        print("=" * 60)
        print(f"[{process_name}] process started")
        print(f"[{process_name}] log file: {log_path}")
        print("=" * 60)

        try:
            target(*args)

        except Exception:
            print(f"[{process_name}] crashed with exception:")
            traceback.print_exc()
            raise

        finally:
            print("=" * 60)
            print(f"[{process_name}] process exited")
            print("=" * 60)