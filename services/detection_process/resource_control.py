import os
from typing import Iterable

def configure_detection_process(
    cpu_affinity: Iterable[int] | None,
    torch_num_threads: int,
    torch_num_interop_threads: int,
    process_nice: int,
) -> None:
    """
    Configure CPU resources for detection_process.
    - Call this before loading YOLO / torch model.
    - CPU affinity limits which cores this process can run on.
    - thread settings reduce how many CPU threads PyTorch/OpenMP uses.
    """

    # ---------------------------------
    # Limit native library threads
    # ---------------------------------
    os.environ["OMP_NUM_THREADS"]      = str(torch_num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_num_threads)
    os.environ["MKL_NUM_THREADS"]      = str(torch_num_threads)
    os.environ["NUMEXPR_NUM_THREADS"]  = str(torch_num_threads)

    # ---------------------------------
    # CPU affinity
    # ---------------------------------
    if cpu_affinity is not None:
        cores = {int(core_id) for core_id in cpu_affinity}

        if len(cores) == 0:
            raise ValueError("cpu_affinity cannot be empty.")

        try:
            os.sched_setaffinity(0, cores)
            print(f"[resource] cpu_affinity={sorted(cores)}")
        except AttributeError:
            print("[resource] os.sched_setaffinity is not available on this platform")
        except PermissionError:
            print("[resource] permission denied when setting cpu affinity")

    # ---------------------------------
    # Nice value
    # ---------------------------------
    try:
        os.nice(process_nice)
        print(f"[resource] process_nice=+{process_nice}")
    except OSError as exc:
        print(f"[resource] failed to set nice value: {exc}")

    # ---------------------------------
    # PyTorch thread control
    # ---------------------------------
    try:
        import torch

        torch.set_num_threads(torch_num_threads)
        torch.set_num_interop_threads(torch_num_interop_threads)

        print(f"[resource] torch_num_threads={torch_num_threads}")
        print(f"[resource] torch_num_interop_threads={torch_num_interop_threads}")

    except ImportError:
        print("[resource] torch is not installed; skip torch thread config")
    except RuntimeError as exc:
        print(f"[resource] failed to set torch threads: {exc}")