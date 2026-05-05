"""
Background GPU memory monitor using PyTorch CUDA.

Monitors GPU memory allocation during long-running processes via PyTorch CUDA
queries. Designed for unified-memory systems (e.g., NVIDIA GB10) where
nvidia-smi reports [N/A] for memory.used, making direct monitoring impossible.

Polls GPU memory at a configurable interval and logs to CSV with timestamp,
device info, and memory metrics. Catches SIGTERM/SIGINT for graceful shutdown,
always writing a final row with peak memory on exit.

Usage
-----
uv run python scripts/monitor_gpu_mem.py <csv_path> [interval_seconds]

Parameters
----------
csv_path : str
    Path to output CSV file. File is created/overwritten.
interval_seconds : int, optional
    Polling interval in seconds. Defaults to 10.
    Set to 0 to disable polling; only write peak on exit.

CSV Output Format
-----------------
Columns:
    timestamp : int
        UNIX timestamp (seconds since epoch)
    device : int
        GPU device index (e.g., 0 for /dev/nvidia0)
    name : str
        GPU device name (e.g., "NVIDIA A100-PCIE-40GB")
    allocated_mb : float
        Currently allocated GPU memory (MB)
    max_allocated_mb : float
        Peak allocated GPU memory since process start (MB)
    reserved_mb : float
        Currently reserved GPU memory (MB)

Behavior
--------
- SIGTERM/SIGINT: Gracefully stops polling and writes final peak row
- interval=0: Waits for signal, writes only peak on exit
- interval>0: Polls at specified interval, always writes peak on exit
- Sleep in 0.1s increments for responsive signal handling

Examples
--------
Monitor every 10 seconds (default):
    python monitor_gpu_mem.py gpu_memory.csv

Monitor every 5 seconds:
    python monitor_gpu_mem.py gpu_memory.csv 5

Wait for signal, write peak only:
    python monitor_gpu_mem.py gpu_memory.csv 0

Notes
-----
- Requires CUDA-capable GPU and torch.cuda availability
- Excellent for tracking memory growth during COLMAP/NeRF training
- Peak memory metric shows maximum allocation, useful for OOM debugging
- Reserved memory often exceeds allocated due to CUDA caching
"""

import csv
import signal
import sys
import time
import torch.cuda as cuda

POLL_INTERVAL_DEFAULT: int = 10


def _row(writer: csv.writer) -> None:  # type: ignore[type-arg]
    """Write one CSV row with current CUDA memory stats.

    Parameters
    ----------
    writer : csv.writer
        The CSV writer object used to write the row.
    """
    writer.writerow([
        int(time.time()),
        cuda.current_device(),
        cuda.get_device_name(),
        f"{cuda.memory_allocated() / (1024**2):.1f}",
        f"{cuda.max_memory_allocated() / (1024**2):.1f}",
        f"{cuda.memory_reserved() / (1024**2):.1f}",
    ])


def monitor(csv_path: str, interval: int) -> None:
    """Poll GPU memory and write to `csv_path`.

    Parameters
    ----------
    csv_path : str
        Destination CSV file.
    interval : int
        Seconds between samples.  0 = no polling (write on exit only).
    """
    # ─── Ensure CUDA is initialised ───
    if not cuda.is_available():
        print("CUDA not available, exiting.", file=sys.stderr)
        sys.exit(1)
    cuda.init()

    stop = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "timestamp",
            "device",
            "name",
            "allocated_mb",
            "max_allocated_mb",
            "reserved_mb",
        ])

        if interval > 0:
            while not stop:
                _row(writer)
                fh.flush()
                # ─── Sleep in small increments so SIGTERM is responsive ───
                for _ in range(interval * 10):
                    if stop:
                        break
                    time.sleep(0.1)

        else:
            # interval == 0: just wait for signal
            while not stop:
                time.sleep(0.5)

        # ─── Final row captures peak regardless of interval ───
        _row(writer)
        fh.flush()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_path> [interval_seconds]",
              file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else POLL_INTERVAL_DEFAULT
    monitor(csv_path, interval)
