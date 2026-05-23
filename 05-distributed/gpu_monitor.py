"""
Real-time GPU utilization and memory monitor using pynvml.

Prints a live-updating table of per-GPU: utilization %, memory used/total,
temperature, and power draw.  Run alongside inference scripts to observe
GPU saturation and memory pressure during benchmarks.
"""

import argparse
import time

import pynvml
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()


def build_table() -> Table:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    table = Table(title="GPU Monitor")
    table.add_column("GPU", justify="center")
    table.add_column("Name")
    table.add_column("Util %", justify="right")
    table.add_column("Mem Used (MB)", justify="right")
    table.add_column("Mem Total (MB)", justify="right")
    table.add_column("Temp (°C)", justify="right")
    table.add_column("Power (W)", justify="right")

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W

        table.add_row(
            str(i),
            name if isinstance(name, str) else name.decode(),
            f"{util.gpu}",
            f"{mem.used / 1024**2:.0f}",
            f"{mem.total / 1024**2:.0f}",
            f"{temp}",
            f"{power:.0f}",
        )

    pynvml.nvmlShutdown()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time GPU monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    with Live(build_table(), refresh_per_second=1 / args.interval, console=console) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(build_table())
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped.[/yellow]")


if __name__ == "__main__":
    main()
