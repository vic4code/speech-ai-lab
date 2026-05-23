"""
Prometheus metrics exporter and Grafana dashboard generator for the ASR API.

Exports custom metrics and prints the Grafana dashboard JSON to stdout so
it can be imported via the Grafana UI or provisioned via ConfigMap.
"""

import json
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from rich.console import Console

console = Console()

GATEWAY_URL = "http://localhost:9091"


def push_gpu_metrics(registry: CollectorRegistry | None = None) -> None:
    """Push real-time GPU stats to Prometheus Pushgateway."""
    try:
        import pynvml
        pynvml.nvmlInit()
        reg = registry or CollectorRegistry()

        gpu_util = Gauge("gpu_utilization_percent", "GPU utilization %", ["gpu_index"], registry=reg)
        gpu_mem = Gauge("gpu_memory_used_mb", "GPU memory used MB", ["gpu_index"], registry=reg)

        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util.labels(gpu_index=str(i)).set(util.gpu)
            gpu_mem.labels(gpu_index=str(i)).set(mem.used / 1024**2)

        push_to_gateway(GATEWAY_URL, job="asr_gpu_metrics", registry=reg)
        pynvml.nvmlShutdown()
    except Exception as e:
        console.print(f"[yellow]GPU metrics push failed: {e}[/yellow]")


GRAFANA_DASHBOARD = {
    "title": "Speech AI Lab — ASR Dashboard",
    "panels": [
        {
            "title": "Request Rate (req/s)",
            "type": "graph",
            "targets": [{"expr": "rate(asr_requests_total[1m])"}],
        },
        {
            "title": "P95 Latency (s)",
            "type": "graph",
            "targets": [{"expr": "histogram_quantile(0.95, rate(asr_latency_seconds_bucket[5m]))"}],
        },
        {
            "title": "GPU Utilization (%)",
            "type": "graph",
            "targets": [{"expr": "gpu_utilization_percent"}],
        },
        {
            "title": "GPU Memory Used (MB)",
            "type": "graph",
            "targets": [{"expr": "gpu_memory_used_mb"}],
        },
    ],
}


def main() -> None:
    out = Path("results/grafana_dashboard.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(GRAFANA_DASHBOARD, indent=2))
    console.print(f"Grafana dashboard JSON saved → {out}")
    console.print("Import via: Grafana → Dashboards → Import → Upload JSON file")


if __name__ == "__main__":
    main()
