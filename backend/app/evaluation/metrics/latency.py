import time
import numpy as np
from typing import List, Dict, Any, Callable
from contextlib import contextmanager

class LatencyTracker:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.start_time: float = 0.0

    @contextmanager
    def measure(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            if stage_name not in self.timings:
                self.timings[stage_name] = []
            self.timings[stage_name].append(duration_ms)

    def summarize(self) -> Dict[str, Any]:
        summary = {}
        total_e2e = 0.0
        for stage, durations in self.timings.items():
            if not durations:
                continue
            arr = np.array(durations)
            summary[stage] = {
                "count": int(len(arr)),
                "mean_ms": float(np.mean(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "total_ms": float(np.sum(arr))
            }
        return summary
