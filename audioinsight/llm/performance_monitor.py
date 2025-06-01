import time
from typing import Any, Dict, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor and track performance of LLM workers."""

    def __init__(self):
        self.metrics = {
            "parser": {
                "total_requests": 0,
                "total_time": 0.0,
                "queue_times": [],
                "processing_times": [],
                "errors": 0,
                "timeouts": 0,
            },
            "summarizer": {
                "total_requests": 0,
                "total_time": 0.0,
                "queue_times": [],
                "processing_times": [],
                "errors": 0,
                "timeouts": 0,
            },
        }
        self.last_report_time = time.time()

    def record_request(self, component: str, processing_time: float, queue_time: float = 0.0):
        """Record a completed request."""
        if component not in self.metrics:
            return

        metrics = self.metrics[component]
        metrics["total_requests"] += 1
        metrics["total_time"] += processing_time
        metrics["processing_times"].append(processing_time)

        if queue_time > 0:
            metrics["queue_times"].append(queue_time)

        # Keep only recent samples for moving averages
        if len(metrics["processing_times"]) > 100:
            metrics["processing_times"] = metrics["processing_times"][-50:]
        if len(metrics["queue_times"]) > 100:
            metrics["queue_times"] = metrics["queue_times"][-50:]

    def record_error(self, component: str, error_type: str = "general"):
        """Record an error."""
        if component not in self.metrics:
            return

        if error_type == "timeout":
            self.metrics[component]["timeouts"] += 1
        else:
            self.metrics[component]["errors"] += 1

    def get_stats(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        if component:
            return self._get_component_stats(component)

        return {comp: self._get_component_stats(comp) for comp in self.metrics.keys()}

    def _get_component_stats(self, component: str) -> Dict[str, Any]:
        """Get statistics for a specific component."""
        if component not in self.metrics:
            return {}

        metrics = self.metrics[component]

        avg_processing_time = 0.0
        avg_queue_time = 0.0

        if metrics["processing_times"]:
            avg_processing_time = sum(metrics["processing_times"]) / len(metrics["processing_times"])

        if metrics["queue_times"]:
            avg_queue_time = sum(metrics["queue_times"]) / len(metrics["queue_times"])

        return {
            "total_requests": metrics["total_requests"],
            "avg_processing_time": avg_processing_time,
            "avg_queue_time": avg_queue_time,
            "recent_processing_times": metrics["processing_times"][-10:],
            "errors": metrics["errors"],
            "timeouts": metrics["timeouts"],
            "success_rate": self._calculate_success_rate(metrics),
        }

    def _calculate_success_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate success rate for a component."""
        total = metrics["total_requests"]
        failures = metrics["errors"] + metrics["timeouts"]

        if total == 0:
            return 0.0

        return (total - failures) / total * 100.0

    def should_report(self, interval: float = 60.0) -> bool:
        """Check if it's time to report performance stats."""
        current_time = time.time()
        if current_time - self.last_report_time >= interval:
            self.last_report_time = current_time
            return True
        return False

    def generate_report(self) -> str:
        """Generate a performance report."""
        report_lines = ["=== LLM Performance Report ==="]

        for component, stats in self.get_stats().items():
            report_lines.append(f"\n{component.upper()}:")
            report_lines.append(f"  Total Requests: {stats['total_requests']}")
            report_lines.append(f"  Avg Processing Time: {stats['avg_processing_time']:.2f}s")
            report_lines.append(f"  Avg Queue Time: {stats['avg_queue_time']:.2f}s")
            report_lines.append(f"  Success Rate: {stats['success_rate']:.1f}%")
            report_lines.append(f"  Errors: {stats['errors']}, Timeouts: {stats['timeouts']}")

            if stats["recent_processing_times"]:
                recent_avg = sum(stats["recent_processing_times"]) / len(stats["recent_processing_times"])
                report_lines.append(f"  Recent Avg Time: {recent_avg:.2f}s")

        return "\n".join(report_lines)


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def log_performance_if_needed():
    """Log performance report if enough time has passed."""
    monitor = get_performance_monitor()
    if monitor.should_report():
        logger.info(monitor.generate_report())
