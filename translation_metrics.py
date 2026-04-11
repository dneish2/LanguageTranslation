from __future__ import annotations

from dataclasses import dataclass
from statistics import quantiles
from typing import Protocol


class TranslationMetrics(Protocol):
    """Lightweight instrumentation contract reusable by async/job workers."""

    def start_file(self, file_type: str, correlation_id: str | None = None) -> None: ...

    def record_segment(self, duration_seconds: float, token_count: int, estimated_cost: float) -> None: ...

    def add_segment_duration(self, duration_seconds: float) -> None: ...

    def record_cache_hit(self) -> None: ...

    def record_cache_miss(self) -> None: ...

    def record_retry(self) -> None: ...

    def finish_file(self, file_type: str, segment_count: int, duration_seconds: float) -> None: ...

    def snapshot(self) -> dict: ...


@dataclass
class MetricsCollector:
    correlation_id: str | None = None
    file_type: str | None = None
    duration_seconds: float = 0.0
    segment_count: int = 0
    segment_duration_seconds: float = 0.0
    token_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    estimated_cost: float = 0.0

    def start_file(self, file_type: str, correlation_id: str | None = None) -> None:
        self.correlation_id = correlation_id
        self.file_type = file_type
        self.duration_seconds = 0.0
        self.segment_count = 0
        self.segment_duration_seconds = 0.0
        self.token_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.retries = 0
        self.estimated_cost = 0.0

    def record_segment(self, duration_seconds: float, token_count: int, estimated_cost: float) -> None:
        self.segment_count += 1
        self.segment_duration_seconds += max(0.0, duration_seconds)
        self.token_count += max(0, token_count)
        self.estimated_cost += max(0.0, estimated_cost)

    def add_segment_duration(self, duration_seconds: float) -> None:
        self.segment_duration_seconds += max(0.0, duration_seconds)

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def record_retry(self) -> None:
        self.retries += 1

    def finish_file(self, file_type: str, segment_count: int, duration_seconds: float) -> None:
        self.file_type = file_type
        self.segment_count = max(self.segment_count, segment_count)
        self.duration_seconds = max(0.0, duration_seconds)

    def snapshot(self) -> dict:
        return {
            "correlation_id": self.correlation_id,
            "file_type": self.file_type,
            "duration_seconds": round(self.duration_seconds, 3),
            "segment_count": self.segment_count,
            "segment_duration_seconds": round(self.segment_duration_seconds, 3),
            "token_count": self.token_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retries": self.retries,
            "estimated_cost": round(self.estimated_cost, 6),
        }


@dataclass
class DashboardThresholds:
    error_rate: float = 0.1
    p95_duration_seconds: float = 30.0
    retry_spike_count: int = 10
    queue_depth: int = 20


class MetricsDashboard:
    """Aggregates recent job metrics for dashboard widgets and alerting rules."""

    def __init__(self, thresholds: DashboardThresholds | None = None) -> None:
        self.thresholds = thresholds or DashboardThresholds()
        self._job_rows: list[dict] = []

    def ingest_job(
        self,
        *,
        status: str,
        duration_seconds: float,
        retries: int,
        queue_depth: int,
        correlation_id: str | None = None,
    ) -> None:
        self._job_rows.append(
            {
                "status": status,
                "duration_seconds": max(0.0, duration_seconds),
                "retries": max(0, retries),
                "queue_depth": max(0, queue_depth),
                "correlation_id": correlation_id,
            }
        )

    def snapshot(self) -> dict:
        total = len(self._job_rows)
        if total == 0:
            return {
                "job_count": 0,
                "error_rate": 0.0,
                "p95_duration_seconds": 0.0,
                "retry_spike_count": 0,
                "max_queue_depth": 0,
                "alerts": [],
            }

        failures = sum(1 for row in self._job_rows if row["status"] in {"failed", "rejected", "dropped"})
        error_rate = failures / total
        durations = [row["duration_seconds"] for row in self._job_rows]
        p95_duration = (
            quantiles(durations, n=100, method="inclusive")[94]
            if len(durations) > 1
            else durations[0]
        )
        retry_spike_count = sum(row["retries"] for row in self._job_rows)
        max_queue_depth = max(row["queue_depth"] for row in self._job_rows)

        alerts: list[dict[str, str | float | int]] = []
        if error_rate >= self.thresholds.error_rate:
            alerts.append({"type": "error_rate", "value": round(error_rate, 4), "threshold": self.thresholds.error_rate})
        if p95_duration >= self.thresholds.p95_duration_seconds:
            alerts.append(
                {
                    "type": "p95_duration_seconds",
                    "value": round(p95_duration, 3),
                    "threshold": self.thresholds.p95_duration_seconds,
                }
            )
        if retry_spike_count >= self.thresholds.retry_spike_count:
            alerts.append(
                {
                    "type": "retry_spike_count",
                    "value": retry_spike_count,
                    "threshold": self.thresholds.retry_spike_count,
                }
            )
        if max_queue_depth >= self.thresholds.queue_depth:
            alerts.append(
                {
                    "type": "queue_depth",
                    "value": max_queue_depth,
                    "threshold": self.thresholds.queue_depth,
                }
            )

        return {
            "job_count": total,
            "error_rate": round(error_rate, 4),
            "p95_duration_seconds": round(p95_duration, 3),
            "retry_spike_count": retry_spike_count,
            "max_queue_depth": max_queue_depth,
            "alerts": alerts,
        }
