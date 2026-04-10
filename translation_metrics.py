from __future__ import annotations

from dataclasses import dataclass
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
