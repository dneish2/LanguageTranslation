from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_metrics import DashboardThresholds, MetricsDashboard


def test_metrics_dashboard_alerts_for_error_p95_retries_and_queue_depth():
    dashboard = MetricsDashboard(
        DashboardThresholds(error_rate=0.2, p95_duration_seconds=1.0, retry_spike_count=3, queue_depth=3)
    )
    dashboard.ingest_job(status="completed", duration_seconds=0.3, retries=0, queue_depth=1)
    dashboard.ingest_job(status="failed", duration_seconds=1.4, retries=2, queue_depth=3)
    dashboard.ingest_job(status="failed", duration_seconds=1.2, retries=2, queue_depth=4)

    snapshot = dashboard.snapshot()

    assert snapshot["job_count"] == 3
    assert snapshot["error_rate"] > 0.2
    assert snapshot["p95_duration_seconds"] >= 1.0
    assert snapshot["retry_spike_count"] >= 3
    assert snapshot["max_queue_depth"] >= 3
    alert_types = {alert["type"] for alert in snapshot["alerts"]}
    assert {"error_rate", "p95_duration_seconds", "retry_spike_count", "queue_depth"}.issubset(alert_types)
