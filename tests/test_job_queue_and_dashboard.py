import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_queue import JobQueueExecutor, JobStore
from translation_metrics import DashboardThresholds, MetricsDashboard


def _wait_for_status(executor: JobQueueExecutor, job_id: str, expected: set[str], timeout: float = 3.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = executor.get_job(job_id)
        if job and job.status in expected:
            return job.status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} not in expected statuses {expected}")


def test_job_queue_applies_bounded_depth_with_reject_policy(tmp_path: Path):
    db = tmp_path / "jobs.db"

    def slow_handler(payload):
        time.sleep(payload.get("sleep", 0.2))
        return {"result": payload["value"]}

    store = JobStore(str(db))
    executor = JobQueueExecutor(
        handlers={"translate": slow_handler},
        store=store,
        max_workers=1,
        max_queue_depth=1,
        queue_policy="reject_new",
    )

    first = executor.submit("translate", {"value": 1, "sleep": 0.2})
    second = executor.submit("translate", {"value": 2, "sleep": 0.2})
    third = executor.submit("translate", {"value": 3, "sleep": 0.2})

    status_first = _wait_for_status(executor, first, {"running", "completed"})
    assert status_first in {"running", "completed"}

    second_job = executor.get_job(second)
    third_job = executor.get_job(third)
    assert second_job is not None
    assert second_job.status in {"queued", "running", "completed"}
    assert third_job is not None
    assert third_job.status == "rejected"


def test_job_metadata_and_result_persist_across_executor_restart(tmp_path: Path):
    db = tmp_path / "jobs.db"

    def handler(payload):
        return {"result_ref": f"artifact://{payload['value']}"}

    store_a = JobStore(str(db))
    executor_a = JobQueueExecutor(
        handlers={"translate": handler},
        store=store_a,
        max_workers=1,
        max_queue_depth=5,
        queue_policy="fifo",
    )

    job_id = executor_a.submit("translate", {"value": "doc-123"})
    _wait_for_status(executor_a, job_id, {"completed"})

    # New store simulates reconnect/restart.
    store_b = JobStore(str(db))
    restored = store_b.get(job_id)
    assert restored is not None
    assert restored.status == "completed"
    assert restored.result_ref == {"result_ref": "artifact://doc-123"}


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
