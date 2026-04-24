from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


JobHandler = Callable[[dict[str, Any]], Any]
QueuePolicy = str


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    payload: dict[str, Any]
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result_ref: dict[str, Any] | None = None
    error: str | None = None
    retries: int = 0


class JobStore:
    """Durable sqlite-backed storage for job metadata and result references."""

    def __init__(self, db_path: str = "translation_jobs.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    result_ref_json TEXT,
                    error TEXT,
                    retries INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def save(self, record: JobRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, payload_json, status, created_at, started_at,
                    finished_at, result_ref_json, error, retries
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type = excluded.job_type,
                    payload_json = excluded.payload_json,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    result_ref_json = excluded.result_ref_json,
                    error = excluded.error,
                    retries = excluded.retries
                """,
                (
                    record.job_id,
                    record.job_type,
                    json.dumps(record.payload),
                    record.status,
                    record.created_at,
                    record.started_at,
                    record.finished_at,
                    json.dumps(record.result_ref) if record.result_ref is not None else None,
                    record.error,
                    record.retries,
                ),
            )

    def get(self, job_id: str) -> JobRecord | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_unfinished(self) -> list[JobRecord]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at ASC"
        ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def list_recent(self, limit: int = 100) -> list[JobRecord]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_job(row) for row in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            job_type=row["job_type"],
            payload=json.loads(row["payload_json"]),
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result_ref=json.loads(row["result_ref_json"]) if row["result_ref_json"] else None,
            error=row["error"],
            retries=row["retries"],
        )


class JobQueueExecutor:
    """Bounded worker pool + policy-driven queueing for translation jobs."""

    def __init__(
        self,
        handlers: dict[str, JobHandler],
        store: JobStore,
        max_workers: int = 2,
        max_queue_depth: int = 20,
        queue_policy: QueuePolicy = "fifo",
        max_retries: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_queue_depth < 1:
            raise ValueError("max_queue_depth must be at least 1")
        if queue_policy not in {"fifo", "lifo", "reject_new", "drop_oldest"}:
            raise ValueError("unsupported queue_policy")

        self.handlers = handlers
        self.store = store
        self.max_workers = max_workers
        self.max_queue_depth = max_queue_depth
        self.queue_policy = queue_policy
        self.max_retries = max_retries

        self._queue: deque[str] = deque()
        self._running = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._recover_unfinished_jobs()

    def submit(self, job_type: str, payload: dict[str, Any]) -> str:
        if job_type not in self.handlers:
            raise ValueError(f"Unknown job type: {job_type}")

        now = time.time()
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, job_type=job_type, payload=payload, status="queued", created_at=now)

        with self._lock:
            if len(self._queue) >= self.max_queue_depth:
                if self.queue_policy == "reject_new":
                    record.status = "rejected"
                    record.finished_at = now
                    record.error = "queue_depth_limit_reached"
                    self.store.save(record)
                    return job_id
                if self.queue_policy == "drop_oldest" and self._queue:
                    dropped = self._queue.popleft()
                    dropped_record = self.store.get(dropped)
                    if dropped_record is not None:
                        dropped_record.status = "dropped"
                        dropped_record.finished_at = now
                        dropped_record.error = "dropped_by_policy"
                        self.store.save(dropped_record)
                elif self.queue_policy in {"fifo", "lifo"}:
                    record.status = "rejected"
                    record.finished_at = now
                    record.error = "queue_depth_limit_reached"
                    self.store.save(record)
                    return job_id

            self.store.save(record)
            self._queue.append(job_id)
            self._dispatch_locked()

        return job_id

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.store.get(job_id)

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queue)

    def _recover_unfinished_jobs(self) -> None:
        for job in self.store.list_unfinished():
            job.status = "queued"
            job.error = "recovered_after_restart"
            self.store.save(job)
            self._queue.append(job.job_id)
        with self._lock:
            self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        while self._running < self.max_workers and self._queue:
            job_id = self._queue.pop() if self.queue_policy == "lifo" else self._queue.popleft()
            self._running += 1
            self._executor.submit(self._run_job, job_id)

    def _run_job(self, job_id: str) -> None:
        try:
            record = self.store.get(job_id)
            if record is None:
                return

            handler = self.handlers[record.job_type]
            record.started_at = time.time()
            record.status = "running"
            self.store.save(record)

            attempt = 0
            while True:
                try:
                    result = handler(record.payload)
                    record.result_ref = result if isinstance(result, dict) else {"value": result}
                    record.status = "completed"
                    record.finished_at = time.time()
                    self.store.save(record)
                    break
                except Exception as exc:  # noqa: BLE001
                    attempt += 1
                    record.retries = attempt
                    if attempt > self.max_retries:
                        record.status = "failed"
                        record.error = str(exc)
                        record.finished_at = time.time()
                        self.store.save(record)
                        break
                    self.store.save(record)
                    time.sleep(0.05)
        finally:
            with self._lock:
                self._running = max(0, self._running - 1)
                self._dispatch_locked()


def build_default_job_executor() -> JobQueueExecutor:
    db_path = os.getenv("TRANSLATION_JOB_DB", "translation_jobs.db")
    max_workers = int(os.getenv("TRANSLATION_JOB_MAX_WORKERS", "2"))
    max_depth = int(os.getenv("TRANSLATION_JOB_MAX_QUEUE_DEPTH", "20"))
    policy = os.getenv("TRANSLATION_JOB_QUEUE_POLICY", "fifo").strip().lower()

    store = JobStore(db_path=db_path)

    _backend = {"instance": None}

    def _get_backend():
        if _backend["instance"] is None:
            from TranslationBackend import TranslationBackend
            _backend["instance"] = TranslationBackend()
        return _backend["instance"]

    def _translate_text_handler(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "")).strip()
        target_language = str(payload.get("target_language", "")).strip()
        if not text:
            raise ValueError("payload.text is required")
        if not target_language:
            raise ValueError("payload.target_language is required")
        translated = _get_backend().translate_text(text, target_language)
        return {
            "translated_text": translated,
            "target_language": target_language,
        }

    handlers = {
        "translate_text": _translate_text_handler,
    }
    return JobQueueExecutor(
        handlers=handlers,
        store=store,
        max_workers=max_workers,
        max_queue_depth=max_depth,
        queue_policy=policy,
    )
