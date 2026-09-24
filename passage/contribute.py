"""Opt-in contribution of corrections to a private bucket (Cloud mode).

A Cloud deployment keeps nothing by default: its disk is wiped on restart, and
DECISIONS.md §10 says the hosted service stores nothing unless the person using
it turns on "contribute corrections". When they do, the trace rows for that
session (segment pairs, edits, approvals, never an original file) are queued
here and uploaded in small batches to ``gs://$PASSAGE_CONTRIBUTION_BUCKET``.

Layout: ``contributions/YYYY-MM-DD/<session>/<epoch>-<id>.jsonl``. One object per
flush per session, because GCS objects cannot be appended to. Reading it back is
``gcloud storage cp -r gs://BUCKET/contributions ./dump`` and then
``python -m passage.export --in ./dump``.

No client library: on Cloud Run the service account token comes from the
metadata server, and a JSON upload is one POST. Everything is best effort and
off the request path. A failed upload is logged and dropped, never raised into
a translation.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from typing import Any, Callable

BUCKET = os.getenv("PASSAGE_CONTRIBUTION_BUCKET", "").strip()

#: Rows arriving within this window go up as one object per session: a
#: document's generations land together rather than as hundreds of objects.
FLUSH_SECONDS = float(os.getenv("PASSAGE_CONTRIBUTION_FLUSH_SECONDS", "5"))

_METADATA_TOKEN_URL = ("http://metadata.google.internal/computeMetadata/v1/"
                       "instance/service-accounts/default/token")

_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=10_000)
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_token: dict[str, Any] = {"value": None, "expires": 0.0}
_stopping = threading.Event()

#: The only row types that leave the machine. The switch promises "approved and
#: edited segment pairs", so a generation row (every segment, reviewed or not)
#: or a trace row (it carries the uploaded file's name) must never be queued,
#: whatever a caller routes here. Correction rows carry their own source text
#: and language (traces.record_edit), so export needs nothing else.
CONTRIBUTED_TYPES = frozenset({"score", "judgement"})


def available() -> bool:
    """Whether contributing is possible here at all. The UI only offers the
    switch when it is, so the offer never outruns the capability."""
    return bool(BUCKET)


def enqueue(row: dict[str, Any]) -> None:
    if not available() or row.get("type") not in CONTRIBUTED_TYPES:
        return
    try:
        _queue.put_nowait(row)
    except queue.Full:
        logging.warning("[Contribute] queue full; dropping a row")
        return
    _ensure_worker()


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="passage-contribute", daemon=True)
            _worker.start()


def _run() -> None:
    while not (_stopping.is_set() and _queue.empty()):
        try:
            first = _queue.get(timeout=0.5)
        except queue.Empty:
            continue
        batch = [first]
        deadline = time.time() + FLUSH_SECONDS
        while not _stopping.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(_queue.get(timeout=min(remaining, 0.5)))
            except queue.Empty:
                continue
        if _stopping.is_set():
            while True:
                try:
                    batch.append(_queue.get_nowait())
                except queue.Empty:
                    break
        flush(batch)


def drain(timeout: float = 8.0) -> None:
    """Flush whatever is queued before the process exits.

    Prod scales to zero: once the last tab closes, the instance is stopped
    with SIGTERM and a daemon thread dies mid-window, taking the last few
    seconds of corrections with it (the approval a visitor clicks right before
    closing the tab is exactly that row). Cloud Run allows 10 s after SIGTERM.
    """
    _stopping.set()
    worker = _worker
    if worker is not None and worker.is_alive():
        worker.join(timeout)


atexit.register(drain)


def flush(rows: list[dict[str, Any]]) -> list[str]:
    """Upload `rows` grouped by session. Returns the object names written."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row.get("session") or "anonymous")].append(row)
    written = []
    for session, group in by_session.items():
        name = (f"contributions/{time.strftime('%Y-%m-%d', time.gmtime())}/"
                f"{session}/{int(time.time())}-{uuid.uuid4().hex[:8]}.jsonl")
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group).encode("utf-8")
        try:
            uploader(name, body)
            written.append(name)
        except Exception as error:
            logging.warning("[Contribute] upload of %d row(s) failed (%s)", len(group), error)
    return written


def _access_token() -> str:
    if _token["value"] and time.time() < _token["expires"] - 60:
        return _token["value"]
    request = urllib.request.Request(_METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read())
    _token["value"] = payload["access_token"]
    _token["expires"] = time.time() + float(payload.get("expires_in", 300))
    return _token["value"]


def _upload_gcs(name: str, body: bytes) -> None:
    url = ("https://storage.googleapis.com/upload/storage/v1/b/"
           f"{urllib.parse.quote(BUCKET, safe='')}/o?uploadType=media&name="
           f"{urllib.parse.quote(name, safe='')}")
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {_access_token()}",
        "Content-Type": "application/x-ndjson",
    })
    with urllib.request.urlopen(request, timeout=15):
        pass


#: Replaced in tests with a fake that records what would have been uploaded.
uploader: Callable[[str, bytes], None] = _upload_gcs
