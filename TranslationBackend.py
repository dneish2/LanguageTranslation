import logging
import os
import string
import time
import uuid
import json
import random
from dataclasses import dataclass, field
from threading import Lock, Thread
from html import escape
from io import BytesIO
from typing import Any, Callable, Optional, Tuple

import dotenv
import fitz  # PyMuPDF
import openai
import tiktoken
from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Pt

from translation_metrics import MetricsCollector, TranslationMetrics

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("translation.backend")

WHITE = (1, 1, 1)  # RGB white for PDF overwrite
MODEL_COST_PER_1K_TOKENS = 0.002


JOB_STATE_QUEUED = "queued"
JOB_STATE_RUNNING = "running"
JOB_STATE_SUCCEEDED = "succeeded"
JOB_STATE_FAILED = "failed"
JOB_STATE_CANCELED = "canceled"


@dataclass
class TranslationJob:
    job_id: str
    state: str = JOB_STATE_QUEUED
    progress: float = 0.0
    status_message: str = "Queued"
    result_handle: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def _log_event(event: str, correlation_id: str | None = None, **fields: Any) -> None:
    payload: dict[str, Any] = {"event": event, **fields}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    LOGGER.info(json.dumps(payload, default=str))


class TranslationBackend:
    """Handles GPT-based text/document translation and experimental voice I/O."""

    # ─────────────────────────── INITIALISATION ────────────────────────── #
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        self.client = openai.OpenAI(api_key=self.api_key)

        self.cancel_requested = False
        self.segment_map: dict[str, dict] = {}
        self.current_file_type = None
        self.current_document = None
        self.current_presentation = None
        self.current_pdf = None
        self.output_stream: BytesIO | None = None
        self.pdf_overlay_ocg = None
        self.translation_cache: dict[tuple[str, str, str], str] = {}
        self.max_openai_attempts = 4
        self.retry_base_delay = 0.5
        self.retry_max_delay = 8.0
        self.metrics = MetricsCollector()
        self._jobs: dict[str, TranslationJob] = {}
        self._job_results: dict[str, dict[str, Any]] = {}
        self._jobs_lock = Lock()

    def reset_cancel(self) -> None:
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True
        logging.info("[Backend] Cancel requested.")

    def start_translation_job(
        self,
        *,
        input_stream: BytesIO,
        file_extension: str,
        target_language: str,
        processed: bool = False,
        font_size: int | None = None,
        autofit: bool = False,
        correlation_id: str | None = None,
    ) -> str:
        job_id = self.generate_segment_id()
        job = TranslationJob(
            job_id=job_id,
            state=JOB_STATE_QUEUED,
            progress=0.0,
            status_message="Queued",
            metadata={
                "file_extension": file_extension,
                "target_language": target_language,
                "processed": processed,
                "correlation_id": correlation_id,
            },
        )
        with self._jobs_lock:
            self._jobs[job_id] = job

        def worker():
            self._run_translation_job(
                job_id=job_id,
                input_stream=input_stream,
                file_extension=file_extension,
                target_language=target_language,
                processed=processed,
                font_size=font_size,
                autofit=autofit,
                correlation_id=correlation_id,
            )

        Thread(target=worker, daemon=True).start()
        return job_id

    def get_job(self, job_id: str) -> TranslationJob | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def get_job_result(self, result_handle: str) -> dict[str, Any] | None:
        with self._jobs_lock:
            return self._job_results.get(result_handle)

    def cancel_job(self, job_id: str) -> bool:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state in {JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELED}:
                return False
            job.state = JOB_STATE_CANCELED
            job.status_message = "Cancel requested."
            job.updated_at = time.time()
        self.request_cancel()
        return True

    def _set_job_state(
        self,
        job_id: str,
        *,
        state: str | None = None,
        progress: float | None = None,
        status_message: str | None = None,
        result_handle: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if state is not None:
                job.state = state
            if progress is not None:
                job.progress = max(0.0, min(100.0, progress))
            if status_message is not None:
                job.status_message = status_message
            if result_handle is not None:
                job.result_handle = result_handle
            if error is not None:
                job.error = error
            job.updated_at = time.time()

    def _run_translation_job(
        self,
        *,
        job_id: str,
        input_stream: BytesIO,
        file_extension: str,
        target_language: str,
        processed: bool,
        font_size: int | None,
        autofit: bool,
        correlation_id: str | None,
    ) -> None:
        self._set_job_state(job_id, state=JOB_STATE_RUNNING, status_message="Starting translation...")
        file_metrics = MetricsCollector()
        try:
            out_stream, count, tokens, text_accum, seg_map = self.translate_file(
                input_stream=input_stream,
                file_extension=file_extension,
                target_language=target_language,
                processed=processed,
                font_size=font_size,
                autofit=autofit,
                correlation_id=correlation_id,
                file_metrics=file_metrics,
                progress_callback=lambda progress, message: self._set_job_state(
                    job_id,
                    progress=progress,
                    status_message=message,
                ),
            )
            if self.cancel_requested:
                self._set_job_state(
                    job_id,
                    state=JOB_STATE_CANCELED,
                    status_message="Translation canceled.",
                )
                return

            result_handle = self.generate_segment_id()
            with self._jobs_lock:
                self._job_results[result_handle] = {
                    "output_stream": out_stream,
                    "count": count,
                    "tokens": tokens,
                    "text": text_accum,
                    "segment_map": seg_map,
                    "metrics": file_metrics.snapshot(),
                }
            self._set_job_state(
                job_id,
                state=JOB_STATE_SUCCEEDED,
                progress=100.0,
                status_message="Translation complete.",
                result_handle=result_handle,
            )
        except Exception as exc:
            self._set_job_state(
                job_id,
                state=JOB_STATE_FAILED,
                status_message="Translation failed.",
                error=str(exc),
            )

    def generate_segment_id(self) -> str:
        return str(uuid.uuid4())

    def _normalize_cache_key(self, text: str, target_language: str, mode: str) -> tuple[str, str, str]:
        normalized_text = " ".join(text.replace("\t", " ").split())
        normalized_target = " ".join(target_language.lower().split())
        normalized_mode = " ".join(mode.lower().split())
        return normalized_text, normalized_target, normalized_mode

    def _is_transient_openai_error(self, error: Exception) -> bool:
        transient_types = tuple(
            err
            for err in (
                getattr(openai, "APIConnectionError", None),
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "RateLimitError", None),
                getattr(openai, "InternalServerError", None),
            )
            if err is not None
        )

        if transient_types and isinstance(error, transient_types):
            return True

        status_code = getattr(error, "status_code", None)
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    def _create_chat_completion_with_retry(self, messages):
        attempts = self.max_openai_attempts
        for attempt in range(1, attempts + 1):
            try:
                return self.client.chat.completions.create(model="gpt-4.1-nano", messages=messages, max_tokens=4000)
            except Exception as error:
                is_transient = self._is_transient_openai_error(error)
                if attempt >= attempts or not is_transient:
                    raise

                delay = min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)
                jitter = random.uniform(0, delay * 0.25)
                sleep_seconds = delay + jitter
                logging.warning(
                    "[Backend] transient OpenAI error (%s) on attempt %d/%d, retrying in %.2fs",
                    type(error).__name__,
                    attempt,
                    attempts,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    # ───────────────────────────── GPT CORE ────────────────────────────── #
    def translate_text(
        self,
        text: str,
        target_language: str,
        correlation_id: str | None = None,
        file_metrics: TranslationMetrics | None = None,
    ) -> str:
        """Translate free-form text via GPT."""
        text = text.replace("\t", " ").strip()
        if not text:
            return text
        metrics = file_metrics or self.metrics
        cache_key = (target_language, text)
        if cache_key in self.translation_cache:
            cached = self.translation_cache[cache_key]
            metrics.record_cache_hit()
            _log_event(
                "translation.cache_hit",
                correlation_id=correlation_id,
                source_length=len(text),
                translated_length=len(cached),
            )
            return cached
        metrics.record_cache_miss()

        cache_key = self._normalize_cache_key(text, target_language, mode="translate")
        cached = self.translation_cache.get(cache_key)
        if cached is not None:
            logging.info("[Backend] translate_text cache hit for target=%s", target_language)
            return cached

        prompt = (
            f"Translate the following text to {target_language}, preserving meaning and context. "
            f"Do not translate personal names or trademarked terms. "
            "If it's an email address or URL, leave it unchanged.\n\n"
            f"Text:\n{text}"
        )
        messages = [
            {"role": "system", "content": f"You are a helpful assistant translating to {target_language}."},
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self._create_chat_completion_with_retry(messages)
            result = completion.choices[0].message.content.strip() or text
            self.translation_cache[cache_key] = result
            logging.info("[Backend] Translated len=%d → len=%d", len(text), len(result))
            return result
        except Exception as e:
            logging.error("[Backend] translate_text error: %s", e, exc_info=True)
            return text

    def translate_text_with_instructions(
        self, original_text: str, target_language: str, instructions: str
    ) -> str:
        """Refine an existing translation with user instructions."""
        original_text = original_text.replace("\t", " ").strip()
        if not original_text:
            return original_text

        normalized_instructions = " ".join(instructions.replace("\t", " ").strip().split())
        mode = f"instructions:{normalized_instructions}" if normalized_instructions else "instructions"
        cache_key = self._normalize_cache_key(original_text, target_language, mode=mode)
        cached = self.translation_cache.get(cache_key)
        if cached is not None:
            logging.info("[Backend] instruction translation cache hit for target=%s", target_language)
            return cached

        prompt = (
            "Please refine the following translation according to these instructions. "
            "Ensure that any requested changes—including changing the language—are applied.\n\n"
            f"Instructions: {instructions}\n\n"
            f"Original text: {original_text}\n\n"
            "Final translation:"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant refining translations."},
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self._create_chat_completion_with_retry(messages)
            result = completion.choices[0].message.content.strip() or original_text
            self.translation_cache[cache_key] = result
            logging.info("[Backend] Refined translation len=%d", len(result))
            return result
        except Exception as e:
            logging.error("[Backend] refine translation error: %s", e, exc_info=True)
            return original_text

    # ──────────────────────── VOICE (WHISPER + TTS) ────────────────────── #
    def translate_audio(self, audio_bytes: bytes, target_language: str) -> Tuple[str, bytes]:
        """
        1. Transcribe `audio_bytes` with Whisper.  
        2. Translate resulting text.  
        3. Return TTS MP3 bytes of the translation.
        """
        try:
            logging.info("[Backend] Voice pipeline start → %s (%d bytes)", target_language, len(audio_bytes))
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "speech.webm"  # Whisper needs a filename

            transcription = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            source_text: str = transcription.text
            logging.info("[Backend] Whisper transcription: %s", source_text[:60] + "…")

            translated_text = self.translate_text(source_text, target_language)

            tts_resp = self.client.audio.speech.create(
                model="tts-1", voice="nova", input=translated_text, response_format="mp3"
            )
            audio_mp3: bytes = tts_resp.content if hasattr(tts_resp, "content") else tts_resp
            logging.info("[Backend] TTS done (%d bytes)", len(audio_mp3))
            return translated_text, audio_mp3
        except Exception as e:
            logging.error("[Backend] translate_audio error: %s", e, exc_info=True)
            raise

    # ─────────────────────────── TOKEN COUNTS ─────────────────────────── #
    def calculate_tokens(self, total_text: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(encoding.encode(total_text))

    # ──────────────────────── SMALL PDF HELPER ────────────────────────── #
    def is_meaningful_text(self, text: str) -> bool:
        norm = text.strip().lower().replace(" ", "")
        if norm in {"tm", "™", "©", "®"}:
            return False
        stripped = text.translate(str.maketrans("", "", string.punctuation + "™©®")).strip()
        return any(ch.isalnum() for ch in stripped)

    # ───────────────────── OUTPUT REGENERATION ────────────────────────── #
    def regenerate_output_stream(self) -> Optional[BytesIO]:
        if self.current_file_type == "docx" and self.current_document:
            out = BytesIO(); self.current_document.save(out); out.seek(0); self.output_stream = out
        elif self.current_file_type == "pptx" and self.current_presentation:
            out = BytesIO(); self.current_presentation.save(out); out.seek(0); self.output_stream = out
        elif self.current_file_type == "pdf" and self.current_pdf:
            out = BytesIO(); self.current_pdf.ez_save(out); out.seek(0); self.output_stream = out
        return self.output_stream

    def _compute_pdf_block_css(self, text: str, bbox: fitz.Rect) -> str:
        """Estimate a legible font size for the block based on its bounding box."""
        line_count = max(1, text.count("\n") + 1)
        usable_height = max(1.0, bbox.height)
        size_from_height = (usable_height / line_count) * 0.8
        font_size = max(8.0, min(size_from_height, 36.0))
        return (
            "body {margin:0;} "
            "div {font-family: sans-serif; line-height:1.1; "
            f"font-size:{font_size:.1f}pt;"
            "}"
        )

    def _render_pdf_block(self, page: fitz.Page, bbox: fitz.Rect, text: str) -> str:
        """Overwrite an existing PDF block and insert updated HTML content."""
        oc = getattr(self, "pdf_overlay_ocg", None)
        safe_html = escape(text).replace("\n", "<br />") or "&nbsp;"
        css = self._compute_pdf_block_css(text, bbox)
        page.draw_rect(bbox, color=None, fill=WHITE, oc=oc)
        page.insert_htmlbox(
            bbox,
            f"<div>{safe_html}</div>",
            css=css,
            overlay=True,
            oc=oc,
        )
        return css


    def delete_segment(self, segment_id):
        if segment_id not in self.segment_map:
            raise ValueError(f"Segment ID {segment_id} not found.")
        seg = self.segment_map[segment_id]
        seg_type = seg["type"]
        # DOCX paragraph or table cell
        if seg_type in ["paragraph", "table_cell"]:
            if "object" in seg:
                seg["object"].text = ""
        # PPTX shape
        elif seg_type == "pptx_shape":
            if "object" in seg and hasattr(seg["object"], "text_frame"):
                seg["object"].text_frame.text = ""

        # PDF text block
        elif seg_type == "pdf_block":
            # seg["page_idx"] is the zero-based page index
            # seg["bbox"] is a fitz.Rect
            if hasattr(self, "current_pdf") and self.current_pdf:
                oc = getattr(self, "pdf_overlay_ocg", None)
                page = self.current_pdf[seg["page_idx"]]
                page.draw_rect(seg["bbox"], color=None, fill=WHITE, oc=oc)
            else:
                logging.warning(f"[Backend] No current_pdf to delete PDF block {segment_id}")

        else:
            # Unknown segment types are simply logged
            logging.warning(f"[Backend] delete_segment: unhandled segment type '{seg_type}' for ID {segment_id}")

        # Remove from map and regenerate output
        del self.segment_map[segment_id]
        logging.info(f"[Backend] {segment_id[:8]} removed; {len(self.segment_map)} segments remain.")
        self.regenerate_output_stream()

    # ------------------
    # PROCESSING DOCX
    # ------------------
    def process_docx(
        self,
        input_stream,
        target_language,
        progress_ui=None,
        label_ui=None,
        do_translate=True,
        correlation_id: str | None = None,
        file_metrics: TranslationMetrics | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        self.reset_cancel()
        metrics = file_metrics or self.metrics
        metrics.start_file(file_type="docx", correlation_id=correlation_id)
        doc = Document(input_stream)
        self.current_file_type = 'docx'
        self.current_document = doc
        self.pdf_overlay_ocg = None

        total_elements = len(doc.paragraphs) + sum(len(t.rows)*len(t.columns) for t in doc.tables)
        processed = 0
        text_accum = ""
        start_time = time.time()

        # Paragraphs
        for idx, para in enumerate(doc.paragraphs):
            original = para.text.strip()
            if not original:
                continue
            if self.cancel_requested:
                break
            seg_start = time.time()
            new_text = (
                self.translate_text(original, target_language, correlation_id=correlation_id, file_metrics=metrics)
                if do_translate else original
            )
            para.text = new_text
            text_accum += new_text + "\n"
            seg_id = self.generate_segment_id()
            self.segment_map[seg_id] = {
                "type": "paragraph",
                "location": f"docx:paragraph:{idx}",
                "original": original,
                "translated": new_text,
                "metadata": {"format": "docx", "index": idx},
                "object": para
            }
            processed += 1
            metrics.add_segment_duration(time.time() - seg_start)
            self.update_progress(
                processed,
                total_elements,
                start_time,
                progress_ui=progress_ui,
                label_ui=label_ui,
                progress_callback=progress_callback,
            )

        # Table cells
        for t_idx, table in enumerate(doc.tables):
            for r_idx, row in enumerate(table.rows):
                for c_idx, cell in enumerate(row.cells):
                    for p_idx, para in enumerate(cell.paragraphs):
                        original = para.text.strip()
                        if not original:
                            continue
                        if self.cancel_requested:
                            break
                        seg_start = time.time()
                        new_text = (
                            self.translate_text(
                                original,
                                target_language,
                                correlation_id=correlation_id,
                                file_metrics=metrics,
                            )
                            if do_translate else original
                        )
                        para.text = new_text
                        text_accum += new_text + "\n"
                        seg_id = self.generate_segment_id()
                        self.segment_map[seg_id] = {
                            "type": "table_cell",
                            "location": f"docx:table:{t_idx}:row:{r_idx}:col:{c_idx}:para:{p_idx}",
                            "original": original,
                            "translated": new_text,
                            "metadata": {"format": "docx", "table_index": t_idx, "row": r_idx, "col": c_idx},
                            "object": para
                        }
                        processed += 1
                        metrics.add_segment_duration(time.time() - seg_start)
                        self.update_progress(
                            processed,
                            total_elements,
                            start_time,
                            progress_ui=progress_ui,
                            label_ui=label_ui,
                            progress_callback=progress_callback,
                        )

        out_stream = BytesIO()
        doc.save(out_stream)
        out_stream.seek(0)
        self.output_stream = out_stream
        tokens = self.calculate_tokens(text_accum)
        metrics.finish_file(
            file_type="docx",
            segment_count=processed,
            duration_seconds=time.time() - start_time,
        )
        return out_stream, processed, tokens, text_accum, self.segment_map

    # ------------------
    # PROCESSING PPTX
    # ------------------
    def process_pptx(
        self,
        input_stream,
        target_language,
        progress_ui=None,
        label_ui=None,
        do_translate=True,
        font_size=None,
        autofit=False,
        correlation_id: str | None = None,
        file_metrics: TranslationMetrics | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        metrics = file_metrics or self.metrics
        self.reset_cancel()
        metrics.start_file(file_type="pptx", correlation_id=correlation_id)
        prs = Presentation(input_stream)
        self.current_file_type = 'pptx'
        self.current_presentation = prs
        self.pdf_overlay_ocg = None

        total_elements = sum(len(slide.shapes) for slide in prs.slides)
        processed = 0
        text_accum = ""
        start_time = time.time()

        for s_idx, slide in enumerate(prs.slides):
            for sh_idx, shape in enumerate(slide.shapes):
                original_text = self._get_shape_text(shape).strip()
                if not original_text or self.cancel_requested:
                    continue

                if do_translate:
                    seg_start = time.time()
                    new_text = self._translate_shape(shape, target_language, font_size, autofit, correlation_id, metrics)
                    metrics.add_segment_duration(time.time() - seg_start)
                else:
                    new_text = original_text

                # record in segment_map
                seg_id = self.generate_segment_id()
                self.segment_map[seg_id] = {
                    "type": "pptx_shape",
                    "location": f"pptx:slide:{s_idx}:shape:{sh_idx}",
                    "original": original_text,
                    "translated": new_text,
                    "metadata": {"format": "pptx", "slide": s_idx, "shape": sh_idx},
                    "object": shape
                }

                text_accum += new_text + "\n"
                processed += 1
                self.update_progress(
                    processed,
                    total_elements,
                    start_time,
                    progress_ui=progress_ui,
                    label_ui=label_ui,
                    progress_callback=progress_callback,
                )

        # output stream
        out_stream = BytesIO()
        prs.save(out_stream)
        out_stream.seek(0)
        self.output_stream = out_stream
        tokens = self.calculate_tokens(text_accum)
        metrics.finish_file(
            file_type="pptx",
            segment_count=processed,
            duration_seconds=time.time() - start_time,
        )
        return out_stream, processed, tokens, text_accum, self.segment_map

    def _translate_shape(
        self,
        shape,
        target_language,
        font_size=None,
        autofit=False,
        correlation_id: str | None = None,
        file_metrics: TranslationMetrics | None = None,
    ):
        def apply_formatting(tf):
            # 1) Set every paragraph & run to the user’s max size
            if font_size:
                for p in tf.paragraphs:
                    p.font.size = Pt(font_size)
                    for run in p.runs:
                        run.font.size = Pt(font_size)
            # 2) Let python-pptx shrink to fit if desired
            if autofit and hasattr(tf, "fit_text"):
                try:
                    tf.fit_text(max_size=font_size or 18)
                except KeyError as e:
                    logging.warning(f"[Backend] fit_text skipped for font {e}: metrics not found")
        result = ""

        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            for row in shape.table.rows:
                for cell in row.cells:
                    tf = cell.text_frame
                    if not tf: continue
                    text = tf.text.strip()
                    new_text = self.translate_text(
                        text,
                        target_language,
                        correlation_id=correlation_id,
                        file_metrics=file_metrics,
                    )
                    tf.text = new_text
                    apply_formatting(tf)
                    result += new_text + "\n"

        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub in shape.shapes:
                result += self._translate_shape(
                    sub,
                    target_language,
                    font_size,
                    autofit,
                    correlation_id=correlation_id,
                    file_metrics=file_metrics,
                ) + "\n"

        elif hasattr(shape, "text_frame") and shape.text_frame:
            tf = shape.text_frame
            original = tf.text.strip()
            new_text = self.translate_text(
                original,
                target_language,
                correlation_id=correlation_id,
                file_metrics=file_metrics,
            )
            tf.text = new_text
            apply_formatting(tf)
            result += new_text + "\n"

        return result.strip()
    
    def _get_shape_text(self, shape):
            """
            Recursively extract all text from a pptx shape (table, group, or text_frame).
            """
            from pptx.enum.shapes import MSO_SHAPE_TYPE

            text = ""
            # Tables
            if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                for row in shape.table.rows:
                    for cell in row.cells:
                        if cell.text_frame:
                            text += cell.text_frame.text + "\n"
            # Groups
            elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                for sub in shape.shapes:
                    text += self._get_shape_text(sub) + "\n"
            # Simple text frames
            elif hasattr(shape, "text_frame") and shape.text_frame:
                text += shape.text_frame.text + "\n"

            return text.strip()

    # ------------------
    # PROCESSING PDF
    # ------------------
    def process_pdf(
        self,
        input_stream,
        target_language,
        progress_ui=None,
        label_ui=None,
        do_translate=True,
        correlation_id: str | None = None,
        file_metrics: TranslationMetrics | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        metrics = file_metrics or self.metrics
        metrics.start_file(file_type="pdf", correlation_id=correlation_id)
        logging.info("[PDF] Opening document for translation")
        doc = fitz.open(stream=input_stream, filetype="pdf")
        self.current_file_type = 'pdf'
        self.current_pdf = doc
        self.pdf_overlay_ocg = None

        # 1) Build page_blocks with proper text extraction
        total_blocks = 0
        page_blocks = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            page_text_blocks = []
            for block in blocks:
                if block["type"] != 0:
                    continue
                # assemble text from spans
                text_accum = ""
                for line in block["lines"]:
                    line_txt = ""
                    for span in line["spans"]:
                        txt = span["text"]
                        if txt.strip() and self.is_meaningful_text(txt):
                            line_txt += txt + " "
                    if line_txt:
                        text_accum += line_txt.strip() + "\n"
                final_text = text_accum.strip()
                if final_text:
                    page_text_blocks.append({
                        "bbox": fitz.Rect(block["bbox"]),
                        "text": final_text
                    })
            page_blocks.append(page_text_blocks)
            total_blocks += len(page_text_blocks)
        logging.info(f"[PDF] Detected {total_blocks} text blocks across {len(doc)} pages")

        # 2) Prepare for translation overlays
        processed = 0
        start_time = time.time()
        self.pdf_overlay_ocg = doc.add_ocg("Translated", on=True)

        # 3) Translate & redraw each block
        for p_idx, (page, blocks) in enumerate(zip(doc, page_blocks), start=1):
            logging.info(f"[PDF] Page {p_idx}/{len(doc)}: {len(blocks)} blocks")
            for blk in blocks:
                bbox = blk["bbox"]
                original = blk["text"]
                seg_id = self.generate_segment_id()
                self.segment_map[seg_id] = {
                    "type": "pdf_block",
                    "page_idx": p_idx-1,
                    "bbox": bbox,
                    "original": original,
                    "translated": None
                }
                logging.debug(f"[PDF] Registered segment {seg_id[:8]} at {bbox}")

                seg_start = time.time()
                new_text = (
                    original
                    if not do_translate
                    else self.translate_text(
                        original,
                        target_language,
                        correlation_id=correlation_id,
                        file_metrics=metrics,
                    )
                )
                self.segment_map[seg_id]["translated"] = new_text

                last_css = self._render_pdf_block(page, bbox, new_text)
                self.segment_map[seg_id]["last_css"] = last_css
                logging.debug(f"[PDF] Translated segment {seg_id[:8]}")

                processed += 1
                metrics.add_segment_duration(time.time() - seg_start)
                self.update_progress(
                    processed,
                    total_blocks,
                    start_time,
                    progress_ui=progress_ui,
                    label_ui=label_ui,
                    progress_callback=progress_callback,
                )

        # 4) Finalize, subset fonts & save
        out_stream = BytesIO()
        logging.info("[PDF] Subsetting fonts and saving output")
        doc.subset_fonts()
        doc.ez_save(out_stream)
        out_stream.seek(0)
        self.output_stream = out_stream

        tokens = self.calculate_tokens("")  # or track actual text if desired
        logging.info(f"[PDF] Done – {processed}/{total_blocks} blocks processed, tokens={tokens}")
        metrics.finish_file(
            file_type="pdf",
            segment_count=processed,
            duration_seconds=time.time() - start_time,
        )
        return out_stream, processed, tokens, "", self.segment_map

    # ------------------
    # PROGRESS
    # ------------------
    def update_progress(
        self,
        current,
        total,
        start_time,
        progress_ui=None,
        label_ui=None,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        elapsed = time.time() - start_time
        avg = elapsed / current if current else 0
        remaining = total - current
        progress_value = (current / total) * 100 if total else 0
        label_text = f"Processing {current}/{total} (≈ {int(avg * remaining)}s remaining)"
        if progress_ui is not None:
            progress_ui.set_value(progress_value)
        if label_ui is not None:
            label_ui.text = label_text
        if progress_callback is not None:
            progress_callback(progress_value, label_text)

    # ------------------
    # UPDATE SEGMENT
    # ------------------
    def update_segment(self, segment_id, new_text, target_language, instructions=None, regenerate=True):
        if segment_id not in self.segment_map:
            raise ValueError(f"Segment ID {segment_id} not found.")

        seg = self.segment_map[segment_id]
        new_text = new_text.replace('\t', ' ').strip()
        if instructions:
            updated = self.translate_text_with_instructions(new_text, target_language, instructions)
        else:
            updated = new_text

        seg["translated"] = updated
        seg["original"] = new_text
        seg_type = seg["type"]
        if seg_type in ["paragraph", "table_cell"]:
            if "object" in seg:
                seg["object"].text = updated
        elif seg_type == "pptx_shape":
            if "object" in seg and hasattr(seg["object"], "text_frame") and seg["object"].text_frame:
                seg["object"].text_frame.text = updated
        elif seg_type == "pdf_block":
            if not self.current_pdf:
                raise ValueError("No active PDF document for update.")
            page = self.current_pdf[seg["page_idx"]]
            last_css = self._render_pdf_block(page, seg["bbox"], updated)
            seg["last_css"] = last_css

        logging.info(f"[Backend] Updated segment {segment_id} with new translation length {len(updated)}")
        if regenerate:
            self.regenerate_output_stream()
        return updated

    # ------------------
    # ROUTING
    # ------------------
    def translate_file(
        self,
        input_stream,
        file_extension,
        target_language,
        progress_ui=None,
        label_ui=None,
        processed=False,
        font_size=None,
        autofit=False,
        correlation_id: str | None = None,
        file_metrics: TranslationMetrics | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        self.reset_cancel()
        self.current_target_language = target_language
        metrics = file_metrics or self.metrics
        _log_event(
            "translation.file_started",
            correlation_id=correlation_id,
            file_extension=file_extension,
            processed=processed,
            target_language=target_language,
        )
        ext = file_extension.lower()
        if ext == "docx":
            result = self.process_docx(
                input_stream,
                target_language,
                progress_ui,
                label_ui,
                do_translate=not processed,
                correlation_id=correlation_id,
                file_metrics=metrics,
                progress_callback=progress_callback,
            )
        elif ext == "pptx":
            result = self.process_pptx(
                input_stream,
                target_language,
                progress_ui,
                label_ui,
                do_translate=not processed,
                font_size=font_size,
                autofit=autofit,
                correlation_id=correlation_id,
                file_metrics=metrics,
                progress_callback=progress_callback,
            )
        elif ext == "pdf":
            result = self.process_pdf(
                input_stream,
                target_language,
                progress_ui,
                label_ui,
                do_translate=not processed,
                correlation_id=correlation_id,
                file_metrics=metrics,
                progress_callback=progress_callback,
            )
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        _log_event(
            "translation.file_finished",
            correlation_id=correlation_id,
            metrics=metrics.snapshot(),
        )
        return result

    def record_feedback(self, *, approved: bool, original: str, translated: str) -> bool:
        """
        Append a JSONL record for approved segments,
        using the language the user originally picked.
        """
        if not isinstance(approved, bool):
            raise TypeError("approved must be a bool.")
        if not isinstance(original, str):
            raise TypeError("original must be a string.")
        if not isinstance(translated, str):
            raise TypeError("translated must be a string.")

        original = original.strip()
        translated = translated.strip()

        if approved and (not original or not translated):
            raise ValueError("original and translated must be non-empty when approved is True.")

        if not approved:
            return False

        # Grab the language the user requested at the start of translate_file
        lang = getattr(self, "current_target_language", "unknown")

        record = {
            "language":   lang,
            "prompt":     f"Translate to {lang}:\n\n{original}",
            "completion": f" {translated}"
        }

        out_dir = os.getenv("FEEDBACK_DIR", ".")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "trl_finetune_data.jsonl")

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
