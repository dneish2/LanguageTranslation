import logging
import os
import string
import time
import uuid
import json
from html import escape
from io import BytesIO
from typing import Tuple
from typing import Optional

import dotenv
import fitz  # PyMuPDF
import openai
import tiktoken
from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Pt

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

WHITE = (1, 1, 1)  # RGB white for PDF overwrite


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

    def reset_cancel(self) -> None:
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True
        logging.info("[Backend] Cancel requested.")

    def generate_segment_id(self) -> str:
        return str(uuid.uuid4())

    # ───────────────────────────── GPT CORE ────────────────────────────── #
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate free-form text via GPT."""
        text = text.replace("\t", " ").strip()
        if not text:
            return text

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
            completion = self.client.chat.completions.create(model="gpt-4.1-nano", messages=messages, max_tokens=4000)
            result = completion.choices[0].message.content.strip() or text
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
            completion = self.client.chat.completions.create(model="gpt-4.1-nano", messages=messages, max_tokens=4000)
            result = completion.choices[0].message.content.strip() or original_text
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
    def process_docx(self, input_stream, target_language, progress_ui, label_ui, do_translate=True):
        self.reset_cancel()
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
            new_text = self.translate_text(original, target_language) if do_translate else original
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
            self.update_progress(processed, total_elements, start_time, progress_ui, label_ui)

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
                        new_text = self.translate_text(original, target_language) if do_translate else original
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
                        self.update_progress(processed, total_elements, start_time, progress_ui, label_ui)

        out_stream = BytesIO()
        doc.save(out_stream)
        out_stream.seek(0)
        self.output_stream = out_stream
        tokens = self.calculate_tokens(text_accum)
        return out_stream, processed, tokens, text_accum, self.segment_map

    # ------------------
    # PROCESSING PPTX
    # ------------------
    def process_pptx(
        self,
        input_stream,
        target_language,
        progress_ui,
        label_ui,
        do_translate=True,
        font_size=None,
        autofit=False
    ):
    
        self.reset_cancel()
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
                    new_text = self._translate_shape(shape, target_language, font_size, autofit)
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
                self.update_progress(processed, total_elements, start_time, progress_ui, label_ui)

        # output stream
        out_stream = BytesIO()
        prs.save(out_stream)
        out_stream.seek(0)
        self.output_stream = out_stream
        tokens = self.calculate_tokens(text_accum)
        return out_stream, processed, tokens, text_accum, self.segment_map

    def _translate_shape(self, shape, target_language, font_size=None, autofit=False):
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
                    new_text = self.translate_text(text, target_language)
                    tf.text = new_text
                    apply_formatting(tf)
                    result += new_text + "\n"

        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub in shape.shapes:
                result += self._translate_shape(sub, target_language, font_size, autofit) + "\n"

        elif hasattr(shape, "text_frame") and shape.text_frame:
            tf = shape.text_frame
            original = tf.text.strip()
            new_text = self.translate_text(original, target_language)
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
    def process_pdf(self, input_stream, target_language, progress_ui, label_ui, do_translate=True):
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

                new_text = original if not do_translate else self.translate_text(original, target_language)
                self.segment_map[seg_id]["translated"] = new_text

                last_css = self._render_pdf_block(page, bbox, new_text)
                self.segment_map[seg_id]["last_css"] = last_css
                logging.debug(f"[PDF] Translated segment {seg_id[:8]}")

                processed += 1
                self.update_progress(processed, total_blocks, start_time, progress_ui, label_ui)

        # 4) Finalize, subset fonts & save
        out_stream = BytesIO()
        logging.info("[PDF] Subsetting fonts and saving output")
        doc.subset_fonts()
        doc.ez_save(out_stream)
        out_stream.seek(0)
        self.output_stream = out_stream

        tokens = self.calculate_tokens("")  # or track actual text if desired
        logging.info(f"[PDF] Done – {processed}/{total_blocks} blocks processed, tokens={tokens}")
        return out_stream, processed, tokens, "", self.segment_map

    # ------------------
    # PROGRESS
    # ------------------
    def update_progress(self, current, total, start_time, progress_ui, label_ui):
        elapsed = time.time() - start_time
        avg = elapsed / current if current else 0
        remaining = total - current
        progress_value = (current / total) * 100 if total else 0
        progress_ui.set_value(progress_value)
        label_ui.text = f"Processing {current}/{total} (≈ {int(avg * remaining)}s remaining)"

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
        progress_ui,
        label_ui,
        processed=False,
        font_size=None,
        autofit=False
    ):
        self.reset_cancel()
        self.current_target_language = target_language
        ext = file_extension.lower()
        if ext == "docx":
            return self.process_docx(input_stream, target_language, progress_ui, label_ui, do_translate=not processed)
        elif ext == "pptx":
            return self.process_pptx(
                input_stream,
                target_language,
                progress_ui,
                label_ui,
                do_translate=not processed,
                font_size=font_size,
                autofit=autofit
            )
        elif ext == "pdf":
            return self.process_pdf(input_stream, target_language, progress_ui, label_ui, do_translate=not processed)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def record_feedback(self, approved: bool, original: str, translated: str):
        """
        Append a JSONL record for approved segments,
        using the language the user originally picked.
        """
        if not approved:
            return

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