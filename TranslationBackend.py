import os
import logging
import uuid
import string
import fitz  # PyMuPDF
import time
from io import BytesIO
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from docx import Document
from pptx.util import Pt
import openai
import dotenv
import tiktoken
import json, os, logging

dotenv.load_dotenv()  # Load environment variables

logging.basicConfig(level=logging.INFO)

WHITE = (1, 1, 1)  # for PDF overwriting

class TranslationBackend:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.cancel_requested = False

        self.segment_map = {}       # segment_id -> {type, location, original, translated, metadata, object}
        self.current_file_type = None
        self.current_document = None
        self.current_presentation = None
        self.current_pdf = None
        self.output_stream = None

    def reset_cancel(self):
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True
        logging.info("[Backend] Cancel requested.")

    def generate_segment_id(self):
        return str(uuid.uuid4())

    # ------------------
    # GPT CORE
    # ------------------
    def translate_text(self, text, target_language):
        text = text.replace('\t', ' ').strip()
        if not text:
            return text
        prompt = (
            f"Translate the following text to {target_language}, preserving meaning and context. "
            f"Do not translate personal names or trademarked terms. If it's an email or URL, keep it unchanged.\n\n"
            f"Text:\n{text}"
        )
        messages = [
            {"role": "system", "content": f"You are a helpful assistant translating to {target_language}."},
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=4000
            )
            result = completion.choices[0].message.content.strip()
            if not result:
                result = text
            logging.info(f"[Backend] Translated text from len={len(text)} to len={len(result)}")
            return result
        except Exception as e:
            logging.error(f"[Backend] translate_text error: {e}", exc_info=True)
            return text

    def translate_text_with_instructions(self, original_text, target_language, instructions):
        original_text = original_text.replace('\t', ' ').strip()
        if not original_text:
            return original_text
        prompt = (
            f"Please refine the following translation according to these instructions. "
            f"Ensure that any requested changes – including changing the language – are fully applied.\n\n"
            f"Instructions: {instructions}\n\n"
            f"Original text: {original_text}\n\n"
            f"Final translation:"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant refining translations."},
            {"role": "user", "content": prompt},
        ]
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=4000
            )
            result = completion.choices[0].message.content.strip()
            if not result:
                result = original_text
            logging.info(f"[Backend] Refined translation with instructions; result length: {len(result)}")
            return result
        except Exception as e:
            logging.error(f"[Backend] refine translation error: {e}", exc_info=True)
            return original_text

    def calculate_tokens(self, total_text):
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(total_text))

    # ------------------
    # PDF UTILS
    # ------------------
    def is_meaningful_text(self, text):
        normalized = text.strip().lower().replace(' ', '')
        skip = {'tm', '™', '©', '®'}
        if normalized in skip:
            return False
        stripped = text.translate(str.maketrans('', '', string.punctuation + "™©®")).strip()
        return any(ch.isalnum() for ch in stripped)

    # ------------------
    # REGENERATE OUTPUT
    # ------------------
    def regenerate_output_stream(self):
        if self.current_file_type == 'docx' and self.current_document:
            out_stream = BytesIO()
            self.current_document.save(out_stream)
            out_stream.seek(0)
            self.output_stream = out_stream
        elif self.current_file_type == 'pptx' and self.current_presentation:
            out_stream = BytesIO()
            self.current_presentation.save(out_stream)
            out_stream.seek(0)
            self.output_stream = out_stream
        elif self.current_file_type == 'pdf' and self.current_pdf:
            out_stream = BytesIO()
            self.current_pdf.save(out_stream, garbage=4)
            out_stream.seek(0)
            self.output_stream = out_stream
        return self.output_stream

    # ------------------
    # SEGMENT DELETION
    # ------------------
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
                page = self.current_pdf[ seg["page_idx"] ]
                page.draw_rect(seg["bbox"], color=None, fill=WHITE)
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
        ocg = doc.add_ocg("Translated", on=True)

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

                page.draw_rect(bbox, color=None, fill=WHITE, oc=ocg)
                page.insert_htmlbox(
                    bbox, new_text,
                    css=f"* {{font-size:{min(len(new_text)/50,24):.0f}pt;}}",
                    overlay=True, oc=ocg
                )
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
    def update_segment(self, segment_id, new_text, target_language, instructions=None):
        if segment_id not in self.segment_map:
            raise ValueError(f"Segment ID {segment_id} not found.")

        seg = self.segment_map[segment_id]
        new_text = new_text.replace('\t', ' ').strip()
        if instructions:
            updated = self.translate_text_with_instructions(new_text, target_language, instructions)
        else:
            updated = new_text

        # Preserve the source text in "original" and only update the
        # translated version.
        seg["translated"] = updated
        seg_type = seg["type"]
        if seg_type in ["paragraph", "table_cell"]:
            if "object" in seg:
                seg["object"].text = updated
        elif seg_type == "pptx_shape":
            if "object" in seg and hasattr(seg["object"], "text_frame") and seg["object"].text_frame:
                seg["object"].text_frame.text = updated

        logging.info(f"[Backend] Updated segment {segment_id} with new translation length {len(updated)}")
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
        
    def record_feedback(self, segment_id: str, approved: bool, original: str, translated: str):
        """
        Append a feedback record to feedback.jsonl, for fine-tuning or audit.
        """
        record = {
            "segment_id": segment_id,
            "approved": approved,
            "original": original,
            "translated": translated,
            "timestamp": time.time()
        }
        # ensure directory exists
        out_dir = os.getenv("FEEDBACK_DIR", ".")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "feedback.jsonl")

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info(f"[Backend:feedback] Recorded feedback for {segment_id[:8]} → {approved} to {path}")