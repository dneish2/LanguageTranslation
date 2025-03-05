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
import openai
import dotenv
import tiktoken

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
        # Clear the text in the underlying doc object if docx/pptx
        if seg_type in ["paragraph", "table_cell"]:
            if "object" in seg:
                seg["object"].text = ""
        elif seg_type == "pptx_shape":
            if "object" in seg and hasattr(seg["object"], "text_frame"):
                seg["object"].text_frame.text = ""
        del self.segment_map[segment_id]
        logging.info(f"[Backend] Deleted segment {segment_id}")
        self.regenerate_output_stream()

    # ------------------
    # PROCESSING DOCX
    # ------------------
    def process_docx(self, input_stream, target_language, progress_ui, label_ui, do_translate=True):
        from docx import Document
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
    def process_pptx(self, input_stream, target_language, progress_ui, label_ui, do_translate=True):
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

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
                # Gather original text
                original_text = self._get_shape_text(shape).strip()
                if not original_text:
                    continue
                if self.cancel_requested:
                    break

                if do_translate:
                    # We do a shape-level translation function
                    new_text = self._translate_shape(shape, target_language)
                else:
                    # No re-translation: just keep original
                    new_text = original_text
                text_accum += new_text + "\n"

                seg_id = self.generate_segment_id()
                self.segment_map[seg_id] = {
                    "type": "pptx_shape",
                    "location": f"pptx:slide:{s_idx}:shape:{sh_idx}",
                    "original": original_text,
                    "translated": new_text,
                    "metadata": {"format": "pptx", "slide": s_idx, "shape": sh_idx},
                    "object": shape
                }
                processed += 1
                self.update_progress(processed, total_elements, start_time, progress_ui, label_ui)

        out_stream = BytesIO()
        prs.save(out_stream)
        out_stream.seek(0)
        self.output_stream = out_stream
        tokens = self.calculate_tokens(text_accum)
        return out_stream, processed, tokens, text_accum, self.segment_map

    def _translate_shape(self, shape, target_language):
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        result = ""
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            for row in shape.table.rows:
                for cell in row.cells:
                    if cell.text_frame:
                        text = cell.text_frame.text.strip()
                        if text:
                            new_text = self.translate_text(text, target_language)
                            cell.text_frame.text = new_text
                            result += new_text + "\n"
        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub in shape.shapes:
                result += self._translate_shape(sub, target_language) + "\n"
        elif hasattr(shape, "text_frame") and shape.text_frame:
            text = shape.text_frame.text.strip()
            if text:
                new_text = self.translate_text(text, target_language)
                shape.text_frame.text = new_text
                result += new_text + "\n"
        return result.strip()

    def _get_shape_text(self, shape):
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        text = ""
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            for row in shape.table.rows:
                for cell in row.cells:
                    if cell.text_frame:
                        text += cell.text_frame.text + "\n"
        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub in shape.shapes:
                text += self._get_shape_text(sub) + "\n"
        elif hasattr(shape, "text_frame") and shape.text_frame:
            text += shape.text_frame.text + "\n"
        return text.strip()

    # ------------------
    # PROCESSING PDF
    # ------------------
    def process_pdf(self, input_stream, target_language, progress_ui, label_ui, do_translate=True):
        import fitz
        self.reset_cancel()

        try:
            doc = fitz.open(stream=input_stream, filetype="pdf")
        except Exception as e:
            logging.error(f"[Backend] Failed to open PDF: {e}")
            raise e

        self.current_file_type = 'pdf'
        self.current_pdf = doc
        total_pages = len(doc)
        processed_pages = 0
        text_accum = ""
        start_time = time.time()

        for p_idx in range(total_pages):
            if self.cancel_requested:
                break
            page = doc.load_page(p_idx)
            text_dict = page.get_text("dict")

            for block in text_dict["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        txt = span["text"]
                        if txt.strip() and self.is_meaningful_text(txt):
                            line_text += txt + " "
                    line_text = line_text.strip()
                    if line_text:
                        if do_translate:
                            new_line = self.translate_text(line_text, target_language)
                        else:
                            new_line = line_text
                        text_accum += new_line + "\n"
                        seg_id = self.generate_segment_id()
                        self.segment_map[seg_id] = {
                            "type": "pdf_line",
                            "location": f"pdf:page:{p_idx}",
                            "original": line_text,
                            "translated": new_line,
                            "metadata": {"format": "pdf", "page": p_idx}
                        }
                        page.insert_text((50, 50), new_line, overlay=True)

            processed_pages += 1
            self.update_progress(processed_pages, total_pages, start_time, progress_ui, label_ui)

        out_stream = BytesIO()
        try:
            doc.save(out_stream, garbage=4)
            out_stream.seek(0)
        except Exception as e:
            logging.error(f"[Backend] Failed to save PDF: {e}")
            raise e

        self.output_stream = out_stream
        tokens = self.calculate_tokens(text_accum)
        return out_stream, processed_pages, tokens, text_accum, self.segment_map

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

        seg["translated"] = updated
        seg["original"] = new_text
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
    def translate_file(self, input_stream, file_extension, target_language, progress_ui, label_ui, processed=False):
        self.reset_cancel()
        ext = file_extension.lower()
        if ext == "docx":
            return self.process_docx(input_stream, target_language, progress_ui, label_ui, do_translate=not processed)
        elif ext == "pptx":
            return self.process_pptx(input_stream, target_language, progress_ui, label_ui, do_translate=not processed)
        elif ext == "pdf":
            return self.process_pdf(input_stream, target_language, progress_ui, label_ui, do_translate=not processed)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")