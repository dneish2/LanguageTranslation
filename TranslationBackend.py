import os
import logging
import re
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

# Define color "white" for PDF overlay
WHITE = (1, 1, 1)

class TranslationBackend:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")
        self.client = openai.OpenAI(api_key=self.api_key)

    # ------------------
    # NEW PDF HELPERS
    # ------------------
    def is_meaningful_text(self, text):
        """
        Check if the text is worth translating, skipping trivial elements like TM, ©, ®, etc.
        """
        normalized_text = text.strip().lower().replace(' ', '')
        skip_texts = {'tm', '™', '©', '®'}
        tm_pattern = re.compile(r'^(tm|™|©|®)$', re.IGNORECASE)

        if normalized_text in skip_texts or tm_pattern.match(normalized_text):
            return False

        # Remove punctuation & special symbols for the check
        stripped_text = text.translate(str.maketrans('', '', string.punctuation + "™©®")).strip()
        return any(char.isalnum() for char in stripped_text)

    def int_to_rgb(self, color):
        """Convert integer color to an (r, g, b) tuple (0-1 range)."""
        r = ((color >> 16) & 0xFF) / 255.0
        g = ((color >> 8) & 0xFF) / 255.0
        b = (color & 0xFF) / 255.0
        return (r, g, b)

    def should_retain_original(self, text):
        """
        If the translated text is some sort of incomplete/error message from the model,
        revert to original text.
        """
        assistant_msgs = [
            "The text provided is incomplete and cannot be accurately translated",
            "I'm sorry, but the text you provided seems incomplete",
            "It seems that the text you provided is incomplete",
            "The text appears to be incomplete",
            "The text is incomplete",
        ]
        return any(msg in text for msg in assistant_msgs)

    def create_css(self, font_size, rgb_color):
        """Create CSS styling to match original font size and color for PDF."""
        hex_color = '#%02x%02x%02x' % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255)
        )
        return f"* {{ font-size: {font_size}pt; color: {hex_color}; }}"

    # ------------------
    # GENERIC TRANSLATION LOGIC
    # ------------------
    def translate_text(self, text, target_language):
        if not text.strip():
            return text  # Skip empty strings

        prompt = (
            f"Translate the following text to {target_language}, preserving the meaning and context. "
            f"Do not translate personal names, internationally recognized technical terms, or trademarked terms. "
            f"**If the text is an email address or a URL, do not translate or alter it**; just return the exact original text. "
            f"**Do not add any extra commentary or remarks.** "
            f"Translate everything else as best as possible, even if it is incomplete or fragmented.\n\n"
            f"Text:\n{text.strip()}"
        )

        messages = [
            {"role": "system", "content": f"You are a helpful assistant that translates text to {target_language}."},
            {"role": "user", "content": prompt},
        ]

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o", 
                messages=messages, 
                max_tokens=4000
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text  # Return original text in case of error

    def calculate_tokens(self, total_text):
        """Calculate the number of tokens used."""
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(total_text))

    # ------------------
    # DOCX + PPTX LOGIC
    # ------------------
    def process_pptx(self, input_stream, target_language, progress_ui, label_ui):
        """Process and translate a PowerPoint file."""
        prs = Presentation(input_stream)
        total_shapes = sum(len(slide.shapes) for slide in prs.slides)
        shape_count = 0
        translated_text = ""
        original_segments = []
        translated_segments = []

        start_time = time.time()
        for slide in prs.slides:
            for shape in slide.shapes:
                original_text = self._get_shape_text(shape)
                if original_text.strip():
                    new_text = self._translate_shape(shape, target_language)
                    translated_text += new_text + "\n"
                    original_segments.append(original_text)
                    translated_segments.append(new_text)
                shape_count += 1
                self.update_progress(shape_count, total_shapes, start_time, progress_ui, label_ui)

        output_stream = BytesIO()
        prs.save(output_stream)
        output_stream.seek(0)
        tokens = self.calculate_tokens(translated_text)
        return output_stream, shape_count, tokens, translated_text, original_segments, translated_segments

    def _translate_shape(self, shape, target_language):
        """Translate the text within a shape."""
        translated_text = ""
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            for row in shape.table.rows:
                for cell in row.cells:
                    if cell.text_frame:
                        cell_text = cell.text_frame.text
                        new_text = self.translate_text(cell_text, target_language)
                        cell.text_frame.text = new_text
                        translated_text += new_text + "\n"
        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub_shape in shape.shapes:
                translated_text += self._translate_shape(sub_shape, target_language) + "\n"
        elif hasattr(shape, "text_frame") and shape.text_frame:
            shape_text = shape.text_frame.text
            new_text = self.translate_text(shape_text, target_language)
            shape.text_frame.text = new_text
            translated_text += new_text + "\n"
        return translated_text.strip()

    def _get_shape_text(self, shape):
        """Extract original text from a shape."""
        text = ""
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            for row in shape.table.rows:
                for cell in row.cells:
                    if cell.text_frame:
                        text += cell.text_frame.text + "\n"
        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub_shape in shape.shapes:
                text += self._get_shape_text(sub_shape) + "\n"
        elif hasattr(shape, "text_frame") and shape.text_frame:
            text += shape.text_frame.text + "\n"
        return text.strip()

    def process_docx(self, input_stream, target_language, progress_ui, label_ui):
        """Process and translate a Word document."""
        doc = Document(input_stream)
        total_paragraphs = len(doc.paragraphs) + sum(len(table.rows)*len(table.columns) for table in doc.tables)
        paragraph_count = 0
        translated_text = ""
        original_segments = []
        translated_segments = []

        start_time = time.time()
        # Process paragraphs
        for paragraph in doc.paragraphs:
            original = paragraph.text
            new_text = self.translate_text(original, target_language)
            paragraph.text = new_text
            translated_text += new_text + "\n"
            original_segments.append(original)
            translated_segments.append(new_text)

            paragraph_count += 1
            self.update_progress(paragraph_count, total_paragraphs, start_time, progress_ui, label_ui)

        # Process tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        original = paragraph.text
                        new_text = self.translate_text(original, target_language)
                        paragraph.text = new_text
                        translated_text += new_text + "\n"
                        original_segments.append(original)
                        translated_segments.append(new_text)

                        paragraph_count += 1
                        self.update_progress(paragraph_count, total_paragraphs, start_time, progress_ui, label_ui)

        output_stream = BytesIO()
        doc.save(output_stream)
        output_stream.seek(0)
        tokens = self.calculate_tokens(translated_text)
        return output_stream, paragraph_count, tokens, translated_text, original_segments, translated_segments

    # ------------------
    # NEW PDF LOGIC
    # ------------------
    def process_pdf(self, input_stream, target_language, progress_ui, label_ui):
        """
        Process and translate a PDF file using PyMuPDF.
        Overwrites text spans with their translations.
        """
        translated_text = ""
        original_segments = []
        translated_segments = []

        try:
            doc = fitz.open(stream=input_stream, filetype="pdf")
        except Exception as e:
            logging.error(f"Failed to open input PDF: {e}")
            raise e

        ocg_xref = doc.add_ocg(target_language, on=True)

        total_pages = len(doc)
        page_count = 0
        start_time = time.time()

        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            text_dict = page.get_text("dict")

            for block in text_dict["blocks"]:
                # Only text blocks
                if block["type"] != 0:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = fitz.Rect(span["bbox"])
                        text = span["text"]

                        if not text.strip():
                            continue

                        if not self.is_meaningful_text(text):
                            continue

                        # Original text
                        processed_text = ' '.join(text.splitlines()).strip()

                        # Reuse the same GPT-based translator
                        new_text = self.translate_text(processed_text, target_language)
                        if self.should_retain_original(new_text):
                            new_text = text  # revert if the model gave an incomplete message

                        # Keep track for output
                        original_segments.append(text)
                        translated_segments.append(new_text)
                        translated_text += new_text + "\n"

                        color = span.get("color", 0)
                        rgb_color = self.int_to_rgb(color)
                        css = self.create_css(span['size'], rgb_color)

                        # Overwrite the existing text with a white rectangle
                        page.draw_rect(bbox, color=None, fill=WHITE, oc=ocg_xref, overlay=True)

                        # Insert the new translated text
                        page.insert_htmlbox(bbox, new_text, css=css, oc=ocg_xref, overlay=True)

            page_count += 1
            self.update_progress(page_count, total_pages, start_time, progress_ui, label_ui)

        output_stream = BytesIO()

        try:
            doc.save(output_stream, garbage=4)
            output_stream.seek(0)
        except Exception as e:
            logging.error(f"Failed to save output PDF: {e}")
            raise e
        finally:
            doc.close()

        tokens = self.calculate_tokens(translated_text)
        # We'll approximate count as the total number of meaningful text segments
        count = len(original_segments)
        return output_stream, count, tokens, translated_text, original_segments, translated_segments

    # ------------------
    # PROGRESS UPDATES
    # ------------------
    def update_progress(self, current, total, start_time, progress_ui, label_ui):
        """Update progress UI with the current progress."""
        elapsed_time = time.time() - start_time
        avg_time_per_element = elapsed_time / current if current else 0
        remaining_elements = total - current
        remaining_time = avg_time_per_element * remaining_elements

        progress_ui.set_value((current / total) * 100)
        label_ui.text = (
            f"Processing element {current}/{total}\n"
            f"Estimated time remaining: {int(remaining_time)} seconds"
        )

    # ------------------
    # KEY CHANGE: SUPPORT PDF
    # ------------------
    def translate_file(self, input_stream, file_extension, target_language, progress_ui, label_ui):
        """
        Translate a file based on its extension, returning:
          (output_stream, count, tokens, translated_text, original_segments, translated_segments)
        """
        file_extension = file_extension.lower()
        if file_extension == "pptx":
            return self.process_pptx(input_stream, target_language, progress_ui, label_ui)
        elif file_extension == "docx":
            return self.process_docx(input_stream, target_language, progress_ui, label_ui)
        elif file_extension == "pdf":
            # <-- Minimal change: PDF route
            return self.process_pdf(input_stream, target_language, progress_ui, label_ui)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")