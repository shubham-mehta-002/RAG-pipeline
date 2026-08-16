"""PDF parser — extracts text and images from a PDF file.

For each page:
  - Text-based pages: text is extracted directly with PyMuPDF.
  - Scanned pages (no text): the page is rendered as an image and sent
    to GPT-4o Vision for OCR. Falls back to Tesseract if the API call fails.
  - Embedded images on any page are also extracted and sent to GPT-4o
    to describe / extract any text they contain.

OCR backend priority: GPT-4o → Tesseract fallback
"""

import base64
import io
import os
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from openai import OpenAI
from PIL import Image

from rag.ingestion.models import Document, Element
from rag.ingestion.utils import make_doc_id


# --- OCR helpers ---

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _image_to_b64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _ocr_gpt4v(img: Image.Image, prompt: str = "Extract all text from this image exactly as it appears.") -> str | None:
    """Send an image to GPT-4o and return the extracted text.

    Returns None if the API call fails, so the caller can fall back to Tesseract.
    """
    try:
        b64 = _image_to_b64(img)
        response = _client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def _ocr(img: Image.Image, prompt: str = "Extract all text from this image exactly as it appears.") -> str:
    """OCR an image — tries GPT-4o first, falls back to Tesseract."""
    result = _ocr_gpt4v(img, prompt)
    if result is not None:
        return result
    # Tesseract fallback
    return pytesseract.image_to_string(img).strip()


# --- PDF helpers ---

def _page_to_image(page: fitz.Page) -> Image.Image:
    """Render a PDF page to a PIL Image (200 DPI)."""
    pix = page.get_pixmap(dpi=200)
    return Image.open(io.BytesIO(pix.tobytes("png")))


# --- Parser ---

def parse_pdf(path: str) -> Document:
    """Parse a PDF file and return a Document with text and image elements.

    Text pages are extracted directly. Scanned pages and embedded images
    are processed via GPT-4o Vision (Tesseract fallback on API failure).
    """
    abs_path = str(Path(path).resolve())
    doc_id = make_doc_id(abs_path)

    document = Document(
        id=doc_id,
        source=abs_path,
        mime_type="application/pdf",
        title=Path(path).stem,
    )

    pdf = fitz.open(abs_path)
    element_index = 0

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text().strip()

        if text:
            # Normal text page — use extracted text directly
            document.elements.append(Element(
                id=f"{doc_id}-p{page_num}-e{element_index}",
                type="text",
                content=text,
                page=page_num,
            ))
            element_index += 1
        else:
            # Scanned page — render and OCR
            page_img = _page_to_image(page)
            ocr_text = _ocr(page_img)

            if ocr_text:
                document.elements.append(Element(
                    id=f"{doc_id}-p{page_num}-e{element_index}",
                    type="text",
                    content=ocr_text,
                    page=page_num,
                    metadata={"ocr": True}, # flag it as OCR'd
                ))
                element_index += 1

        # Extract embedded images from every page
        for img_index, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            base_image = pdf.extract_image(xref)
            img = Image.open(io.BytesIO(base_image["image"]))

            # Ask GPT-4o to describe the image and extract any text
            description = _ocr(
                img,
                prompt="Describe this image and extract any text or labels visible in it.",
            )

            document.elements.append(Element(
                id=f"{doc_id}-p{page_num}-img{img_index}",
                type="image",
                content=None,
                page=page_num,
                metadata={
                    "ocr_text": description,
                    "width": img.width,
                    "height": img.height,
                },
            ))
            element_index += 1

    pdf.close()
    return document
