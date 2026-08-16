FROM python:3.11-slim

# System dependencies:
# - tesseract-ocr: OCR fallback if GPT-4o API call fails
# - libmagic1: required by python-magic for MIME detection
# - poppler-utils: PDF utilities used by PyMuPDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the rest of the project
COPY . .

CMD ["python", "main.py"]
