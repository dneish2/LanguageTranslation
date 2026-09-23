# Use a slim official Python image
FROM python:3.11-slim

# DejaVu gives the image overlay a scalable font. Without it, image_compositor
# fell through to PIL's bitmap default in prod (/api/health reported
# font.bitmap_fallback=true), which renders accented text as tofu boxes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the tokenizer file in, so no instance downloads it while serving a
# document (TranslationBackend._token_encoding).
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

# Copy app files
COPY . .

# Run as an unprivileged user; the app only writes session storage and traces
# under /app.
RUN useradd --create-home --uid 10001 passage && chown -R passage /app
USER passage

# Expose port for Cloud Run
EXPOSE 8080

# Run the app
CMD ["python", "TranslationUI.py"]
