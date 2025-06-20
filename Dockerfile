# Use a slim official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Run the app
CMD ["python", "TranslationUI.py"]