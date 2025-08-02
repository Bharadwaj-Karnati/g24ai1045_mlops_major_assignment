# Use official Python base image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy everything into container
COPY . .

# Set PYTHONPATH so Python can find `src` as a package
ENV PYTHONPATH=/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "src/train.py"]
