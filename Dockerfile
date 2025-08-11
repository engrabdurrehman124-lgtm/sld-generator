FROM python:3.10-slim

# Install system dependencies for mdbtools
RUN apt-get update && apt-get install -y \
    mdbtools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV FLASK_APP=app:app
ENV PORT=8080

# Run the Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
