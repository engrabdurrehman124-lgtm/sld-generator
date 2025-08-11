# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install system dependencies, including mdbtools and matplotlib dependencies
RUN apt-get update && apt-get install -y \
    mdbtools \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create uploads directory
RUN mkdir -p /app/Uploads

# Expose the port the app runs on
EXPOSE 10000

# Command to run the Flask app
CMD ["python", "app.py"]
