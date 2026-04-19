# Use the official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirement files and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (including index_output and static)
COPY . .

# Expose the API port
EXPOSE 8000

# Start Uvicorn
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
