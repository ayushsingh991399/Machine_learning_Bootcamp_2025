# Use the base image with the pre-trained model
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install uv (the modern dependency manager)
RUN pip install uv

# Sync dependencies (this installs FastAPI, Uvicorn, etc.)
RUN uv sync

# Copy your FastAPI app
COPY app.py .
COPY pipeline_v1.bin .
# Expose port 8000
EXPOSE 8000

# Run the FastAPI server
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
