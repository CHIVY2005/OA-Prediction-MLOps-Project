FROM python:3.9-slim

WORKDIR /app

# Install dependencies for OpenCV and others if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (super lightweight for Hugging Face build)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the API port (7860 for Hugging Face Spaces)
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
