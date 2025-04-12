# Use a specific NVIDIA CUDA base image compatible with host driver (trying CUDA 12.1)
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set environment variables to make CUDA libs discoverable
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies 
# Need Python prerequisites, git, and parsing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for newer Python versions and install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    # python3-pip \ # Removed apt version of pip
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
# Bootstrap/Upgrade pip using ensurepip for Python 3.12
RUN python3.12 -m ensurepip --upgrade

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
# Activate the virtual environment by adding it to the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip within the venv and install requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download the spaCy model using the venv's python
RUN python -m spacy download en_core_web_sm

# Copy the rest of the backend source code
COPY ./src /app/src

# Expose the port
EXPOSE 8000

# Define the command to run the application (using the venv python)
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# --- DIAGNOSTIC CMD: Run simple HTTP server instead of Uvicorn --- 
# CMD ["python", "-m", "http.server", "8000", "--bind", "0.0.0.0"] 