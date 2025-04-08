# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt using uv
# Using --system to install in the main environment since it's a container
# Using --no-cache to reduce image size
RUN uv pip install --system --no-cache -r /app/requirements.txt

# Download the spaCy model using uv
# Note: This URL might need updating if a newer spaCy/model version is used
RUN uv pip install --system --no-cache https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Copy the rest of the backend source code into the container
COPY ./src /app/src

# Expose the port the app runs on
# Note: The actual port mapping is done in docker-compose.yml
EXPOSE 8000

# Define the command to run the application
# The host 0.0.0.0 makes it accessible from outside the container
# Port is hardcoded here but can be overridden by env var in docker-compose
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 