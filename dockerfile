# Base image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN python -m ensurepip --upgrade

# Clone the repository
RUN git clone https://github.com/deadsmash07/Few-Shot-KWS.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the protonets package
RUN python setup.py develop

# Expose the port (if needed for specific tasks)
EXPOSE 8000

# Default command
CMD ["bash"]
