FROM python:3.11-slim

# Install system dependencies required for pip packages and AI development
RUN apt-get update && apt-get install -y \
    git build-essential  \
    libpython3.11-dev libyaml-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a new user 'python'
RUN useradd --create-home --shell /bin/bash python

# Set working directory
WORKDIR /workspaces/3dl-code-ai
COPY requirements.txt .
USER python

# Copy and install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Start an interactive Bash shell by default
CMD ["bash"]