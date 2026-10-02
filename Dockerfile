FROM python:3.12-slim

WORKDIR /app

# Install curl (needed for the healthcheck below) - slim images omit it by default
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy just requirements first - Docker caches this layer, so re-building
# after a code change (without changing requirements.txt) skips reinstalling
# everything, making rebuilds much faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the actual project code.
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 0.0.0.0, not localhost/127.0.0.1 - inside a container, binding to
# localhost only accepts connections from WITHIN the container itself.
# 0.0.0.0 means "accept connections from any network interface", # which is required for the host machine to reach it through the port mapping.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]