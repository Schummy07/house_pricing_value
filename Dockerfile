### Stage 1: Build dependencies with Poetry
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy only dependency manifest for caching
COPY pyproject.toml poetry.lock ./

# Configure Poetry to install into the system environment
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev


### Stage 2: Build runtime image
FROM python:3.9-slim

# Install runtime system deps (if needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy application code
COPY . /app

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
# (Optionally override API_KEY at runtime)
# ENV API_KEY=your_api_key_here

# Launch the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
