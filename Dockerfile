# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Dependencies first (better layer caching)
COPY pyproject.toml README.md ./
COPY src/atlas/__init__.py src/atlas/__init__.py
RUN pip install ".[postgres]" && pip uninstall -y atlas-ai

COPY src ./src
COPY sql ./sql
COPY knowledge_base ./knowledge_base
COPY docker/entrypoint.sh /entrypoint.sh
RUN pip install --no-deps . \
    && sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh \
    && useradd --create-home --uid 1000 atlas \
    && mkdir -p /app/artifacts /app/data /mlflow \
    && chown -R atlas /app /mlflow

USER atlas
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["atlas", "serve", "--host", "0.0.0.0", "--port", "8000"]
