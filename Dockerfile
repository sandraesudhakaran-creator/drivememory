FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DRIVEMEMORY_DB=/data/drivememory.db

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,mcp]" && mkdir -p /data

RUN useradd --create-home appuser && chown -R appuser /data
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "drivememory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
