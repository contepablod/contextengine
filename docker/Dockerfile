FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY context.json ./context.json
COPY uploads ./uploads

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "app.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
