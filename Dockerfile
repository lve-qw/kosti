FROM python:3.11.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
ARG INSTALL_ML=1
COPY pyproject.toml README.md requirements-core.lock requirements-ml.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-core.lock && \
    if [ "$INSTALL_ML" = "1" ]; then \
      pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-ml.lock; \
    fi && \
    pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 kosti
USER kosti
EXPOSE 8000
CMD ["kosti", "serve", "--host", "0.0.0.0", "--port", "8000"]
