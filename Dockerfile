# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- Fix: corp networks often block HTTP:80, so force HTTPS for Debian repos ---
RUN set -eux; \
  if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
  else \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
    sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; \
  fi; \
  apt-get update; \
  apt-get install -y --no-install-recommends libgomp1; \
  rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY src /app/src

RUN pip install --no-cache-dir -e .

ENV UCPP_ARTIFACTS_DIR=/app/artifacts
ENV UCPP_HOST=0.0.0.0
ENV UCPP_PORT=8000

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "ucpp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]