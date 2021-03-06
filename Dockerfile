# temp stage
FROM python:3.9-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN export DEBIAN_FRONTEND=noninteractive && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y --no-install-recommends gcc && \
  apt-get -y clean && \
  rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# final stage
FROM python:3.9-slim

RUN useradd --create-home appuser
WORKDIR /home/appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

USER appuser

ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv

EXPOSE 8080

COPY --chown=appuser /app ./

CMD streamlit run web.py --server.port=8080 --browser.serverAddress="0.0.0.0"
