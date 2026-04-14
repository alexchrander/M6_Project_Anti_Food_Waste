# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# curl: needed to download supercronic
# supercronic: Docker-friendly cron daemon used by the scheduler service
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
 && ARCH=$(uname -m) \
 && if [ "$ARCH" = "x86_64" ]; then SC_ARCH=amd64; \
    elif [ "$ARCH" = "aarch64" ]; then SC_ARCH=arm64; \
    else echo "Unsupported arch: $ARCH" && exit 1; fi \
 && curl -fsSLo /usr/local/bin/supercronic \
      "https://github.com/aptible/supercronic/releases/download/v0.2.29/supercronic-linux-${SC_ARCH}" \
 && chmod +x /usr/local/bin/supercronic \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# ── Runtime directories (persisted via Docker volumes) ────────────────────────
RUN mkdir -p data/dataset data/features data/predictions models outputs mlruns
