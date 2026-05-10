FROM python:3.12-slim

WORKDIR /app

# System deps for aiortc (WebRTC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev libopus0 libvpx7 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --group dev --all-extras --no-extra gstreamer --no-extra local

EXPOSE 8080

CMD ["uv", "run", "python", "examples/peercot-voice/bot.py", "-t", "webrtc", "--host", "0.0.0.0", "--port", "8080"]
