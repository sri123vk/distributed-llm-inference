FROM nvidia/cuda:12.4.1-base-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY telemetry/agent.py /app/agent.py
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=utility
CMD ["python3", "/app/agent.py"]
