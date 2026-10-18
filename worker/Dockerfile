FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY worker /app/worker
COPY scripts/download_model.py /app/scripts/download_model.py
RUN python scripts/download_model.py
EXPOSE 8001
CMD ["python", "-m", "worker.server"]
