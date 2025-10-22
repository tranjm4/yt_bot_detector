FROM python:3.13

WORKDIR /app

COPY requirements.txt .
COPY requirements.data.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements.data.txt

COPY scripts/processing/run_pipeline.py ./scripts/processing/run_pipeline.py
COPY config/pipelines/ ./config/pipelines
COPY src ./src


ENTRYPOINT ["python", "-m", "scripts.processing.run_pipeline"]