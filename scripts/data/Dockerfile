FROM python:3.13

WORKDIR /app

# From yt_bot_detector context
COPY requirements.txt .
COPY requirements.data.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements.data.txt

COPY ./scripts/data/ ./scripts/data
COPY ./src ./src

ENTRYPOINT ["python", "-m", "scripts.data.api"]

