FROM python:3.13

WORKDIR /app

COPY requirements.txt .
COPY requirements.model.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements.model.txt

COPY scripts/models/ ./scripts/models
COPY /src ./src

ENTRYPOINT ["python", "-m", "scripts.models.fit_model"]
CMD []