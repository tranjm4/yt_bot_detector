FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements.app.txt

# Copy application code
COPY app/ ./app/
COPY public/ ./public/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
