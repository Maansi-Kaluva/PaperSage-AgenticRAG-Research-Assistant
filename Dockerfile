FROM python:3.12

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend package
COPY backend/ backend/

# Documents
COPY documents/ documents/

# Streamlit config
RUN mkdir -p .streamlit
COPY config.toml .streamlit/config.toml

# Application files
COPY app.py .
COPY evaluate.py .
COPY goldens.json .
COPY sessions.json .
COPY deepeval_gpt.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]