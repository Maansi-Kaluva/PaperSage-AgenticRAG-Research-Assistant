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
COPY .streamlit/config.toml .streamlit/config.toml
COPY .streamlit/secrets.toml .streamlit/secrets.toml

# Application files
COPY app.py .
COPY evaluate.py .
COPY goldens.json .
COPY sessions.json .
COPY deepeval_gpt.py .
COPY login_wall.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]