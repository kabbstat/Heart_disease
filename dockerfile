FROM python:3.9-slim

WORKDIR /app

COPY heart-disease.csv . 
COPY Heart-Disease.jpg . 
COPY HD_stream.py .
COPY requirements.txt .
# installer les dependances 
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080

# Exposer le port utilisé par Streamlit
CMD ["streamlit", "run", "HD_stream.py", "--server.port=8080", "--server.headless=true"]