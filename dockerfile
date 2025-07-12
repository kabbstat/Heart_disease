FROM python:3.9-slim
WORKDIR /app

# Installer les dépendances système (ici libgomp1 pour sklearn)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'application (y compris utils.py, params.yaml, etc.)
COPY . .

# Exposer le port et lancer l'app Streamlit
EXPOSE 8080
CMD ["streamlit", "run", "HD_stream.py", "--server.port=8080", "--server.address=0.0.0.0"]
