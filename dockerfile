FROM python:3.9-slim

WORKDIR /app

COPY heart-disease.csv . 
COPY Heart-Disease.jpg . 
COPY HD_st.py .
# installer les dependances 
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas==1.5.3 \
    streamlit==1.23.1 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    statsmodels==0.14.0

# Exposer le port utilisé par Streamlit
CMD ["streamlit", "run", "HD_st.py"]