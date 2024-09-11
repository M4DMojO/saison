# Utiliser une image Python officielle comme base
FROM python:3.10.12-slim

# Définir le répertoire de travail sur /app
WORKDIR /app

# Mettre à jour le gestionnaire de paquets et installer les dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    build-essential \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier les fichiers requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les paquets Python nécessaires
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copier le code source de l'application dans le conteneur
COPY src /app/src

# Copier les fichiers de données nécessaires
COPY data /app/data

# Copier les fichiers de modèles nécessaires
COPY models /app/models

# Exposer le port 5000 au monde extérieur
EXPOSE 5000

WORKDIR /app/src 

# Définir la variable d'environnement pour Flask
ENV FLASK_APP=app.py

# Commande d'entrée pour lancer l'application Flask
ENTRYPOINT ["python3", "app.py"]
