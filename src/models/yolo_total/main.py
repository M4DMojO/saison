
import torch
from ultralytics import YOLO
import os

# Chemin vers le fichier de poids
model_path = 'runs/detect_iter2/train12/weights/best.pt'


# Chemin vers le fichier de données
data_path = 'data.yaml'

# Chemin vers le répertoire de sauvegarde
save_dir = 'runs/detect'

# Nombre d'epochs
epochs = 400


model = YOLO(model_path)
model.train(data=data_path, epochs=epochs, save_dir=save_dir)

