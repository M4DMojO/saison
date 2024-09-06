
import torch
from ultralytics import YOLO
import os


# Chemin vers le fichier de poids
model_path = 'yolov8n.pt'

# Chemin vers le fichier de données
data_path = 'data.yaml'

# Chemin vers le répertoire de sauvegarde
save_dir = 'runs/detect'

# Nombre d'epochs
epochs = 4




# URL du fichier de poids YOLOv8n
weights_url = 'https://github.com/ultralytics/yolov8/releases/download/v8.0.0/yolov8n.pt'

# Vérification si le fichier de poids existe déjà
if not os.path.exists(model_path):
    print(f"Téléchargement de {model_path}...")
    urllib.request.urlretrieve(weights_url, model_path)
    print(f"Téléchargement terminé : {model_path}")



model = YOLO(model_path)
model.train(data=data_path, epochs=epochs, save_dir=save_dir)

