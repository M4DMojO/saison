import cv2
import logging
from os.path import join

from ultralytics import YOLO
from google.cloud import storage
import unicodedata
from werkzeug.utils import secure_filename
import os

import math


def get_all_weights_from_bucket():
    """
    Load the wieghts of all the models : cls, seg and total
    """
    for model in ["cls", "seg", 'total']:
        get_weights_from_bucket(model)


def get_weights_from_bucket(model:str):
    """
    Load the weights of a given model
    Args:
        model (str): the model to laod : cls|total|seg

    Raises:
        Exception: raise when not the correct arg is unrecognized
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket('all-weights')
    base_path = join("src", "appFlask", "models")
    if model == "cls" or model == "total":
        if model == "cls":
            name = "vgg_classification_big.keras"
        else:
            name = "yolo_total.pt"
        blob = bucket.blob(name)
        destination_file_name = join(base_path, name)
        blob.download_to_filename(destination_file_name)
    elif model == "seg":
        for name in ["vgg_classification_small.keras", "yolo_segmentation.pt"]:
            blob = bucket.blob(name)
            destination_file_name = join(base_path, name)
            blob.download_to_filename(destination_file_name)
    else:
        raise Exception("No such argument, use : cls|total|seg")


def get_result_from_yolo_total(results):
    """
    Extrait les informations de détection d'objet à partir des résultats YOLO.

    Parameters:
    results (list): Liste de résultats renvoyée par YOLO.

    Returns:
    list: Liste de dictionnaires contenant l'ID de l'objet (fruit_id), la confiance, et les coordonnées de la bounding box (x1, y1, x2, y2).
            Un dictionnaire par fruit/légume détecté.
    """
    boxes = [x.boxes for x in results[0]]
    output = []
    if len(boxes) > 0:
        for box in boxes:
            fruit_id = int(box.cls.cpu().numpy()[0])  # Label détecté par YOLO
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])  # Coordonnées de la bounding box
            confidence = box.conf.cpu().numpy()[0]  # Confiance de la détection

            output.append({
                'fruit_id': fruit_id,
                'confidence': confidence,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            })
    return output


def date_period(list_of_month, all_months):
    """
    Convertit une liste de mois sous forme de chaînes en leurs équivalents entiers.

    Parameters:
    list_of_month (list): Liste de mois sous forme de chaînes (ex: ['January', 'February']).
    all_months (dict): Dictionnaire de correspondance mois-chaîne vers mois-id.

    Returns:
    list: Liste d'entiers correspondant aux mois.
    """
    return [int(x) for x, y in all_months.items() if y.lower() in list_of_month]


def _number_around(integer, width):
    """
    Retourne une liste des mois proches d'un mois donné.

    Parameters:
    integer (int): Numéro du mois (1-12).
    width (int): Nombre de mois à inclure avant et après le mois donné.

    Returns:
    list: Liste des mois proches sous forme d'entiers.
    """
    list_around = []
    for i in range(1, width + 1):
        list_around.append((integer + i) % 12)
        list_around.append((integer - i) % 12)
    
    list_around_2 = [12 if i == 0 else i for i in list_around]
    return list_around_2


def enclosing_month(list_of_month, all_months, width=1):
    """
    Trouve les mois adjacents à une liste de mois donnée.

    Parameters:
    list_of_month (list): Liste de mois sous forme de chaînes.
    all_months (dict): Dictionnaire de correspondance mois-chaîne vers mois-id.
    width (int): Nombre de mois avant et après à inclure.

    Returns:
    list: Liste de mois adjacents sous forme d'entiers.
    """
    enclosing = set()
    list_of_month_number = date_period(list_of_month, all_months)
    
    for month in list_of_month_number:
        enclosing = enclosing.union(set(_number_around(month, width)))
    
    return list(enclosing.difference(set(list_of_month_number)))


import logging

def _draw_one_bounding_box(img, data):
    """
    Dessine une bounding box et son label sur une image.

    Parameters:
    img (numpy array): Image sur laquelle dessiner la bounding box.
    data (dict): Dictionnaire contenant les informations de la bounding box (coordonnées, couleur, label, etc.).
                    format du dictionnaire :
                    {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'color': (255, 0, 0), #bgr
                        'fruit_name': 'fruit_name',
                        'confidence': 0.8
                    }

    Returns:
    numpy array: Image modifiée avec la bounding box.
    """
    
    # Coordonnées de la bounding box
    x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
    color_bgr = data['color']  # Couleur de la bounding box
    label_text = f"{data['fruit_name']}: {data['confidence']:.2f}"  # Label (fruit + confiance)

    # Dessiner la bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

    # Préparer et dessiner le texte du label
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), (255, 255, 255), -1)
    cv2.putText(img, label_text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 1)

    return img


def draw_bounding_boxes(img, config_dict, results):
    """
    Applique le modèle choisit par l'utilisateur, détermine la saisonnalité, et dessine des bounding boxes autour des fruits détectés.

    Parameters:
    img (numpy array): Image sur laquelle dessiner les bounding boxes.
    config_dict (dict): Dictionnaire de configuration contenant les informations du modèle, des fruits, de la saisonnalité, etc.

    Returns:
    numpy array: Image avec les bounding boxes dessinées.
    """
    for result in results:
        boxes = result.boxes  # Get the boxes for the result
        for box in boxes:
            confidence = float(box.conf)  # Extract the confidence
            if confidence >= config_dict['MINIMUM_CONFIDENCE']:
                fruit_id = int(box.cls)  # Extract the class ID

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Determine the fruit's seasonality based on the current month
                fruit_months = config_dict['FRUIT_SEASONS'][fruit_id][config_dict['CURRENT_COUNTRY_ID']]
                current_month = int(config_dict['CURRENT_MONTH_ID'])
                all_months = config_dict['MONTHS']

                if current_month in date_period(fruit_months, all_months):
                    seasonality = "2"  # In-season
                elif current_month in enclosing_month(fruit_months, all_months):
                    seasonality = "1"  # Near-season
                else:
                    seasonality = "0"  # Out-of-season

                # Prepare the bounding box data
                data = {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "confidence": confidence,
                    "fruit_name": config_dict["FRUITS"][fruit_id],
                    "color": config_dict['SEASONALITY_TO_COLOR'][seasonality]
                }

                logging.debug(f"Dessin de la bounding box pour {data['fruit_name']}")
                # Draw the bounding box
                img = _draw_one_bounding_box(img, data)

    logging.debug("Fin du dessin des bounding boxes")
    return img


# Fonction pour supprimer les accents
def _remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

# Fonction pour sécuriser le nom du fichier
def clean_filename(filename):
    # Supprimer les accents du nom du fichier
    filename = _remove_accents(filename)

    # Séparer le nom du fichier et l'extension
    name, ext = os.path.splitext(filename)

    # Remplacer les points dans le nom par des underscores
    name = name.replace('.', '_')

    # Recréer le nom du fichier avec l'extension
    cleaned_filename = f"{name}{ext}"

    # Sécuriser le nom du fichier en utilisant secure_filename
    return secure_filename(cleaned_filename)

# Fonction pour vérifier l'extension du fichier image
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg", "jpeg"}



def generate_frame(config_dict, models):
    source_type = config_dict.get('SOURCE_TYPE', 'local')
    source_url = config_dict.get('SOURCE_URL', '')

    if source_type == 'local':
        cap = cv2.VideoCapture(0)  # Local camera
    elif source_type == 'remote' and source_url:
        cap = cv2.VideoCapture(source_url)  # Remote camera
    else:
        logging.error("Source type non valide ou URL manquante.")
        return
    
    if not cap.isOpened():
        logging.error(f"Erreur d'ouverture du flux vidéo à l'URL: {source_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Erreur lors de la lecture de la frame vidéo.")
            break

        model_id = config_dict.get('CURRENT_MODEL_ID', '0')
        model = models[int(model_id)]

        try:
            results = model(frame)
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction sur la frame vidéo : {e}")
            continue

        border_color = (245, 245, 245)
        frame_with_border = cv2.copyMakeBorder(
            frame, 50, 50, 50, 50,
            cv2.BORDER_CONSTANT,
            value=border_color
        )

        try:
            frame_out = draw_bounding_boxes(frame_with_border, config_dict, results)
        except Exception as e:
            logging.error(f"Erreur lors du dessin des boîtes de délimitation: {e}")
            continue

        ret, buffer = cv2.imencode('.jpg', frame_out)
        frame_out = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_out + b'\r\n')

    cap.release()
