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
    logging.debug("Début du dessin des bounding boxes")
    
    # Parcourir les résultats de détection
    for result in results:
        confidence = result['confidence']
        
        if confidence >= config_dict['MINIMUM_CONFIDENCE']:
            fruit_id = result['fruit_id']
                        
            # Détermination de la saisonnalité du fruit
            fruit_months = config_dict['FRUIT_SEASONS'][fruit_id][config_dict['CURRENT_COUNTRY_ID']]
            logging.debug(f"fruit_months: {fruit_months}")
            current_month = int(config_dict['CURRENT_MONTH_ID'])
            logging.debug(f"current_month: {current_month}")
            all_months = config_dict['MONTHS']
            logging.debug(f"all_months: {all_months}")

            logging.debug(f"date_period(fruit_months, all_months): {date_period(fruit_months, all_months)}")
            logging.debug(f"enclosing_month(fruit_months, all_months): {enclosing_month(fruit_months, all_months)}")
            # Vérification de la saisonnalité
            if current_month in date_period(fruit_months, all_months):
                seasonality = "2"  # En saison
            elif current_month in enclosing_month(fruit_months, all_months):
                seasonality = "1"  # Hors saison proche
            else:
                seasonality = "0"  # Hors saison


            logging.debug(f"Détermination de la saisonnalité pour {config_dict['FRUITS'][fruit_id]}: {seasonality}")
            # Création du dictionnaire de données pour chaque fruit détecté
            data = {
                "x1": result['x1'], "y1": result['y1'],
                "x2": result['x2'], "y2": result['y2'],
                "confidence": confidence,
                "fruit_name": config_dict["FRUITS"][fruit_id],
                "color": config_dict['SEASONALITY_TO_COLOR'][seasonality]
            }

            logging.debug(f"Dessin de la bounding box pour {data['fruit_name']}")
            # Dessiner la bounding box avec le label
            img = _draw_one_bounding_box(img, data)

    logging.debug("Fin du dessin des bounding boxes")
    # Retourner l'image finale avec les bounding boxes
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


def generate_frame(config_dict):
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        results = config_dict['model']

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0]) 
                
                        
                # Détermination de la saisonnalité du fruit
                fruit_months = ["01", "02", "03"]
                current_month = "05" # Mois actuel
                all_months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

                # Vérification de la saisonnalité
                if current_month in fruit_months:
                    seasonality = "2"  # En saison
                    color = (0, 255, 0)
                elif current_month in ["01", "02", "03","04","12"]:
                    seasonality = "1"  # Hors saison proche
                    color = (0, 165, 255)
                else:
                    seasonality = "0"  # Hors saison
                    color = (0, 0, 255)

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, config_dict['classes'][cls], org, font, fontScale, color, thickness)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()