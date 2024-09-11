import cv2
from ultralytics import YOLO


def _get_result_from_yolo_total(results):
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
    return [int(x) for x, y in all_months.items() if y in list_of_month]


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


def draw_bounding_boxes(img, config_dict):
    """
    Applique le modèle choisit par l'utilisateur, détermine la saisonnalité, et dessine des bounding boxes autour des fruits détectés.

    Parameters:
    img (numpy array): Image sur laquelle dessiner les bounding boxes.
    config_dict (dict): Dictionnaire de configuration contenant les informations du modèle, des fruits, de la saisonnalité, etc.

    Returns:
    numpy array: Image avec les bounding boxes dessinées.
    """

    # Chargement du modèle YOLO selon l'ID du modèle
    if config_dict['CURRENT_MODEL_ID'] == "0":  # modèle YOLO total
        model = YOLO('../models/yolo_total.pt')
        predict = model(config_dict['CURRENT_IMAGE_PATH'])
        results = _get_result_from_yolo_total(predict)
    elif config_dict['CURRENT_MODEL_ID'] == "1":
        pass
    elif config_dict['CURRENT_MODEL_ID'] == "2":
        pass

    # Parcourir les résultats de détection
    for result in results:
        confidence = config_dict['MINIMUM_CONFIDENCE']
        
        if confidence >= 0.5:
            fruit_id = result['fruit_id']
            
            # Détermination de la saisonnalité du fruit
            fruit_months = config_dict['FRUIT_SEASONS'][fruit_id][config_dict['CURRENT_COUNTRY_ID']]
            current_month = int(config_dict['CURRENT_MONTH_ID'])
            all_months = config_dict['MONTHS']

            # Vérification de la saisonnalité
            if current_month in date_period(fruit_months, all_months):
                seasonality = "2"  # En saison
            elif current_month in enclosing_month(fruit_months, all_months):
                seasonality = "1"  # Hors saison proche
            else:
                seasonality = "0"  # Hors saison

            # Création du dictionnaire de données pour chaque fruit détecté
            data = {
                "x1": result['x1'], "y1": result['y1'],
                "x2": result['x2'], "y2": result['y2'],
                "confidence": confidence,
                "fruit_name": config_dict["FRUITS"][fruit_id],
                "color": config_dict['SEASONALITY_TO_COLOR'][seasonality]
            }

            # Dessiner la bounding box avec le label
            img = _draw_one_bounding_box(img, data)

    
    # ajout d'une bordure au bord de l'image
    border_color = (245, 245, 245)  # Couleur en RGB : --color-background-dark: #f5f5f5 du HTML
    img_with_border = cv2.copyMakeBorder(
                                        img,
                                        50, 50, 50, 50,  # Top, Bottom, Left, Right
                                        cv2.BORDER_CONSTANT,
                                        value=border_color
                                    )

    # Retourner l'image finale avec les bounding boxes
    return img_with_border
