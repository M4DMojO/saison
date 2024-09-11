from flask import Flask, render_template, request
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import shutil
import json
import cv2
import custom_function as custom
 
app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])


# Fonction pour vérifier l"extension du fichier image (en principe déjà vérifiée dans le formulaire HTML)
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg", "jpeg"}


# Chargement des dictionnaires à partir du fichier JSON
with open('../data/season.json', 'r') as f:
    data = json.load(f)

app.config['FRUITS'] = {item['_id']: item['name'] for item in data['master_season']}
app.config['MODELS'] = data['model_dict']
app.config['MONTHS'] = data['month_dict']
app.config['COUNTRIES'] = data['countries_dict']
app.config['SEASONALITY_TO_COLOR'] = data['seasonality_color_dict_bgr']
app.config['FRUIT_SEASONS'] = {
                                item['_id']: {key: item.get(key, [])    for key in item if key.startswith('season_')}
                                                                        for item in data['master_season']
                            }

# Configuration du dossier de sauvegarde
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

# Confidence mini pour les détections
app.config['MINIMUM_CONFIDENCE'] = 0.5

# init variables 
app.config['CURRENT_MODEL_ID'] = "0" # yolo_total par défaut
app.config['CURRENT_COUNTRY_ID'] = "season_fr" # France par défaut
app.config['CURRENT_MONTH_ID'] = datetime.today().strftime("%m") # Mois actuel
app.config['CURRENT_IMAGE_PATH'] = ''

# Vider le dossier "uploads"
for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
            logging.info(f"Fichier supprimé : {file_path}")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            logging.info(f"Dossier supprimé : {file_path}")
    except Exception as e:
        logging.error(f"Erreur lors de la suppression de {file_path}: {e}")



# Page d'accueil
@app.route("/")
def index():
    logging.info("Accès à la page d'accueil.")    

    return render_template("index.html", app_dict=app.config)




# Page de résultat
@app.route("/result", methods=["POST"])
def result():
    logging.info("Route '/result' : Traitement de la requête POST.")
    

    # Récupérer l"image depuis le formulaire HTML
    if "img_ipt" not in request.files:
        logging.error("Route '/result' : Aucun fichier sélectionné.")

    file = request.files["img_ipt"]

    if file.filename == "":
        logging.error("Route '/result' : Aucun fichier sélectionné.")

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename) # assurer la sécurité du nom du fichier (pas d"accents, espace, ...)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename) # sauvegarder le fichier dans le dossier "uploads"
            file.save(file_path)
            logging.info(f"Route '/result' :  Image enregistrée ici : {file_path}")

        except Exception as e:
            logging.error(f"Route '/result' : Sauvegarde impossible: {e}")

    else:
        file_path = app.config['CURRENT_IMAGE_PATH'] # pour le cas où on revient de result.html sans changer d'image
        logging.error("Route '/result' : Format du fichier non supporté.")

    # Récupérer les données du formulaire HTML
    model_id = request.form.get("model_id")
    month_id = request.form.get("month_id")
    country_id = request.form.get("country_id")

    # Mettre à jour les variables d"application
    app.config['CURRENT_MODEL_ID']   = model_id
    app.config['CURRENT_MONTH_ID']   = month_id
    app.config['CURRENT_COUNTRY_ID'] = country_id
    app.config['CURRENT_IMAGE_PATH'] = file_path
    app.config['IMAGE_PATH_OUTPUT']  = file_path 

    logging.error(f"Route '/result' : model _id = {app.config['CURRENT_MODEL_ID']}.")

    # Image d'entrée
    img = cv2.imread(app.config['CURRENT_IMAGE_PATH'])  
    # Prediction 
    img_out = custom.draw_bounding_boxes(img, app.config)
    
    # Définir le chemin de sortie de l'image modifiée
    output_path = app.config['CURRENT_IMAGE_PATH'].split(".")[0] + "_out.jpg"

    # Enregistrer l'image modifiée avec les bounding boxes
    cv2.imwrite(output_path, img_out) 
    
    # Mettre à jour le chemin de sortie dans la configuration
    app.config['IMAGE_PATH_OUTPUT'] = output_path
  

    output_dict = { "model_name"    : app.config['MODELS'][ app.config['CURRENT_MODEL_ID'] ],
                    "month_name"    : app.config['MONTHS'][ app.config['CURRENT_MONTH_ID'] ],
                    "country_name"  : app.config['COUNTRIES'][ app.config['CURRENT_COUNTRY_ID'] ],
                    "image_path"    : app.config['IMAGE_PATH_OUTPUT']
    }
    
    return render_template("result.html", output_dict=output_dict)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
    # port 8080 pour google cloud run
# 