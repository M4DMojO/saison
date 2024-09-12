from flask import Flask, render_template, request, redirect, url_for
from models.model import load_models, get_results
import custom_function as custom
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import json
import shutil
import json
import logging
import os
import custom_function as custom
 
app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])

# Fonction pour vérifier l'extension du fichier image
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
    item['_id']: {key: item.get(key, []) for key in item if key.startswith('season_')}
    for item in data['master_season']
}

# Configuration du dossier de sauvegarde
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

# Confidence mini pour les détections
app.config['MINIMUM_CONFIDENCE'] = 0.5

# Initialisation des variables
app.config['CURRENT_MODEL_ID'] = "0"  # yolo_total par défaut
app.config['CURRENT_COUNTRY_ID'] = "season_fr"  # France par défaut
app.config['CURRENT_MONTH_ID'] = datetime.today().strftime("%m")  # Mois actuel

models = load_models()

# Assurer que le dossier de téléchargement existe
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def load_form():
    # Récupérer les données du formulaire HTML
    model_id = request.form.get("model_id")
    month_id = request.form.get("month_id")
    country_id = request.form.get("country_id")

    # Mettre à jour les variables de l'application
    if model_id:
        app.config['CURRENT_MODEL_ID'] = model_id
    if month_id:
        app.config['CURRENT_MONTH_ID'] = month_id
    if country_id:
        app.config['CURRENT_COUNTRY_ID'] = country_id

# Page d'accueil
@app.route("/")
def index():
    logging.info("Accès à la page d'accueil.")
    return render_template("index.html", app_dict=app.config)

@app.route("/mode_photo", methods=["POST", "GET"])
def mode_photo():
    if request.method == "POST":
        load_form()
    return render_template("mode_photo.html", app_dict=app.config)

# Page de résultat
@app.route("/mode_photo_result", methods=["POST"])
def mode_photo_result():
    logging.info("Route '/mode_photo_result' : Traitement de la requête POST.")

    # Vérifier si un fichier a été soumis
    if 'img_ipt' not in request.files or request.files['img_ipt'].filename == '':
        logging.error("Route '/mode_photo_result' : Aucun fichier sélectionné.")
        return redirect(url_for('mode_photo'))  # Rediriger vers le formulaire

    file = request.files['img_ipt']

    # Valider et sécuriser le nom du fichier
    if not allowed_file(file.filename):
        logging.error("Route '/mode_photo_result' : Format de fichier non supporté.")
        return redirect(url_for('mode_photo'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    logging.info(f"Route '/mode_photo_result' : Fichier enregistré sous {file_path}")

    # Vérifier l'ID du modèle actuel
    model_id = app.config.get('CURRENT_MODEL_ID', '0')
    logging.error(f"Route '/mode_photo_result' : model _id = {model_id}")

    # Lire l'image depuis le chemin du fichier
    img = cv2.imread(file_path)
    if img is None:
        logging.error("Route '/mode_photo_result' : Impossible de lire l'image.")
        return redirect(url_for('mode_photo'))
    
    results = get_results(models[model_id], img, model_id)

    # Prédiction
    try:
        img_out = custom.draw_bounding_boxes(img, app.config,results)
    except Exception as e:
        logging.error(f"Route '/mode_photo_result' : Erreur lors du traitement de l'image - {e}")
        return redirect(url_for('mode_photo'))

    # Définir le chemin de sortie de l'image modifiée
    output_filename = f"{os.path.splitext(filename)[0]}_out.jpg"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    # Sauvegarder l'image traitée
    cv2.imwrite(output_path, img_out)
    logging.info(f"Route '/mode_photo_result' : Image modifiée enregistrée sous {output_path}")

    # Supprimer le fichier original
    try:
        os.remove(file_path)
        logging.info(f"Route '/mode_photo_result' : Fichier original supprimé : {file_path}")
    except Exception as e:
        logging.error(f"Route '/mode_photo_result' : Erreur lors de la suppression du fichier original - {e}")

    # Préparer les données à afficher
    output_dict = {
        "model_name": app.config['MODELS'].get(model_id, 'Modèle Inconnu'),
        "month_name": app.config['MONTHS'].get(app.config.get('CURRENT_MONTH_ID', '0'), 'Mois Inconnu'),
        "country_name": app.config['COUNTRIES'].get(app.config.get('CURRENT_COUNTRY_ID', '0'), 'Pays Inconnu'),
        "image_path": output_path
    }

    return render_template("mode_photo_result.html", output_dict=output_dict)


@app.route("/mode_video", methods=["POST"])
def mode_video():
    load_form()
    return render_template("mode_photo.html", app_dict=app.config)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
