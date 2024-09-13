from flask import Flask, render_template, request, redirect, url_for, Response
from datetime import datetime
import cv2
import json
import logging
import os

from model import load_models, get_results
from custom_function import allowed_file, clean_filename, draw_bounding_boxes, generate_frame
 
app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])

# Chargement des dictionnaires à partir du fichier JSON
with open('../data/season.json', 'r') as f:
    data = json.load(f)

app.config['FRUITS'] = {item['_id']: item['name'] for item in data['master_season']}
app.config['MODELS'] = data['model_dict']
app.config['MONTHS'] = { key : value.capitalize() for key, value in data['month_dict'].items()}
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


# Flag changement de param pour le check d'une image
flag_recheck_pic = False

models = load_models()

# Assurer que le dossier de téléchargement existe
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def load_form():
    # Récupérer les données du formulaire HTML
    model_id = request.form.get("model_id")
    month_id = request.form.get("month_id")
    country_id = request.form.get("country_id")

    source_type = request.form.get("source_type")
    source_url = request.form.get("source_url")


    # Mettre à jour les variables de l'application
    if model_id:
        app.config['CURRENT_MODEL_ID'] = model_id
    if month_id:
        app.config['CURRENT_MONTH_ID'] = month_id
    if country_id:
        app.config['CURRENT_COUNTRY_ID'] = country_id
    if source_type:
        app.config['SOURCE_TYPE'] = source_type
    if source_url:
        app.config['SOURCE_URL'] = source_url

# Page d'accueil
@app.route("/")
def index():
    return render_template("index.html", app_dict=app.config)

@app.route("/mode_photo", methods=["POST", "GET"])
def mode_photo():
    if request.method == "POST":
        load_form()
    return render_template("mode_photo.html", app_dict=app.config)

# Page de résultat
@app.route("/mode_photo_result", methods=["POST"])
def mode_photo_result():
    # Vérification de la valeur du flag 'flag_recheck_pic'
    flag_recheck_pic = request.form.get('flag_recheck_pic', 'False') == 'True'

    # Chargement de l'image uploadée sauf en cas de re-check
    if not flag_recheck_pic :
        # Vérifier si un fichier a été soumis
        if 'img_ipt' not in request.files or request.files['img_ipt'].filename == '':
            logging.error("Route '/mode_photo_result' : Aucun fichier sélectionné.")
            return redirect(url_for('mode_photo'))  # Rediriger vers le formulaire

        file = request.files['img_ipt']

        # Valider et sécuriser le nom du fichier
        if not allowed_file(file.filename):
            logging.error("Route '/mode_photo_result' : Format de fichier non supporté.")
            return redirect(url_for('mode_photo'))

        # Nettoyer le nom du fichier en supprimant les accents et les points non autorisés
        filename = clean_filename(file.filename)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        app.config['IMG_SRC_PATH'] = file_path
        
        logging.info(f"Route '/mode_photo_result' : Fichier enregistré sous {file_path}")

    # Re-check : on recharge le formulaire pour récupérer les nouvelles valeurs des paramètres
    if flag_recheck_pic :
        load_form()

    # Vérifier l'ID du modèle actuel
    model_id = app.config.get('CURRENT_MODEL_ID', '0')
    logging.info(f"Route '/mode_photo_result' : model _id = {model_id}")

    # Lire l'image depuis le chemin du fichier
    img = cv2.imread(app.config['IMG_SRC_PATH'])
    if img is None:
        logging.info("Route '/mode_photo_result' : Impossible de lire l'image.")
        return redirect(url_for('mode_photo'))
    


    # ajout d'une bordure au bord de l'image
    border_color = (245, 245, 245)  # Couleur en RGB : --color-background-dark: #f5f5f5 du HTML
    img_with_border = cv2.copyMakeBorder(
                                        img,
                                        50, 50, 50, 50,  # Top, Bottom, Left, Right
                                        cv2.BORDER_CONSTANT,
                                        value=border_color
                                    )
    
    app.config['IMG_SRC_BORDER_PATH'] = app.config['IMG_SRC_PATH'].split('.')[0] + '_border.jpg'
    cv2.imwrite(app.config['IMG_SRC_BORDER_PATH'], img_with_border)

    results = get_results(models[int(model_id)], app.config['IMG_SRC_BORDER_PATH'], int(model_id))

    # Prédiction
    try:
        img_out = draw_bounding_boxes(img_with_border, app.config, results)
    except Exception as e:
        logging.error(f"Route '/mode_photo_result' : Erreur lors du traitement de l'image - {e}")
        return redirect(url_for('mode_photo'))

    # Définir le chemin de sortie de l'image modifiée
    logging.info(f"Route '/mode_photo_result' : path source : {app.config['IMG_SRC_PATH']}")
    file_path_start, file_path_end = app.config['IMG_SRC_PATH'].split('.')
    logging.info(f"Route '/mode_photo_result' : file_path_start = {file_path_start}; file_path_end = {file_path_end}")
    output_path = file_path_start + "_out." + file_path_end

    # Sauvegarder l'image traitée
    cv2.imwrite(output_path, img_out)
    logging.info(f"Route '/mode_photo_result' : Image modifiée enregistrée sous {output_path}")

    # Préparer les données à afficher
    output_dict = {
        "model_name": app.config['MODELS'].get(model_id, 'Modèle Inconnu'),
        "month_name": app.config['MONTHS'].get(app.config.get('CURRENT_MONTH_ID', '0'), 'Mois Inconnu'),
        "country_name": app.config['COUNTRIES'].get(app.config.get('CURRENT_COUNTRY_ID', '0'), 'Pays Inconnu'),
        "image_path": output_path
    }

    return render_template("mode_photo_result.html", output_dict=output_dict, app_dict=app.config)



@app.route("/mode_video_page", methods=["GET", "POST"])
def mode_video_page():
    return render_template('mode_video.html', app_dict=app.config)


@app.route("/mode_video", methods=["POST", "GET"])
def mode_video():
<<<<<<< HEAD
    if request.method == "POST":
        load_form()  

    return Response(generate_frame(config_dict=app.config, models=models), mimetype="multipart/x-mixed-replace; boundary=frame")




=======
    load_form()
    config_dict = { 'model' : models[int(app.config.get('CURRENT_MODEL_ID', '0'))],
                   'classes' :  app.config['FRUITS'].values}
    return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")
>>>>>>> 8a84b7ea2a7b7677b830d691fbbba60bc7d5ab19

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
