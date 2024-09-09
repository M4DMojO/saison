from flask import Flask, render_template, request, redirect, url_for
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import shutil

app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])


# Fonction pour vérifier l"extension du fichier image (en principe déjà vérifiée dans le formulaire HTML)
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg", "jpeg"}



# Dictionnaires
idx_to_model = {0: "yolo_total", 1: "yolo_segmentation", 2: "vgg"} # TODO : lableliser les modèles de manière plus explicite
idx_to_month = {"01": "January", "02": "February", "03": "March", "04": "April",
                "05": "May", "06": "June", "07": "July", "08": "August",
                "09": "September", "10": "October", "11": "November", "12": "December"
}
idx_to_country  = { 0: "France", 1: "USA"} # TODO : récupérer depuis la bdd 
idx_to_fruit    = { 0: 'Apple', 1: 'Artichoke', 2: 'Banana',3: 'Bell pepper',4: 'Broccoli',
                    5: 'Carrot', 6: 'Orange', 7: 'Pear', 8: 'Pumpkin',9: 'Strawberry'}
idx_to_bdbcolor = { 0 : (255, 0, 0)} # TODO : définir les couleurs des bounding boxes pour chaque fruit
# idx_to_season   = 


# Configuration du dossier de sauvegarde
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

# init variables 
app.config['CURRENT_MODEL_ID'] = 0 # yolo_total par défaut
app.config['CURRENT_COUNTRY_ID'] = 0 # France par défaut
app.config['CURRENT_MONTH_ID'] = datetime.today().strftime("%m") # Mois actuel
app.config['CURRENT_IMAGE_PATH'] = ''

# Vider le dossier "uploads"
for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
            logging.info(f"Route '/' : Fichier supprimé : {file_path}")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            logging.info(f"Route '/'' : Dossier supprimé : {file_path}")
    except Exception as e:
        logging.error(f"Route '/' : Erreur lors de la suppression de {file_path}: {e}")



def date_period (list_of_string) : 
    return [int(x) for x, y in idx_to_month.items() if y in list_of_string]

def number_around (integer, width) :
    list_around = [] 
    for i in range(1, width+1): 
        list_around.append((integer+i)%12)
        list_around.append((integer-i)%12)
    print(f"list_around : {list_around}")
    list_around_2=[12 if i == 0 else i for i in list_around]
    return list_around_2
        
def enclosing_month (list_of_month, width) : 
    enclosing=set()
    list_of_month_number = date_period(list_of_month)
    for month in list_of_month_number :
        enclosing = enclosing.union(set(number_around(month, width))) 
    return list(enclosing.difference(set(list_of_month_number)))
        
def month_wrong (list_of_month) : 
    return list( set( range(13) ).difference( set( enclosing_month(list_of_month), date_period(list_of_month) ) ) )



# Page d'accueil
@app.route("/")
def index():
    logging.info("Route '/' : Page d'accueil.")    

    # Formater le dictionnaire pour le template HTML
    output_dict = {
        "models"    : idx_to_model,
        "months"    : idx_to_month,
        "countries" : idx_to_country,
        "image_path":  app.config['CURRENT_IMAGE_PATH'],
        "current_model_id"   : app.config['CURRENT_MODEL_ID'],
        "current_month_id"   : app.config['CURRENT_MONTH_ID'],
        "current_country_id" : app.config['CURRENT_COUNTRY_ID']
    }

    return render_template("index.html", output_dict=output_dict)




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
        logging.error("Route '/result' : Format du fichier non supporté.")

    # Récupérer les données du formulaire HTML
    model_id = request.form.get("model_id")
    month_id = request.form.get("month_id")
    country_id = request.form.get("country_id")

    # Mettre à jour les variables d"application
    app.config['CURRENT_MODEL_ID']   = int(model_id)
    app.config['CURRENT_MONTH_ID']   = month_id
    app.config['CURRENT_COUNTRY_ID'] = int(country_id)
    app.config['CURRENT_IMAGE_PATH'] = file_path
    app.config['IMAGE_PATH_OUTPUT']  = file_path 

    # Prediction 
    # TODO : générer les résultats de la prédiction à partir des données du formulaire HTML et de l"image sauvegardée


    # Calcul saisonnalité 
    # TODO : algo saisonnalité
    


    # Formater le dictionnaire pour le template HTML
    output_dict = { "image_path":  app.config['IMAGE_PATH_OUTPUT'],
                    "results" : {
                                "random_name_0": {
                                        "seasonality": 0, # TODO : résultat du calcul de saisonnalité
                                        "boundingbox_color": idx_to_bdbcolor[0], # TODO : remplacer 0 par l'id du label prédit
                                        "fruit_name": idx_to_fruit[0], # TODO : remplacer 0 par l'id du label prédit
                                        "fruit_confidence": 90, # TODO : récupérer depuis le predict
                                        "fruit_month": "Jan., Feb." # TODO : aller chercher dans la bdd les mois de saisonnalité du fruit
                                    },
                                "random_name_1": {
                                        "seasonality": 2,
                                        "boundingbox_color": (255, 255, 0),
                                        "fruit_name": "banana",
                                        "fruit_confidence": 30,
                                        "fruit_month": "Sept., Oct., Dec., Jan."
                                    }
                                },
                    "model_name"   : idx_to_model[ app.config['CURRENT_MODEL_ID'] ],
                    "month_name"   : idx_to_month[ app.config['CURRENT_MONTH_ID'] ],
                    "country_name" : idx_to_country[ app.config['CURRENT_COUNTRY_ID'] ]
                }
    
    return render_template("result.html", output_dict=output_dict)

if __name__ == "__main__":
    app.run(debug=True)
