import os
import shutil
import pickle

def traiter_donnees(racine, nouvelle_racine, max_elements, avec_boundingbox=False):
    for fruit in os.listdir(racine):
        chemin_fruit = os.path.join(racine, fruit)
        
        if os.path.isdir(chemin_fruit):
            nouveau_chemin_fruit = os.path.join(nouvelle_racine, fruit.lower())
            os.makedirs(nouveau_chemin_fruit, exist_ok=True)
            
            if avec_boundingbox:
                nouveau_chemin_boundingboxes = os.path.join(nouveau_chemin_fruit, 'boundingboxes')
                os.makedirs(nouveau_chemin_boundingboxes, exist_ok=True)
                chemin_boundingboxes = os.path.join(chemin_fruit, 'Label')
            
            images = [f for f in os.listdir(chemin_fruit) if f.endswith('.jpg')]
            images.sort()
            
            images = images[:max_elements]
            
            # Copier et renommer les images
            for i, image in enumerate(images, start=1):
                nouveau_nom_image = f"{fruit.lower()}_{i}.jpg"
                
                # Chemin complet pour l'image source et destination
                chemin_image_source = os.path.join(chemin_fruit, image)
                chemin_image_dest = os.path.join(nouveau_chemin_fruit, nouveau_nom_image)
                
                # Copier et renommer l'image
                shutil.copy2(chemin_image_source, chemin_image_dest)
                
                if avec_boundingbox:
                    # Correspondance du fichier boundingbox dans le dossier BoundingBoxes
                    chemin_boundingbox_source = os.path.join(chemin_boundingboxes, image.replace('.jpg', '.txt'))
                    chemin_boundingbox_dest = os.path.join(nouveau_chemin_boundingboxes, f"boundingbox_{fruit.lower()}_{i}.txt")
                    
                    # Copier et renommer le fichier boundingbox si le fichier existe
                    if os.path.exists(chemin_boundingbox_source):
                        shutil.copy2(chemin_boundingbox_source, chemin_boundingbox_dest)

def sauvegarder_arborescence_et_fichiers(chemin):
    arborescence = {}
    for racine, dirs, fichiers in os.walk(chemin):
        dossier_relatif = os.path.relpath(racine, chemin)
        if dossier_relatif == '.':
            dossier_relatif = ''  
        arborescence[dossier_relatif] = {
            'fichiers': {},
            'dossiers': {}
        }
        for fichier in fichiers:
            chemin_fichier = os.path.join(racine, fichier)
            with open(chemin_fichier, 'rb') as f:
                arborescence[dossier_relatif]['fichiers'][fichier] = f.read()
        for dossier in dirs:
            sous_dossier = os.path.join(dossier_relatif, dossier)
            arborescence[dossier_relatif]['dossiers'][dossier] = {}
    
    return arborescence

# Configuration
racine = 'databrut'
nouvelle_racine = 'dataset'
max_elements = 400
dataset_root = 'dataset'

# Traitement sans bounding boxes
traiter_donnees(racine, nouvelle_racine, max_elements, avec_boundingbox=False)
arborescence_dataset = sauvegarder_arborescence_et_fichiers(dataset_root)
pickle_file = 'dataset_reduit.pkl'

with open(pickle_file, 'wb') as f:
    pickle.dump(arborescence_dataset, f)

    
# Traitement avec bounding boxes
traiter_donnees(racine, nouvelle_racine, max_elements, avec_boundingbox=True)
arborescence_dataset = sauvegarder_arborescence_et_fichiers(dataset_root)
pickle_file = 'dataset_reduit_boundingbox.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(arborescence_dataset, f)



# Supprimer le dossier dataset 
shutil.rmtree(dataset_root)
