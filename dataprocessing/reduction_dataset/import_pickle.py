import os
import pickle

pickle_file = 'dataset_reduit.pkl'

with open(pickle_file, 'rb') as f:
    arborescence = pickle.load(f)

# Fonction pour recréer l'arborescence et écrire les fichiers
def recreer_arborescence_et_fichiers(arbo, chemin_base='dataset'):
    for dossier, contenu in arbo.items():
        nouveau_dossier = os.path.join(chemin_base, dossier)
        os.makedirs(nouveau_dossier, exist_ok=True)

        if isinstance(contenu, dict) and 'fichiers' in contenu:
            for fichier, donnees in contenu['fichiers'].items():
                chemin_fichier = os.path.join(nouveau_dossier, fichier)
                with open(chemin_fichier, 'wb') as f:
                    f.write(donnees)

        if isinstance(contenu, dict) and 'dossiers' in contenu:
            for sous_dossier, sous_contenu in contenu['dossiers'].items():
                sous_dossier_path = os.path.join(nouveau_dossier, sous_dossier)
                recreer_arborescence_et_fichiers(sous_contenu, sous_dossier_path)


recreer_arborescence_et_fichiers(arborescence)

