import os
import pickle
import shutil
import argparse

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

def main():
    # Configuration de argparse pour gérer les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Importer une arborescence et des fichiers depuis un pickle.')
    parser.add_argument('pickle_file', type=str, help='Le fichier pickle à utiliser pour l\'import')
    
    args = parser.parse_args()
    pickle_file = args.pickle_file

    # Supprimer le dossier dataset avant de faire l'import
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')

    # Charger le fichier pickle
    with open(pickle_file, 'rb') as f:
        arborescence = pickle.load(f)

    # Recréer l'arborescence et les fichiers
    recreer_arborescence_et_fichiers(arborescence)

if __name__ == '__main__':
    main()
