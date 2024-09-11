# saison





# Configuration de la VM

## Mise à Jour et Installation de Zsh
```bash
sudo apt update
sudo apt install zsh -y
``` 

## Définir Zsh comme Shell par Défaut

```bash
sudo nano /etc/default/useradd
SHELL=/usr/bin/zsh
``` 

## Configurer ~/.zshrc pour l'installation automatique d'Oh My Zsh
`nano ~/.zshrc`

ajouter : 
```bash
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
fi
```

## Si le Shell par Défaut Ne Fonctionne Pas
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```


## Instalation de python et des packages
Les VM sont vides originalement. Il faut installer python et pip avant de pouvoir exécuter le code.
```bash
sudo apt-get update
sudo apt-get install python3
sugo apt-get install python3-pip
sudo apt-get update
```

Pour télécharger les autres librairies, il faut d'abord créer un environnement virtuel.
```bash
sudo apt-get update
sudo apt-get install python3.10-venv
python -m venv /path/to/new/virtual/environment
```
Puis se placer dedans:
```bash
source /path/to/new/virtual/environment/bin/activate
```

Enfin, aller dans le projet et installer les librairies
```bash
cd saison
pip install .
```

Pour pouvoir modifier le code, il faut aussi installer les librairies de développement
```bash
pip install -e .
pip install -r requirments.txt
```
## Avoir le bon format de données
Il faut en premier lieu importer les donneées `brut` et les mettre dans le dossier `data`. 
### Pour yolo total
```bash
saison datamaker total
```
### Pour yolo et vgg16 segmentés
A compléter avec la partie de Luc.
```bash
saison datamaker seg
```
### Pour vgg16
```bash
saison datamaker vgg
```
## Lancer les entraînements
### Lancer l'entraînement des modèles YOLO
### Lancer l'entraînement des modèles vgg16
On peut lancer l'entraînement de vgg avec la commande :
```bash
saison finetuning
```
Cela lancera l'entraînement avec les options par défaut.


## Transfert des données depuis la machine locale 

```bash
gcloud compute scp --recurse /path/to/folder/or/file user@vm:~/new/path/to/folder/or/file
```

## transférer un fichier vers un bucket 
`gsutil cp models/yolo_total/runs/detect/train7/weights/best.pt gs://yolo_total_weights/iter1.pt`

