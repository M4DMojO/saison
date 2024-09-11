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
`sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"`


## instalation de python et des packages


## transfert des données depuis la machine locale 
se placer dans le dossier data saison/data/yolo_total
`gcloud compute scp --recurse datasets beltranluc0@vm-yolo:~/code/BeltranLuc/saison/data/yolo_total/`


## transférer un fichier vers un bucket 
`gsutil cp models/yolo_total/runs/detect/train7/weights/best.pt gs://yolo_total_weights/iter1.pt`


## docker
En local : 

créer l'image : `docker buildx build . -t flask_app_saison`
lancer le docker : `docker run -p 5000:5000 flask_app_saison`

L'appli est dispo en local http://localhost:5000

taguer l'image pour GCR : `docker tag flask_app_saison gcr.io/saison-artefact-projet/flask_app_saison:latest`
pousser l'image sur GCR : `docker push gcr.io/saison-artefact-projet/flask_app_saison:latest `

Dans la VM : 

update : `sudo apt update`
installer docker : `sudo apt install docker.io`
démarrer docker : `sudo systemctl start docker`
enabler docker au démarrage : `sudo systemctl enable docker`


Cloud run :
https://console.cloud.google.com/run/create?hl=fr&project=saison-artefact-projet&authuser=1
