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