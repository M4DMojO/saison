import os
import shutil
import errno 
from PIL import Image

def openImage(filename):
    try:
        img = Image.open( os.path.join('data', 'yolo_total', 'datasets', 'images', 'train', filename.split(".")[0] + '.jpg' ))
        return img
    except:
        pass  

    try:
        img = Image.open( os.path.join('data', 'yolo_total', 'datasets', 'images', 'val', filename.split(".")[0] + '.jpg' ))
        return img
    except:
        pass  

def create_directories(base_path):
    os.makedirs(os.path.join(base_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', 'val'), exist_ok=True)

def copy_files(src_paths, dest_base_path, subfolder):
    for src in src_paths:
        src_file = src
        dest_file = os.path.join(dest_base_path, subfolder)
        dest_folder = os.path.dirname(dest_file)
        os.makedirs(dest_folder, exist_ok=True)
        try:
            shutil.copy(src_file, dest_file)
        except Exception as e:
            print(f"Error copying {src_file}: {e}")

def delete_folder(folder_path):
    """Deletes a folder and its contents.

    Args:
        folder_path (str): The path to the folder to delete.
    """

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    try:
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except OSError as e:
        if e.errno == errno.ENOTEMPTY:
            print(f"Error deleting folder '{folder_path}': Folder is not empty.")
        else:
            print(f"Error deleting folder '{folder_path}': {e}")



def modify_txt_files(directory, fruit_to_id):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()  
  
            new_lines = []
            for line in lines:
                values = line.strip().split()
                if len(values) == 6:
                    # Cas particulier pour "Bell pepper"
                    class_name = "Bell pepper"
                    _, _,x_tl, y_tl, x_br, y_br = values
                else:
                    class_name, x_tl, y_tl, x_br, y_br = values
                
                class_id = fruit_to_id.get(class_name, -1)  # Get ID, default -1 if not found
                
                if class_id != -1:  # Only modify if class is found
                    img = openImage( filename.split(".")[0] + '.jpg' )
                    img_w, img_h = img.size

                    x_br = float(x_br)
                    x_tl = float(x_tl)
                    y_br = float(y_br)
                    y_tl = float(y_tl)

                    x_center = (x_br + x_tl)/2/img_w
                    y_center = (y_tl + y_br)/2/img_h
                    w_bb = (x_br - x_tl)/img_w
                    h_bb = (y_br - y_tl)/img_h

                    new_line = f"{class_id} {x_center} {y_center} {w_bb} {h_bb}\n"
                    new_lines.append(new_line)
                

            with open(filepath, "w") as f:
                f.writelines(new_lines)

def supprimer_premier_mot(fichier):
    """
    Supprime le premier mot de chaque ligne d'un fichier .txt.

    Args:
        fichier (str): Chemin vers le fichier .txt.
    """

    with open(fichier, 'r') as f:
        lignes = f.readlines()

    with open(fichier, 'w') as f:
        for ligne in lignes:
            mots = ligne.split()
            if mots:
                f.write(' '.join(mots[1:]) + '\n')

def copier_et_modifier(source, destination):
    """
    Copie un dossier et modifie les fichiers .txt.

    Args:
        source (str): Chemin du dossier source.
        destination (str): Chemin du dossier destination.
    """

    for root, dirs, files in os.walk(source):
        for dir in dirs:
            src_dir = os.path.join(root, dir)
            dest_dir = os.path.join(destination, os.path.relpath(src_dir, source))
            if dir not in ['images', 'datasets']:  # Ne pas copier les dossiers images et datasets
                shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        for file in files:
            if file.endswith('.txt'):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(destination, os.path.relpath(src_file, source))
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy2(src_file, dest_file)
                supprimer_premier_mot(dest_file)
            else:
                # Copier les autres fichiers sans modification
                src_file = os.path.join(root, file)
                dest_file = os.path.join(destination, os.path.relpath(src_file, source))
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy2(src_file, dest_file)

def cropper(image_path:str, save_path:str, label_path:str):
    img = Image.open(image_path)

    # print(image_path)
    # print(label_path)
    # print(save_path)
    # print(save_path.replace(".jpg", ''.join(['_', str(1), '.jpg'])))
    with open(label_path, 'r') as labels:
        counter = 0
        for line in labels:
            l = line.split(' ')
            if len(l) == 5:
                x_center, y_center, w, h = l[1], l[2], l[3], l[4]
            else: #Pour Bell pepper
                x_center, y_center, w, h = l[2], l[3], l[4], l[5]

            x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
            img_w, img_h = img.size
            x_center = x_center
            y_center = y_center

            x1 = x_center - (img_w/2)
            x2 = x_center + (img_w/2)
            y1 = y_center - (img_h/2)
            y2 = y_center + (img_h/2)
            img_tmp = img.crop((x1, y1, x2, y2))

            try:
                img_tmp.save(save_path.replace(".jpg", ''.join(['_', str(counter), '.jpg'])), "JPEG")
            except IOError as e:
                print("Can't save")
                print(f"Error saving image: {e}")

            counter += 1

def traiter_donnees(racine, nouvelle_racine, max_elements, avec_boundingbox=False):
    for fruit in os.listdir(racine):
        chemin_fruit = os.path.join(racine, fruit)
        print(fruit)
        if os.path.isdir(chemin_fruit):
            nouveau_chemin_fruit = os.path.join(nouvelle_racine, fruit.lower())
            os.makedirs(nouveau_chemin_fruit, exist_ok=True)

            if avec_boundingbox:
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

                if avec_boundingbox:
                    # Correspondance du fichier boundingbox dans le dossier BoundingBoxes
                    chemin_boundingbox_source = os.path.join(chemin_boundingboxes, image.replace('.jpg', '.txt'))

                    # Copier et renommer le fichier boundingbox si le fichier existe
                    if os.path.exists(chemin_boundingbox_source):
                        cropper(chemin_image_source, chemin_image_dest, chemin_boundingbox_source)
                else:
                    shutil.copy2(chemin_image_source, chemin_image_dest)

def data_vgg_cls():
    racine = os.path.join("data", "brut_")
    nouvelle_racine =  os.path.join("data", 'vgg_classification', "datasets")
    max_elements = 400
    traiter_donnees(racine, nouvelle_racine, max_elements, avec_boundingbox=False)

def data_vgg_seg():
    racine = os.path.join("data", "brut_")
    nouvelle_racine =  os.path.join("data", 'yolo_segmentation', "dataset_cropped")
    max_elements = 400
    traiter_donnees(racine, nouvelle_racine, max_elements, avec_boundingbox=True)

def data_total():
    dataset = {
            'images': {
                'train': [],
                'val': []
            },
            'labels': {
                'train': [],
                'val': []
            }
        }
    source_folder = os.path.join('data', '_brut')
    fruit_list = sorted(os.listdir(source_folder))
    fruit_to_idx = {fruit : i for i, fruit in enumerate(fruit_list)}

    # Remplissage du dictionnaire de données
    for fruit in fruit_list:
        images_path = os.path.join(source_folder, fruit)
        labels_path = os.path.join(images_path, 'Label')

        images = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')])[:400]
        labels = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])[:400]

        if len(images) != len(labels):
            raise ValueError(f"Le nombre d'images et de labels ne correspond pas dans le dossier '{fruit}'")

        split_point = int(len(images) * 0.8)

        dataset['images']['train'].extend([os.path.join(images_path, img) for img in images[:split_point]])
        dataset['images']['val'].extend([os.path.join(images_path, img) for img in images[split_point:]])

        dataset['labels']['train'].extend([os.path.join(labels_path, lbl) for lbl in labels[:split_point]])
        dataset['labels']['val'].extend([os.path.join(labels_path, lbl) for lbl in labels[split_point:]])



    # path sortie
    dataset_path = os.path.join('data', 'yolo_total', 'datasets')

    # Delete existing directories if necessary
    delete_folder(os.path.join(dataset_path, 'images'))
    delete_folder(os.path.join(dataset_path, 'labels'))

    # Create the dataset structure
    create_directories(dataset_path)

    # Copy files
    for split in ['train', 'val']:
        copy_files(dataset['images'][split], dataset_path, os.path.join('images', split))
        copy_files(dataset['labels'][split], dataset_path, os.path.join('labels', split))

        source_folder = os.path.join('..', 'data', 'brut_')
        fruit_list = sorted(os.listdir(source_folder))
        fruit_to_idx = {fruit : i for i, fruit in enumerate(fruit_list)}
        modify_txt_files(os.path.join("data", "yolo_total", "datasets","labels", "train"), 
                        fruit_to_idx)
        modify_txt_files(os.path.join("data", "yolo_total", "datasets","labels", "val"), 
                        fruit_to_idx)


def data_seg():
    source_folder = os.path.join("data", "yolo_total", "datasets")
    destination_folder = os.path.join("data", "yolo_segmentation", "datasets")

    # Copier et modifier
    copier_et_modifier(source_folder, destination_folder)