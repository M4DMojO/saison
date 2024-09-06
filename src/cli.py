import click
import os
from src.models.finetuning import make_train_val_folder, make_generator, fit_and_export
from src.data_cleaner import data_vgg_cls, data_seg, data_total, data_vgg_seg


@click.group(name='"saison')
def saison():
    pass

@saison.command(name="datamaker", help="Make the data from brut_. can be either vgg|seg|total")
@click.argument("goal", nargs=1)
@click.option("--vgg_only", 
              "-vgg", 
              default=True, 
              type=bool)
def datamaker(goal:str, vgg_only:bool=True):
    if goal == 'total':
        data_total()
    elif goal == 'seg':
        
        data_vgg_seg()
        if not vgg_only:
            data_seg()
    elif goal == "vgg":
        data_vgg_cls()
    else:
        print("I did not understand the command, try with total, seg or vgg.")


@saison.command(name="train_val_fol")
@click.option("--task", 
              "-t",
              default="vgg")
@click.option("--nb_img",
              "-nb", 
              default=400, 
              type=int)
def train_val_fol(task:str="vgg", nb_img:int=400):
    if task == "vgg":
        folder = "vgg_classification"
        folder_data = "datasets"
    elif task == "seg":
        folder = 'yolo_segmentation'
        folder_data = "dataset_cropped"
    
    root_dir = os.path.join('data', folder, folder_data)
    train_dir = os.path.join("data", folder, 'train')
    val_dir = os.path.join("data", folder, 'val')

    make_train_val_folder(root_dir, train_path=train_dir, val_path=val_dir, nb_img=nb_img)


@saison.command(name='finetuning', help='Define if we should make the "big" or "short" classification')
@click.argument("type_finetuning", nargs=1)
@click.option('--from_checkpoint',
              '-ckpt', 
              default=False, 
              type=bool)
@click.option('--epochs', 
              '-e', 
              default=20, 
              type=int)
@click.option('--from_epoch', 
              '-f', 
              default=0, 
              type=int)
def finetuning(type_finetuning:str='big',
               from_checkpoint:bool=False,
               epochs:int=20, 
               from_epoch:int=0):

    if type_finetuning == 'small':
        folder = "yolo_segmentation"
        save_path = os.path.join("src", "models", folder, 'vgg')
    else:
        folder = "vgg_classification"
        
    save_path = os.path.join("src", "models", folder , type_finetuning, "vgg_classification.h5")
    chekcpoint_dir = os.path.join("src", "models", folder , type_finetuning, "checkpoint")

    root_dir = os.path.join('data', folder, 'datasets')
    train_dir = os.path.join("data", folder, 'train')
    val_dir = os.path.join("data", folder, 'val')

    make_train_val_folder(root_dir, train_path=train_dir, val_path=val_dir)
    train_gen, val_gen = make_generator(train_path=train_dir, val_path=val_dir)

    fit_and_export(train_gen, val_gen, save_path=save_path, checkpoint_dir=chekcpoint_dir, from_pretrained=from_checkpoint,
                   epochs=epochs, from_epoch)
