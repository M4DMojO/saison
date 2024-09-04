import click
import os
from models.finetuning import make_train_val_folder, make_generator, fit_and_export
from data_cleaner import data_vgg_cls, data_seg, data_total, data_vgg_seg


@click.group(name='"saison')
def saison():
    pass

@saison.command(name="datamaker", help="make the data from content/OID/Dataset")
@click.argument("goal", nargs=1, help='seg|vgg|total')
def datamaker(goal:str):
    if goal == 'total':
        data_total()
    elif goal == 'seg':
        data_vgg_seg()
        data_seg()
    elif goal == "vgg":
        data_vgg_cls()
    else:
        print("I did not understand the command, try with total, seg or vgg.")

@saison.command(name='finetuning', help='Define if we should make the "big" or "short" classification')
@click.argument("type_finetuning", nargs=1)
def finetuning(type_finetuning:str='big'):

    if type_finetuning == 'small':
        folder = "yolo_segmentation"
        label_dir = os.path.join("data", folder , 'val')
        save_path = os.path.join("src", "models", 'yolo_segmentation', 'vgg')
    else:
        folder = "vgg_classification"
        save_path = os.path.join("src", "models", 'vgg_classification')

    root_dir = os.path.join('data', folder, 'datasets')
    train_dir = os.path.join("data", folder, 'train')
    val_dir = os.path.join("data", folder, 'val')

    make_train_val_folder(root_dir, train_path=train_dir, val_path=val_dir)
    train_gen, val_gen = make_generator(train_path=train_dir, val_path=val_dir)

    save_path = os.path.join("src", "models", 'vgg_classification')

    fit_and_export(train_gen, val_gen)
