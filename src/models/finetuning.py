from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from PIL import Image

import shutil
import os
import sys

VGG_IMG_SHAPE = (224, 224, 3)
VGG_SIZE = (224, 224)
BATCH_SIZE = 64


def scheduler(epoch:int, lr:float) -> float:
    """
    Define the scheduler for the trainning to reduce the learning.
    Parameters
    ----------
    epoch : int
        The current epoch
    lr : in
        The current learning rate

    Returns
    -------
    float
        The updated learning rate
    """
    if epoch % 5 and lr > 0.0001:
        return lr * 0.9
    return lr

def make_train_val_folder(root:str, train_path:str, val_path:str, 
                           nb_img:int=400, val_split:float=0.2):
    """
    Make the train and validation folder for trainning
    ----------
    root : str
        The root directory where all subfolders and images are located
    train_path : str
        The path to the futur trainning folder
    val_path : str
        The path to the futur validation folder
    nb_img : int, optional
        The max image per class. Default is 400
    val_split : float, optional
        Between 0 and 1. The ratio of total images to val. 
    """
    for dir in os.listdir(root):
        root_dir = os.path.join(root, dir)
        train_dir = os.path.join(train_path, dir)
        val_dir = os.path.join(val_path, dir)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        for i in range(len(images)):
            source = os.path.join(root_dir, images[i])
            if i < int(nb_img * (1 - val_split)):
                dest = os.path.join(train_dir, images[i])
            elif i < nb_img:
                dest = os.path.join(val_dir, images[i])
            else:
                break

            shutil.copy2(source, dest)


def make_generator(train_path:str, val_path:str):
    """
    Create the training and validation datasets from the given paths. 
    The datasets are tensorflow.keras.preprocessing.image.ImageDataGenerator.
    They rescaleand flip the images randomly.

    Parameters
    ----------
    train_path : str
        The path to the train folder
    val_pathr : str
        The path to the validation folder

    Returns
    -------
    tuple
        A tuple containning the train and val 
        tensorflow.keras.preprocessing.image.ImageDataGenerator
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        vertical_flip=True,
        horizontal_flip=True)
    
    train_generator = datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=VGG_SIZE,  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='categorical', 
        shuffle=True)
    
    val_generator = datagen.flow_from_directory(
        val_path,  # this is the target directory
        target_size=VGG_SIZE,  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='categorical', 
        shuffle=True)
    
    return train_generator, val_generator

def make_vgg_architecture(shape:tuple[int]=VGG_IMG_SHAPE, 
                          from_pretrained:bool=False) -> Model:
    """
    Create a model with vgg architecture.

    Parameters
    ----------
    shape : tuple[int], optional
        The shape of the images. By default they are (223,223,3)
    from_pretrained : bool, optional
        If True, loads the vgg16 weights from imagenet.
        Else weights are None. 

    Returns
    -------
    Model
        A model with vgg architecture
    """

    if from_pretrained:
        weights='imagenet'
    else:
        weights=None

    true_vgg = VGG16(input_shape=shape, weights=weights, 
                    include_top=False)

    for layer in true_vgg.layers:
        layer.trainable = False
    
    #Make new head
    flatten = Flatten()(true_vgg.output)
    d = Dense(4096, activation='relu')(flatten)
    drop = Dropout(0.5)(d)
    output = Dense(10, activation='softmax')(drop)

    #Transfer Learning
    model = Model(inputs=true_vgg.input, 
                outputs=output, 
                name='custom_vgg16')

    return model


def get_latest(weights_dir:str)->str:
    """
    Get the latest weights to load depending on epochs. 
    The epoch of saving must be in the name of the weights.h5 file.

    Parameters
    ----------
    weights_dir : str
        The path to the saved weights.

    Returns
    -------
    str
        The path to the latest weights
    """
    onlyfiles = [f for f in os.listdir(weights_dir) if os.path.isfile(os.path.join(weights_dir, f))]
    if len(onlyfiles) == 1:
        if "weights.h5" in onlyfiles[0]:
            return onlyfiles[0]
        raise Exception("No weights.h5")
    elif len(onlyfiles) == 0:
        raise Exception("No weights.h5")
    else:
        onlyfiles = [f for f in onlyfiles if ".weights.h5" in f]
        nb = [f.replace("vgg16-", "").split('.')[0] for f in onlyfiles]
        nb.remove("vgg16")
        nb = list(map(int, nb))
        #arg max
        index_max = max(enumerate(nb), key=lambda x: x[1])[0]
        return onlyfiles[index_max]

def load_model_from_weights(weights_dir:str, 
                            shape:tuple[int]=VGG_IMG_SHAPE, 
                            from_pretrained:bool=False) -> Model:
    """
    Make a VGG16 model from pretrained wieghts or from imagenet

    Parameters
    ----------
    weights_dir : str
        The path to the saved weights.
    shape : tuple[int], optional
        The shape of the input images. By default they are of shape (223, 223, 3)
    from_pretrained : bool, optional
        True get the weights from custom pretrained VGG16. 
        Else, from VGG16 imagenet.

    Returns
    -------
    Model
        VGG16 model. 
    """
    model = make_vgg_architecture(shape, from_pretrained)

    if from_pretrained:
        latest = get_latest(weights_dir)
        model.load_weights(os.path.join(weights_dir, latest))
    return model

def get_lr_from_epoch(from_epoch:int) -> float:
    """
    Retrieve the learning rate for a given epoch. 

    Parameters
    ----------
    from_epoch : int
        The epoch to actualise the learning rate.

    Returns
    -------
    float
        The learning rate actualised
    """
    n = from_epoch // 5
    return 0.001 * (0.9 ** float(n))

def fit_and_export(train_generator, validation_generator, 
                 save_path:str,
                 checkpoint_dir:str, epochs:int=20, 
                 shape:tuple[int]=VGG_IMG_SHAPE,
                 from_pretrained:bool=False, 
                 from_epoch:int=0)-> Model:
    """
    Fit and export a transfer-learning model based on 
    vgg16 model for the classification task.
    Returns the model.
     Parameters
    ----------
    train_generator : 
        The training dataset
    validation_generator :
        The validation dataset
    checkpoint_dir : str
        The path to save the model at each checkpoints. 
        It will save the wieghts in weights.h5 format.
    epochs : int, optional
        The number of training epochs. Default 20. 
    shape : tuple[int], optional
        The shape of the input images. By default they are of shape (223, 223, 3)
    from_pretrained : bool, optional
        True, load the model with latest weights in checkpoint.
        Else the weights are taken from VGG16 imagenet
    from_epoch : int, optional
        The epoch to resume training

    Returns
    -------
    Model
        The fitted model
    """
    if from_pretrained:
        model = load_model_from_weights(checkpoint_dir, shape, from_pretrained)
        if from_epoch == 0:
            raise Exception("You have to set an epoch")
    else:
        model = make_vgg_architecture(shape, False)

    optimizer = Adam(learning_rate=get_lr_from_epoch(from_epoch=from_epoch))

    model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer,
                        metrics=['categorical_accuracy'])
    
    checkpoint_path = os.path.join(checkpoint_dir, 'vgg16-{epoch:04d}.weights.h5')

    schedule = LearningRateScheduler(scheduler)
    chekpoint = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                save_best_only=True,
                                verbose=1)
    stopping = EarlyStopping(patience=20, min_delta=0.0001)

    #Fit from generator
    model.fit(train_generator, 
                validation_data=validation_generator,
                epochs=epochs, 
                callbacks=[schedule, chekpoint, stopping])
    #Export model
    model.export(
                save_path, 
                format='h5')

    return model


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "vgg":

            folder = "vgg_classification"
            save_path = os.path.join("src", "models", folder , "big", "model", "vgg_classification.h5")
            chekcpoint_dir = os.path.join("src", "models", folder , "big", "checkpoint")

            root_dir = os.path.join('data', folder, 'datasets')
            train_dir = os.path.join("data", folder, 'train')
            val_dir = os.path.join("data", folder, 'val')

            make_train_val_folder(root_dir, train_path=train_dir, val_path=val_dir)
            train_gen, val_gen = make_generator(train_path=train_dir, val_path=val_dir)

            fit_and_export(train_gen, val_gen, save_path, chekcpoint_dir)