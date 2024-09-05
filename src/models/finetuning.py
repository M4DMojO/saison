from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image

import shutil
import os
import sys

VGG_IMG_SHAPE = (224, 224, 3)
VGG_SIZE = (224, 224)
BATCH_SIZE = 64


def scheduler(epoch, lr):
    if epoch % 5:
        return lr * 0.9
    return lr

def make_train_val_folder(root:str, train_path:str, val_path:str, 
                           nb_img:int=400, val_split:int=0.2):
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
    datagen = ImageDataGenerator(
        rescale=1./255,
        vertical_flip=True,
        horizontal_flip=True)
    
    train_generator = datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=VGG_SIZE,  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    val_generator = datagen.flow_from_directory(
        val_path,  # this is the target directory
        target_size=VGG_SIZE,  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    return train_generator, val_generator



def fit_and_export(train_generator, validation_generator, 
                 save_path:str,
                 checkpoint_path:str, epochs:int=20, 
                 shape:tuple[int]=VGG_IMG_SHAPE )-> Model:
    """
    Fit and export a transfer-learning model based on 
    vgg16 model for the classification task.
    Returns the model.
    """

    #Load vgg16
    true_vgg = VGG16(input_shape=shape, weights='imagenet', 
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

    model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['categorical_accuracy'])
    

    schedule = LearningRateScheduler(scheduler)
    chekpoint = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                verbose=1)

    #Fit from generator
    model.fit(train_generator, validation_data=validation_generator,
                        epochs=epochs, 
                        callbacks=[schedule, chekpoint])
    #Export model
    model.export(
                save_path, 
                format='tf_saved_model')

    return model


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "vgg":

            folder = "vgg_classification"
            save_path = os.path.join("src", "models", folder , "big", "model", "vgg16.weights.h5")
            chekcpoint_path = os.path.join("src", "models", folder , "big", "checkpoint", 'vgg16.weights.h5')

            root_dir = os.path.join('data', folder, 'datasets')
            train_dir = os.path.join("data", folder, 'train')
            val_dir = os.path.join("data", folder, 'val')

            make_train_val_folder(root_dir, train_path=train_dir, val_path=val_dir)
            train_gen, val_gen = make_generator(train_path=train_dir, val_path=val_dir)

            fit_and_export(train_gen, val_gen, save_path, chekcpoint_path)