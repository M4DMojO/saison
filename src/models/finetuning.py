from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.train import latest_checkpoint

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

def make_vgg_architecture() -> Model:

    input = Input(VGG_IMG_SHAPE, name="input")
    #block1
    block1_c1 = Conv2D(64, kernel_size=(3,3), activation='relu')(input)
    block1_c2 = Conv2D(64, kernel_size=(3,3), activation='relu')(block1_c1)
    pool1 = MaxPool2D(pool_size=(2,2))(block1_c2)

    #block2
    block2_c1 = Conv2D(128, kernel_size=(3,3), activation='relu')(pool1)
    block2_c2 = Conv2D(128, kernel_size=(3,3), activation='relu')(block2_c1)
    pool2 = MaxPool2D(pool_size=(2,2))(block2_c2)

    #block3
    block3_c1 = Conv2D(256, kernel_size=(3,3), activation='relu')(pool2)
    block3_c2 = Conv2D(256, kernel_size=(3,3), activation='relu')(block3_c1)
    block3_c3 = Conv2D(256, kernel_size=(3,3), activation='relu')(block3_c2)
    pool3 = MaxPool2D(pool_size=(2,2))(block3_c3)

    #block4
    block4_c1 = Conv2D(512, kernel_size=(3,3), activation='relu')(pool3)
    block4_c2 = Conv2D(512, kernel_size=(3,3), activation='relu')(block4_c1)
    block4_c3 = Conv2D(512, kernel_size=(3,3), activation='relu')(block4_c2)
    pool4 = MaxPool2D(pool_size=(2,2))(block4_c3)

    #block5
    block5_c1 = Conv2D(512, kernel_size=(3,3), activation='relu')(pool4)
    block5_c2 = Conv2D(512, kernel_size=(3,3), activation='relu')(block5_c1)
    block5_c3 = Conv2D(512, kernel_size=(3,3), activation='relu')(block5_c2)
    pool5 = MaxPool2D(pool_size=(2,2))(block5_c3)

    #head
    flatten = Flatten()(pool5)
    d = Dense(4096, activation='relu')(flatten)
    drop = Dropout(0.5)(d)
    output = Dense(10, activation='softmax')(drop)

    model = Model(inputs=input, outputs=output) 

    return model

def load_model_from_weights(weights_dir:str):

    model = make_vgg_architecture()
    latest = latest_checkpoint(weights_dir)
    model.load_weights(latest)

    return model



def fit_and_export(train_generator, validation_generator, 
                 save_path:str,
                 checkpoint_dir:str, epochs:int=20, 
                 shape:tuple[int]=VGG_IMG_SHAPE,
                 from_pretrained:bool=False)-> Model:
    """
    Fit and export a transfer-learning model based on 
    vgg16 model for the classification task.
    Returns the model.
    """

    if from_pretrained:
        model = load_model_from_weights(checkpoint_dir)

    else:
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
    
    checkpoint_path = os.path.join(checkpoint_dir, 'vgg16-{epoch:04d}.weights.h5')

    schedule = LearningRateScheduler(scheduler)
    chekpoint = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                verbose=1)

    #Fit from generator
    model.fit(train_generator, 
                validation_data=validation_generator,
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
            save_path = os.path.join("src", "models", folder , "big", "model", "vgg_classification.weights.h5")
            chekcpoint_dir = os.path.join("src", "models", folder , "big", "checkpoint")

            root_dir = os.path.join('data', folder, 'datasets')
            train_dir = os.path.join("data", folder, 'train')
            val_dir = os.path.join("data", folder, 'val')

            make_train_val_folder(root_dir, train_path=train_dir, val_path=val_dir)
            train_gen, val_gen = make_generator(train_path=train_dir, val_path=val_dir)

            fit_and_export(train_gen, val_gen, save_path, chekcpoint_dir)