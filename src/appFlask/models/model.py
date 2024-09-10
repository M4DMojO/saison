from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from ultralytics import YOLO

from custom_function import get_all_weights_from_bucket

import cv2
import numpy as np

import os

VGG_IMG_SHAPE = (224, 224, 3)

def vgg_img_prepro(img):
    img = cv2.resize(img, (VGG_IMG_SHAPE[0], VGG_IMG_SHAPE[1]))
    return np.expand_dims(img, axis=0).astype(np.float32)

def make_vgg_pred(model:Model, img, shape:tuple[int]=None):
    res = model.predict(vgg_img_prepro(img))
    if shape != None:
        x1, y1, x2, y2 = shape
    else:
        x1, y1 = 0, 0
        x2, y2 = img.shape

    return {
            'fruit_id': res.argmax(),
            'confidence': np.max(res),
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2
        }

class YOLOToVGG():
    """
    A model that combines the object detection from yolo 
    and classification from vgg16.
    """
    def __init__(self, yolo:YOLO, vgg:Model) -> None:
        self.yolo = yolo
        self.vgg = vgg
    
    def predict(self, img:str) -> list[dict]:
        """
        Predict the class and bouding boxes of an image
        Args:
            img (str): The path to the image to predict

        Returns:
            list[dict]: list containing the outputs
        """
        cv2_img = cv2.imread(img)
        results = self.yolo(img)
        boxes = [x.boxes for x in results[0]]
        outputs= []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])  # Coordonnées de la bounding box
            cropped_img = cv2_img[x1:x2, y1:y2]
            res = make_vgg_pred(self.vgg, 
                                cropped_img,
                                (x1, y1, x2, y2))
            outputs.append(res)
            
        return outputs
    
    def __call__(self, img:str) -> list[dict]:
        """
        Call YOLOToVGG.predict
        Args:
            img (str): Path to the image to predict

        Returns:
            list[dict]: list containing the outputs
        """
        return self.predict(img)


def make_vgg_architecture(shape:tuple[int]) -> Model:
    """
    Make a model with vgg16 architecture,
    custom head and no loaded weights
    Args:
        shape (tuple[int]): size of input images

    Returns:
        Model: vgg16 model architecture
    """
    true_vgg = VGG16(input_shape=shape, weights=None, 
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


def load_vgg_from_weights(weight_path:str)-> Model:
    """
    Load the weights into a vgg16 model
    Args:
        weight_path (str): Path to the weights

    Returns:
        Model: Model with custom vgg16 architecture and weights
    """
    model = make_vgg_architecture(VGG_IMG_SHAPE)
    model.load_weights(weight_path)
    return model

def load_models() -> list:
    """Loads and returns the model used for the app

    Raises:
        NumberModelsError: 
            occurs when on of the weight is not in the model folder

    Returns:
        list: List of models
    """
    files = [f for f in os.listdir('../models/') if os.path.isfile(f)]
    if files < 4:
        get_all_weights_from_bucket()
    
    yolo_total = YOLO('../models/yolo_total.pt')
    yolo_seg = YOLO('../models/yolo_segmentation.pt')
    vgg_seg = load_vgg_from_weights('../models/vgg_classification_small.h5')
    combined_model = YOLOToVGG(yolo_seg, vgg_seg)
    vgg_cls = load_vgg_from_weights('../models/vgg_classification_big.weights.h5')
    
    return [yolo_total,
            combined_model,
            vgg_cls]