from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from ultralytics import YOLO

from src.appFlask.src.custom_function import get_all_weights_from_bucket

import cv2
import numpy as np

import os

VGG_IMG_SHAPE = (224, 224, 3)

def vgg_img_prepro(img)-> np.array:
    """
    preprocess the image to vgg16 format
    Args:
        img : cv2 image | np.array

    Returns:
        np.array : image preprocessed
    """
    img = cv2.resize(img, (VGG_IMG_SHAPE[0], VGG_IMG_SHAPE[1]))
    return np.expand_dims(img, axis=0).astype(np.float32)

def make_vgg_pred(model:Model, img:str, shape:tuple[int]=None) -> list[dict]:
    """
    Make the prediction with a given vgg16 model
    Args:
        model (Model): vgg16 model
        img (str): path to image
        shape (tuple[int], optional): shape of the image/bouding boxe

    Returns:
        list[dict]: list of outputs
    """
    img = cv2_img = cv2.imread(img)
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
    print("loading vgg from: ", weight_path)
    return load_model(weight_path)

def remake_vgg():
    base_path = os.path.join("src", "appFlask", "models")
    model = make_vgg_architecture(VGG_IMG_SHAPE)
    model.load_weights(os.path.join(base_path, "vgg_classification_big.weights.h5"))
    model.save(os.path.join(base_path, "vgg_classification_big.keras"))

def load_models() -> list:
    """Loads and returns the model used for the app

    Raises:
        NumberModelsError: 
            occurs when on of the weight is not in the model folder

    Returns:
        list: List of models
    """
    base_path = "src/appFlask/models"
    files = [f for f in os.listdir('src/appFlask/models/') if os.path.isfile(f) and ("h5" in f or "pt" in f or "keras" in f)]
    if len(files) < 4:
        print("--"*10)
        print("loading files from bucket")
        get_all_weights_from_bucket()
        print("load from bucket done")
        print("--"*10)

    yolo_total = YOLO(os.path.join(base_path, "yolo_total.pt"))
    yolo_seg = YOLO(os.path.join(base_path, 'yolo_segmentation.pt'))
    vgg_seg = load_vgg_from_weights(os.path.join(base_path, 'vgg_classification_small.keras'))
    combined_model = YOLOToVGG(yolo_seg, vgg_seg)
    vgg_cls = load_vgg_from_weights(os.path.join(base_path, 'vgg_classification_big.keras'))
    
    return [yolo_total,
            combined_model,
            vgg_cls]

def get_results(model:Model|YOLO|YOLOToVGG, 
                img:str, 
                index:int) -> list[dict]:
    """
    Get the prediction result for a given model
    Args:
        model: Model
        img (str): Path to the image to predict
        index (int): index of the model in the Flask app cache

    Raises:
        Exception: if the index of the model is too big

    Returns:
        list[dict]: List of outputs
    """
    if index == 0:#yolo
        return model(img)
    elif index == 1: #yolo + vgg
        return model(img)
    elif index == 2:#vgg
        return make_vgg_pred(model, img)
    else:
        raise Exception(f"No such model at index {index} error.")