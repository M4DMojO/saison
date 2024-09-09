from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model


VGG_IMG_SHAPE = (224, 224, 3)

def make_vgg_architecture(shape:tuple[int]) -> Model:

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
    model = make_vgg_architecture(VGG_IMG_SHAPE)
    model.load_weights(weight_path)
    return model