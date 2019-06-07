import keras
from keras.layers import *
from config import *

def get_models(n_classes, input_height, input_width):
    
    if DATA_FORMAT == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif DATA_FORMAT == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation="relu", padding="same", data_format=DATA_FORMAT, name="conv1")(img_input)
    x = Conv2D(64, (3, 3), )