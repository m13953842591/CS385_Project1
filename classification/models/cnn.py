from keras.models import *
from keras.layers import *
from config import DATA_FORMAT


def get_cnn(input_height=227, input_width=227):

    img_input = Input(shape=(input_height, input_width, 3))

    if DATA_FORMAT == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))

    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='valid', name='conv1', data_format=DATA_FORMAT)(img_input)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='valid', name='conv2', data_format=DATA_FORMAT)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv3', data_format=DATA_FORMAT)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv4', data_format=DATA_FORMAT)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv5', data_format=DATA_FORMAT)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    o = Dense(2, activation='softmax', name='fc8')(x)

    model = Model(img_input, o)
    model.model_name = "cnn_alexnet"
    model.summary()
    return model


def get_vgg16(input_height=96, input_width=96):
    img_input = Input(shape=(input_height, input_width, 3))

    if DATA_FORMAT == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=DATA_FORMAT)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=DATA_FORMAT)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=DATA_FORMAT)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=DATA_FORMAT)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=DATA_FORMAT)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=DATA_FORMAT)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=DATA_FORMAT)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=DATA_FORMAT)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=DATA_FORMAT)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=DATA_FORMAT)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=DATA_FORMAT)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=DATA_FORMAT)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=DATA_FORMAT)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=DATA_FORMAT)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=DATA_FORMAT)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=DATA_FORMAT)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=DATA_FORMAT)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=DATA_FORMAT)(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='block6_fc')(x)
    x = Dropout(rate=0.5, name='block6_drop')(x)

    x = Dense(4096, activation='relu', name='block7_fc')(x)
    x = Dropout(rate=0.5, name='block7_drop')(x)

    o = Dense(2, activation='softmax')(x)

    model = Model(img_input, o)
    model.model_name = "cnn"
    model.summary()
    return model