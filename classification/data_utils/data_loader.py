import numpy as np
import cv2
import glob
import os
import random
import itertools
from config import *


def get_pairs_from_path(images_path):
    print("loading dataset, please wait...")
    images = glob.glob(os.path.join(images_path, "*.jpg"))
    ret = []
    for im in images:
        if int(im.split('.')[-2]) == 0:
            ret.append((im, 1))
        else:
            ret.append((im, 0))
    print("finish loading!")
    return ret


def get_image_arr(path, width, height, img_norm="sub_mean", ordering='channels_first'):
    if type(path) is np.ndarray:
        img = path
    else:
        img = cv2.imread(path, 1)

    if img_norm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif img_norm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif img_norm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def image_generator(images_path, batch_size, input_height, input_width):
    img_label_pairs = get_pairs_from_path(images_path)
    random.shuffle(img_label_pairs)
    zipped = itertools.cycle(img_label_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img, label = next(zipped)
            img = get_image_arr(img, input_width, input_height, ordering=DATA_FORMAT)
            X.append(img)
            y = np.zeros((2, ))
            y[label] = 1
            Y.append(y)

        yield np.array(X), np.array(Y)
