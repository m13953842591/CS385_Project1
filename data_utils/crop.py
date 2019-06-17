import cv2
import os
from matplotlib import pyplot as plt
import numpy as np 
from config import *


def crop(img, x, y, major_r, minor_r, positive=True):
    '''
    Generate the positive or negative feature image according to the annotation
    param img: the input image in numpy form
    the annotation is a elliptical regions bound specified by

    param x: center_x
    param y: center_y
    param major_r: major axis radius
    param minor_r: minor axis radius
    and angle, which is not necessity in this method

    param positive: generative positive images if set to True, else crop
    negative images[96, 96, 3]

    return: if positive=True, return the positive image
            else return nine negative images. [9, 96, 96, 3]
    '''
    width = int(minor_r * 8 / 3) # width of the bounding box
    height = int(major_r * 8 / 3) # height of the bounding box

    if positive:
        # find the bound to crop the image, bound may be negative, thus need 
        # padding
        left_bd = int(x - width / 2)
        right_bd = int(x + width / 2)
        top_bd = int(y - height / 2)
        bottom_bd = int(y + height / 2)
        # calculating the padding on each direction
        left = max(0, -left_bd)
        right = max(0, right_bd - img.shape[1])
        top = max(0, -top_bd)
        bottom = max(0, bottom_bd - img.shape[0])
        # padding
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        # corp the image in [l:l+width, t: t+height]
        l = int(x + left - width/2)
        t = int(y + top - height/2)
        pos_img = cv2.resize(img[t:t+height, l:l+width, :], (96, 96))
        return pos_img

    else:
        one_third_width = int(width * 1 / 3)
        one_third_height = int(height * 1 / 3)

        # same to positive condition, but we need to get 9 negative images
        left_bd = int(x - width * 5 / 6)
        right_bd = int(x + width * 5 / 6)
        top_bd = int(y - height * 5 / 6)
        bottom_bd = int(y + height * 5 / 6)
        left = max(0, -left_bd)
        right = max(0, right_bd - img.shape[1])
        top = max(0, -top_bd)
        bottom = max(0, bottom_bd - img.shape[0])

        # first negative image is just scaling the original image to [96, 96, 3] 
        neg_img = np.zeros((9, 96, 96, 3), dtype=np.uint8)
        neg_img[0] = cv2.resize(img, (96, 96))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        
        l = int (x + left - width * 5 / 6)
        t = int (y + top - height * 5 / 6)
        k = 1
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                if i == 1 and j == 1:
                    continue
                li = l + i * one_third_width
                tj = t + j * one_third_height
                neg_img[k] = cv2.resize(img[tj:tj+height, li:li+width], (96, 96))
                k += 1
        return neg_img


def show_img(img, positive=True):
    if positive:
        cv2.imshow("pos_img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        place = [335, 331, 334, 337, 332, 338, 333, 336, 339]
        for i in range(9):
            plt.subplot(place[i])
            subimg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
            plt.imshow(subimg)
        plt.show()


def test(count=50, isPositive=True):
    # if isPositive == True, test positive example
    # else test negative example
    # "count" represent number of pictures we test

    ellipseListpath = os.path.join(DATA_PATH + '/FDDB-folds', "FDDB-fold-01-ellipseList.txt")
    with open(ellipseListpath, 'r') as f:
        while count > 0:
            line = f.readline()
            if not line:
                break
            line = line[:-1] + ".jpg"
            img_path = os.path.join(DATA_PATH + '/originalPics', line)
            img = cv2.imread(img_path)
            num_face = int(f.readline()[:-1])
            for i in range(num_face):
                annot = f.readline().split(" ")
                face_img = crop(img, float(annot[3]), float(annot[4]), \
                                float(annot[0]), float(annot[1]), positive=isPositive)
                show_img(face_img, positive=isPositive)
            count -= 1


if __name__ == '__main__':
    test()
