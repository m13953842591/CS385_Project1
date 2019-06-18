from data_utils.hog import hog_skimage
import cv2
from models.logistic import LogisticModel
import numpy as np
from config import *
from data_utils.timer import Timer
from matplotlib import pyplot as plt

def get_iou(pt1, pt2):

    inner_left = pt1[1] if pt1[1] > pt2[1] else pt2[1]
    inner_right = pt1[1] + pt1[3] if pt1[1] + pt1[3] < pt2[1] + pt2[3] else pt2[1] + pt2[3]
    inner_top = pt1[0] + pt1[2] if pt1[0] + pt1[2] < pt2[0] + pt2[2] else pt2[0] + pt2[2]
    inner_bottom = pt1[0] if pt1[0] > pt2[0] else pt2[0]

    # 计算内部交叉区域面积，要考虑两个矩形可能不相交的情况
    inner_width = (inner_top - inner_bottom) if inner_top > inner_bottom else 0
    inner_height = (inner_right - inner_left) if inner_right > inner_left else 0
    inner_area = inner_width * inner_height

    return float(inner_area) / (pt1[3]*pt1[2] + pt2[3]*pt2[2] - inner_area)


def detect(input_path, model, hog=True, threshold=0.99, overlap=0.2):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    image_name = input_path.split('/')[-1]
    if image is None:
        print("error: read image failed, please check input path")
        return

    H, W, C = image.shape
    faces = []
    probas = []
    steph = H // 20
    stepw = W // 20
    max_prob = 0
    for x in range(0, int(0.5*H), steph):
        for y in range(0, int(0.5*W), stepw):
            for h in range(steph, int(0.5*H), steph):
                if x + h > H:
                    break
                for w in range(stepw, int(0.5*W), stepw):
                    if h > 2*w:
                        continue
                    if w > 1.5*h:
                        break
                    if y + w > W:
                        break
                    crop = cv2.resize(image[x: x + h, y: y + w, :], (96, 96))

                    if hog:
                        input = hog_skimage(crop)
                    else:
                        input = crop

                    proba = model.predict_proba(input)

                    if proba > threshold:
                        # probably detected a new face
                        if faces:
                            # compare with the last detected face
                            xt, yt, ht, wt = faces[-1]
                            if get_iou([x, y, h, w], [xt, yt, ht, wt]) > overlap:
                                # get a lot of overlaps
                                if proba > probas[-1]:
                                    # the new face is better discard the old face
                                    faces.pop()
                                    probas.pop()
                                else:
                                    continue
                        # add the new face
                        faces.append([x, y, h, w])
                        probas.append(proba)

    if len(faces) == 0:
        print("no face dectected in %s" % input_path)
        print("max probability detected", max_prob)
    else:
        for face in faces:
            cv2.rectangle(image, (face[1], face[0]),
                          (face[1] + face[3], face[0] + face[2]),
                          color=(0, 0, 255),
                          thickness=1)
        cv2.imwrite(os.path.join(WORKSPACE, "images\\detection\\%s_%s" % (
                                            model.name, image_name)), image)


if __name__ == '__main__':
    timer = Timer()
    from models import *
    model = CNN()

    # detect
    with open(os.path.join(DATA_PATH, "FDDB-folds\\FDDB-fold-10.txt"), 'r') as f:

        for i in range(20):
            img_path = f.readline().strip() + ".jpg"
            img_path = os.path.join(DATA_PATH, "originalPics\\" + img_path)
            timer.start()
            detect(img_path, model, hog=False, threshold=0.99, overlap=0.2)
            timer.stop()








