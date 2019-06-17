from data_utils.hog import hog_skimage
import cv2
from models.svm import SVM
import numpy as np
from config import *
from data_utils.timer import Timer


def detect(input_path, model, hog=True):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        print("error: read image failed, please check input path")
        return

    H, W, C = image.shape
    faces = []
    steph = H // 10
    stepw = W // 10
    for x in range(0, H, steph):
        for y in range(0, W, stepw):
            for h in range(steph, H, steph):
                if x + h > H:
                    break
                for w in range(stepw, W, stepw):
                    if y + w > W:
                        break
                    crop = cv2.resize(image[x: x + h][y: y + w], (96, 96),
                                      interpolation=cv2.INTER_CUBIC)
                    if hog:
                        input = hog_skimage(crop)
                    else:
                        input = crop

                    if model.predict(input):
                        faces.append([x, y, h, w])

    if len(faces) == 0:
        print("no face dectected in %s" % input_path)
    else:
        for face in faces:
            cv2.rectangle(image, (face[0], face[1]),
                          (face[0] + face[2], face[1] + face[3]),
                          color=(255, 0, 0),
                          thickness=2)
        cv2.imshow(input_path, image)
        cv2.waitKey(0)


if __name__ == '__main__':

    # classification
    model = SVM(kernel='rbf', gamma='scale', probability=True)
    train_data = np.load(DATA_PATH + "\\hog\\train.npz")
    x, y = train_data['feature'], train_data['label']
    print("number of training sample = %d" % x.shape[0])
    model.train(x, y)
    timer = Timer()
    timer.start()
    timer.stop()
    test_data = np.load(DATA_PATH + "\\hog\\test.npz")
    xt, yt = test_data['feature'], test_data['label']
    acc = model.score(xt, yt)
    print("mean acc = {:.4f}".format(acc))

    # dectection
    with open(os.path.join(DATA_PATH, "FDDB_folds\\FDDB-fold-10.txt", 'r')) as f:
        for i in range(10):
            img_path = f.readline()
            img_path = os.path.join(DATA_PATH, "originalPics\\" + img_path)
            timer.start()
            detect(img_path, model, hog=True)
            timer.stop()








